# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch import nn
import torch.nn.functional as F
from ..builder import RECOGNIZERS
from .base import BaseRecognizer
from .. import builder
from einops import rearrange
from .recognizer2d import Recognizer2D
from torch.nn.functional import normalize
import torch.distributed as dist
from torchvision import transforms

#------------------------------
import numpy as np
import torch
import torch.nn.functional as F
import random

def write_log(log, log_path):
    f = open(log_path, mode='a')
    f.write(str(log))
    f.write('\n')
    f.close()


def fix_python_seed(seed):
    print('seed-----------python', seed)
    random.seed(seed)
    np.random.seed(seed)


def fix_torch_seed(seed):
    print('seed-----------torch', seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def fix_all_seed(seed):
    print('seed-----------all device', seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5*(1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)

    return total_kld

def optimize_beta(beta, MI_loss,alpha2=1e-3):
    beta_new = max(0, beta + alpha2 * (MI_loss - 1) )

    # return the updated beta value:
    return beta_new

def project_l2_ball(z):
    """ project the vectors in z onto the l2 unit norm ball"""
    return z / np.maximum(np.sqrt(np.sum(z**2, axis=1))[:, np.newaxis], 1)


def slerp(low, high, batch):
    low = low.repeat(batch, 1)
    high = high.repeat(batch, 1)
    val = ((0.6 - 0.4) * torch.rand(batch,) + 0.4).cuda()
    omega = torch.acos((low*high).sum(1))
    so = torch.sin(omega)
    res = (torch.sin((1.0-val)*omega)/so).unsqueeze(1)*low + (torch.sin(val*omega)/so).unsqueeze(1)*high
    return res


def get_source_centroid(feature, label, num_class, flag=True, centroids=None):
    new_centroid = torch.zeros(num_class, feature.size(1)).cuda()

    dist = 0
    for i in range(num_class):
        class_mask = (label == i)

        if flag:
            centroids = centroids.cuda()
            new_centroid[i] = 0.5 * torch.mean(feature[class_mask], dim=0) + 0.5 * centroids[i]

        else:
            new_centroid[i] = torch.mean(feature[class_mask], dim=0)
    dist += torch.mean(torch.nn.functional.pairwise_distance(feature[class_mask], new_centroid[i]))
    return new_centroid, dist

def optimize_beta(beta, distance,alpha2=0.5):
    beta_new = min(1, beta + alpha2 * distance )

    # return the updated beta value:
    return beta_new

def get_domain_vector_avg(feature, prototype, label, num_class):
    dist = torch.zeros(num_class).cuda()
    for i in range(num_class):
        class_feature = feature[label == i]
        dist[i] = torch.sqrt((prototype[i] - class_feature).pow(2).sum(1)).mean()
    return dist.mean() + dist.var()


def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = int(source.size()[0])+int(target.size()[0])
    total = torch.cat([source, target], dim=0)
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    L2_distance = ((total0-total1)**2).sum(2)
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)#/len(kernel_val)


def mmd_rbf(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None, ver=2):
    batch_size = int(source.size()[0])
    kernels = guassian_kernel(source, target, kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)

    loss = 0

    if ver == 1:
        for i in range(batch_size):
            s1, s2 = i, (i + 1) % batch_size
            t1, t2 = s1 + batch_size, s2 + batch_size
            loss += kernels[s1, s2] + kernels[t1, t2]
            loss -= kernels[s1, t2] + kernels[s2, t1]
        loss = loss.abs_() / float(batch_size)
    elif ver == 2:
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        loss = torch.mean(XX + YY - XY - YX)
    else:
        raise ValueError('ver == 1 or 2')

    return loss

def conditional_mmd_rbf(source, target, label, num_class, kernel_mul=2.0, kernel_num=5, fix_sigma=None, ver=2):
    loss = 0
    for i in range(num_class):
        source_i = source[label==i]
        target_i = target[label==i]
        loss += mmd_rbf(source_i, target_i)
    return loss / num_class

def domain_mmd_rbf(source, target, num_domain, d_label, kernel_mul=2.0, kernel_num=5, fix_sigma=None, ver=2):
    loss = 0
    loss_overall = mmd_rbf(source, target)
    for i in range(num_domain):
        source_i = source[d_label == i]
        target_i = target[d_label == i]
        loss += mmd_rbf(source_i, target_i)
    return loss_overall - loss / num_domain

def domain_conditional_mmd_rbf(source, target, num_domain, d_label, num_class, c_label):
    loss = 0
    for i in range(num_class):
        source_i = source[c_label == i]
        target_i = target[c_label == i]
        d_label_i = d_label[c_label == i]
        loss_c = mmd_rbf(source_i, target_i)
        loss_d = 0
        for j in range(num_domain):
            source_ij = source_i[d_label_i == j]
            target_ij = target_i[d_label_i == j]
            loss_d += mmd_rbf(source_ij, target_ij)
        loss += loss_c - loss_d / num_domain

    return loss / num_class

def DAN_Linear(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    batch_size = int(source.size()[0])
    kernels = guassian_kernel(source, target,
        kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)

    # Linear version
    loss = 0
    for i in range(batch_size):
        s1, s2 = i, (i+1)%batch_size
        t1, t2 = s1+batch_size, s2+batch_size
        loss += kernels[s1, s2] + kernels[t1, t2]
        loss -= kernels[s1, t2] + kernels[s2, t1]
    return loss / float(batch_size)

def mmd_linear(src_fea, tar_fea):
    delta = (src_fea - tar_fea).squeeze(0)
    loss = torch.pow(torch.mean(torch.mm(delta, torch.transpose(delta, 0, 1))),2)
    return torch.sqrt(loss)

def diverse_conditional_mmd(source, target, label, num_class, iter, d_label=None, num_domain=3):
    loss = 0
    selected_d = iter % num_domain
    for i in range(num_class):
        source_i = source[label == i]
        target_i = target[label == i]
        d_label_i = d_label[label == i]

        source_is = source_i[d_label_i == selected_d]
        target_is = target_i[d_label_i == selected_d]

        source_iu = source_i[d_label_i != selected_d]
        target_iu = target_i[d_label_i != selected_d]

        if source_is.size(0) > 0 and source_iu.size(0) > 0:
            loss += (mmd_rbf(source_iu, target_iu) - 0.4 * mmd_rbf(source_is, target_is))

    return torch.clamp_min_(loss / num_class, 0)


def entropy_loss(x):
    out = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
    out = -1.0 * out.sum(dim=1)
    return out.mean()

def reparametrize(mu, logvar, factor=0.2):
    std = logvar.div(2).exp()
    eps = std.data.new(std.size()).normal_()
    return mu + factor*std*eps

def loglikeli(mu, logvar, y_samples):
    return (-(mu - y_samples)**2 /logvar.exp()-logvar).mean()#.sum(dim=1).mean(dim=0)

def club(mu, logvar, y_samples):

    sample_size = y_samples.shape[0]
    # random_index = torch.randint(sample_size, (sample_size,)).long()
    random_index = torch.randperm(sample_size).long()

    positive = - (mu - y_samples) ** 2 / logvar.exp()
    negative = - (mu - y_samples[random_index]) ** 2 / logvar.exp()
    upper_bound = (positive.sum(dim=-1) - negative.sum(dim=-1)).mean()
    return upper_bound / 2.
#------------------------------

class AugNet(nn.Module):
    def __init__(self, noise_lv):
        super(AugNet, self).__init__()
        ############# Trainable Parameters
        self.noise_lv = nn.Parameter(torch.zeros(1))
        self.shift_var = nn.Parameter(torch.empty(3,216,216))
        nn.init.normal_(self.shift_var, 1, 0.1)
        self.shift_mean = nn.Parameter(torch.zeros(3, 216, 216))
        nn.init.normal_(self.shift_mean, 0, 0.1)

        self.shift_var2 = nn.Parameter(torch.empty(3, 212, 212))
        nn.init.normal_(self.shift_var2, 1, 0.1)
        self.shift_mean2 = nn.Parameter(torch.zeros(3, 212, 212))
        nn.init.normal_(self.shift_mean2, 0, 0.1)

        self.shift_var3 = nn.Parameter(torch.empty(3, 208, 208))
        nn.init.normal_(self.shift_var3, 1, 0.1)
        self.shift_mean3 = nn.Parameter(torch.zeros(3, 208, 208))
        nn.init.normal_(self.shift_mean3, 0, 0.1)

        self.shift_var4 = nn.Parameter(torch.empty(3, 220, 220))
        nn.init.normal_(self.shift_var4, 1, 0.1)
        self.shift_mean4 = nn.Parameter(torch.zeros(3, 220, 220))
        nn.init.normal_(self.shift_mean4, 0, 0.1)

        self.norm = nn.InstanceNorm2d(3)

        ############## Fixed Parameters (For MI estimation
        self.spatial = nn.Conv2d(3, 3, 9).cuda()
        self.spatial_up = nn.ConvTranspose2d(3, 3, 9).cuda()

        self.spatial2 = nn.Conv2d(3, 3, 13).cuda()
        self.spatial_up2 = nn.ConvTranspose2d(3, 3, 13).cuda()

        self.spatial3 = nn.Conv2d(3, 3, 17).cuda()
        self.spatial_up3 = nn.ConvTranspose2d(3, 3, 17).cuda()


        self.spatial4 = nn.Conv2d(3, 3, 5).cuda()
        self.spatial_up4 = nn.ConvTranspose2d(3, 3, 5).cuda()


        self.color = nn.Conv2d(3, 3, 1).cuda()

        for param in list(list(self.color.parameters()) +
                          list(self.spatial.parameters()) + list(self.spatial_up.parameters()) +
                          list(self.spatial2.parameters()) + list(self.spatial_up2.parameters()) +
                          list(self.spatial3.parameters()) + list(self.spatial_up3.parameters()) +
                          list(self.spatial4.parameters()) + list(self.spatial_up4.parameters())
                          ):
            param.requires_grad=False

    def forward(self, x, estimation=False):
        if not estimation:
            spatial = nn.Conv2d(3, 3, 9).cuda()
            spatial_up = nn.ConvTranspose2d(3, 3, 9).cuda()

            spatial2 = nn.Conv2d(3, 3, 13).cuda()
            spatial_up2 = nn.ConvTranspose2d(3, 3, 13).cuda()

            spatial3 = nn.Conv2d(3, 3, 17).cuda()
            spatial_up3 = nn.ConvTranspose2d(3, 3, 17).cuda()

            spatial4 = nn.Conv2d(3, 3, 5).cuda()
            spatial_up4 = nn.ConvTranspose2d(3, 3, 5).cuda()


            color = nn.Conv2d(3,3,1).cuda()
            weight = torch.randn(5)

            x = x + torch.randn_like(x) * self.noise_lv * 0.01
            x_c = torch.tanh(F.dropout(color(x), p=.2))

            x_sdown = spatial(x)
            x_sdown = self.shift_var * self.norm(x_sdown) + self.shift_mean
            x_s = torch.tanh(spatial_up(x_sdown))
            #
            x_s2down = spatial2(x)
            x_s2down = self.shift_var2 * self.norm(x_s2down) + self.shift_mean2
            x_s2 = torch.tanh(spatial_up2(x_s2down))
            #
            #
            x_s3down = spatial3(x)
            x_s3down = self.shift_var3 * self.norm(x_s3down) + self.shift_mean3
            x_s3 = torch.tanh(spatial_up3(x_s3down))

            #
            x_s4down = spatial4(x)
            x_s4down = self.shift_var4 * self.norm(x_s4down) + self.shift_mean4
            x_s4 = torch.tanh(spatial_up4(x_s4down))

     

            output = (weight[0] * x_c + weight[1] * x_s + weight[2] * x_s2+ weight[3] * x_s3 + weight[4]*x_s4) / weight.sum()
        else:
            x = x + torch.randn_like(x) * self.noise_lv * 0.01
            x_c = torch.tanh(self.color(x))
            #
            x_sdown = self.spatial(x)
            x_sdown = self.shift_var * self.norm(x_sdown) + self.shift_mean
            x_s = torch.tanh(self.spatial_up(x_sdown))
            #
            x_s2down = self.spatial2(x)
            x_s2down = self.shift_var2 * self.norm(x_s2down) + self.shift_mean2
            x_s2 = torch.tanh(self.spatial_up2(x_s2down))

            x_s3down = self.spatial3(x)
            x_s3down = self.shift_var3 * self.norm(x_s3down) + self.shift_mean3
            x_s3 = torch.tanh(self.spatial_up3(x_s3down))

            x_s4down = self.spatial4(x)
            x_s4down = self.shift_var4 * self.norm(x_s4down) + self.shift_mean4
            x_s4 = torch.tanh(self.spatial_up4(x_s4down))



            output = (x_c + x_s + x_s2 + x_s3 + x_s4) / 5
        return output



def concat_all_gather(tensor, dim=0):
	"""
	Performs all_gather operation on the provided tensors.
	*** Warning ***: torch.distributed.all_gather has no gradient.
	"""
	tensors_gather = [torch.ones_like(tensor)
		for _ in range(torch.distributed.get_world_size())]
	torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
	tensors_gather[dist.get_rank()] = tensor
	output = torch.cat(tensors_gather, dim=dim)
	return output



@RECOGNIZERS.register_module()
class L2DRecognizer2D(Recognizer2D):
    def __init__(self,
                 backbone,
                 cls_head=None,
                 contrastive_loss=None, # supcon loss
                 train_cfg=None,
                 test_cfg=None):
        super().__init__(backbone=backbone, cls_head=cls_head, train_cfg=train_cfg, test_cfg=test_cfg)
        # record the source of the backbone
        self.backbone_from = 'mmaction2'
        if backbone['type'].startswith('mmcls.'):
            try:
                import mmcls.models.builder as mmcls_builder
            except (ImportError, ModuleNotFoundError):
                raise ImportError('Please install mmcls to use this backbone.')
            backbone['type'] = backbone['type'][6:]
            self.backbone = mmcls_builder.build_backbone(backbone)
            self.backbone_from = 'mmcls'
        elif backbone['type'].startswith('torchvision.'):
            try:
                import torchvision.models
            except (ImportError, ModuleNotFoundError):
                raise ImportError('Please install torchvision to use this '
                                  'backbone.')
            backbone_type = backbone.pop('type')[12:]
            self.backbone = torchvision.models.__dict__[backbone_type](
                **backbone)
            # disable the classifier
            self.backbone.classifier = nn.Identity()
            self.backbone.fc = nn.Identity()
            self.backbone_from = 'torchvision'
        elif backbone['type'].startswith('timm.'):
            try:
                import timm
            except (ImportError, ModuleNotFoundError):
                raise ImportError('Please install timm to use this '
                                  'backbone.')
            backbone_type = backbone.pop('type')[5:]
            # disable the classifier
            backbone['num_classes'] = 0
            self.backbone = timm.create_model(backbone_type, **backbone)
            self.backbone_from = 'timm'
        else:
            self.backbone = builder.build_backbone(backbone)

        self.cls_head = builder.build_head(cls_head) if cls_head else None

        if contrastive_loss:
            self.sup_contrastive_loss = builder.build_loss(contrastive_loss)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        # aux_info is the list of tensor names beyond 'imgs' and 'label' which
        # will be used in train_step and val_step, data_batch should contain
        # these tensors
        # max_testing_views should be int
        self.max_testing_views = None
        if test_cfg is not None and 'max_testing_views' in test_cfg:
            self.max_testing_views = test_cfg['max_testing_views']
            assert isinstance(self.max_testing_views, int)

        if test_cfg is not None and 'feature_extraction' in test_cfg:
            self.feature_extraction = test_cfg['feature_extraction']
        else:
            self.feature_extraction = False

        # mini-batch blending, e.g. mixup, cutmix, etc.
        self.blending = None
        if train_cfg is not None and 'blending' in train_cfg:
            from mmcv.utils import build_from_cfg
            from mmaction.datasets.builder import BLENDINGS
            self.blending = build_from_cfg(train_cfg['blending'], BLENDINGS)

        self.init_weights()

        self.fp16_enabled = True

        self.convertor = AugNet(1).cuda() #augnet
        self.p_logvar = nn.Sequential(nn.Linear(2048, 2048), # paper it is 512
                                      nn.ReLU())
        self.p_mu = nn.Sequential(nn.Linear(2048 , 2048),# paper it is 512
                                  nn.LeakyReLU())
        

    def forward(self, imgs, label=None, return_loss=True, **kwargs):
        """Define the computation performed at every call."""
        if kwargs.get('gradcam', False):
            del kwargs['gradcam']
            return self.forward_gradcam(imgs, **kwargs)
        if return_loss:
            if label is None:
                raise ValueError('Label should not be None.')
            if self.blending is not None:
                imgs, label = self.blending(imgs, label)

            imgs_pathway_A= imgs
            return self.forward_train(imgs_pathway_A, label, **kwargs)

        return self.forward_test(imgs, **kwargs)


    def forward_train(self, imgs_pathway_A, labels, **kwargs):
        """Defines the computation performed at every call when training."""
        losses = dict()
        tran = transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        data = imgs_pathway_A
        
        class_l = labels
        batches = data.shape[0] # 12
        num_segs = data.shape[0] // batches # 8 
        # before reshape -  torch.Size([12, 8, 3, 224, 224])
        print('before reshape - ', data.shape)
        data = data.reshape((-1, ) + data.shape[2:])
        print('class_l- before squeeze - ', class_l.shape)
        print('data - ', data.shape)
   
        class_l = class_l.squeeze()
        print('class_l  after squeeze - ', class_l.shape)
        # ^ torch.Size([12])
        class_l_for_loss = torch.cat([class_l, class_l])
        list_class_l = class_l.tolist()
        all_frame_label = []
        for i in list_class_l:
            inflated = [i] * 8
            for z in inflated:
                all_frame_label.append(z)
        class_l = torch.tensor(all_frame_label)
     
        

        inputs_max = tran(torch.sigmoid(self.convertor(data)))
        inputs_max = inputs_max * 0.6 + data * 0.4
        data_aug = torch.cat([inputs_max, data])
        print('data_aug - ', data_aug.shape)
# before going in to the feature extractor -----------------
        labels = torch.cat([class_l, class_l])
        print('labels - ', labels.shape)
        
        # need to change our input size
        x_pathway_A = self.extract_feat(data_aug) # due to torch.cat x2  bigger than original
        # print('x_pathway_A - ', x_pathway_A.shape)
        # before pool - torch.Size([192, 2048, 7, 7])
        x_pathway_A = nn.AdaptiveAvgPool2d(1)(x_pathway_A)
        # torch.Size([192, 2048, 1, 1])
        x_pathway_A_feature = x_pathway_A.squeeze()
        # torch.Size([192, 2048])
        
        # this is the spot mu should be here
        logvar = self.p_logvar(x_pathway_A_feature.float())
        # torch.Size([192, 2048])
        #^^ logvar
        mu = self.p_mu(x_pathway_A_feature.float())
        # torch.Size([192, 2048])
        #^^ mu
        x_embedding = reparametrize(mu, logvar)
        # torch.Size([192, 2048])
        # ^^^ is the embedding
        # print(x_pathway_A.shape)
        # logit = self.cls_head(x_pathway_A.float(), num_segs)

        
        # print(x_embedding[:class_l.size(0)])
        # class_l.size(0) * num_segs
        emb_src = F.normalize(x_embedding[:class_l.size(0)]).unsqueeze(1)
        emb_aug = F.normalize(x_embedding[class_l.size(0):]).unsqueeze(1)
        print('emb_src -------------- ', emb_src.shape)
        print('emb_aug -------------- ', emb_aug.shape)
        print('labels -------------- ', labels.shape)
    
        q = torch.cat([emb_src, emb_aug], dim=1)
        print('number of feature  -------------- ', q.shape)
        loss_self_supervised = self.sup_contrastive_loss (torch.cat([emb_src, emb_aug], dim=1), class_l)

        mu = mu[labels.size(0):]
            # print(type(mu))
        logvar = logvar[labels.size(0):]
        y_samples = x_pathway_A[:labels.size(0)]
        likeli = -loglikeli(mu, logvar, y_samples)

        #cls_score_pathway_A = self.cls_head(x_pathway_A.float(), num_segs)
# cls_score_pathway_A = self.cls_head(x_pathway_A.float(), num_segs) -  torch.Size([12, 8])
        # gt_labels = labels.squeeze()
        # print(cls_score_pathway_A)
        # print('-------------')
        # print(self.cls_head)
        logit = self.cls_head(x_pathway_A.float(), num_segs)
        print('logit - ', logit.shape)
        #loss_cls = self.cls_head.loss(logit, labels, **kwargs) # class loss 
        criterion = nn.CrossEntropyLoss()
        loss_cls = criterion(logit, class_l_for_loss)
        loss_cls_stage_1 = loss_cls + loss_self_supervised + likeli
        step_1_loss = {'step_1_loss': loss_cls_stage_1}
        losses.update(step_1_loss) 
        print('passed stage 1 ----------------')
     #---------------------------------------STAGE1 END---------------------------------------

        inputs_max_stage_2 =tran(torch.sigmoid(self.convertor(data, estimation=True)))
        inputs_max_stage_2 = inputs_max_stage_2 * 0.6 + data * 0.4
        data_aug_stage2 = torch.cat([inputs_max_stage_2, data])

        x_pathway_A_stage_2 = self.extract_feat(data_aug_stage2)
        x_pathway_A_stage_2 = nn.AdaptiveAvgPool2d(1)(x_pathway_A_stage_2)
        x_pathway_A_stage_2 = x_pathway_A_stage_2.squeeze()
        logvar_stage_2 = self.p_logvar(x_pathway_A_stage_2)#logvar
        mu_stage_2 = self.p_mu(x_pathway_A_stage_2) #mu
        x_pathway_A_stage_2 = reparametrize(mu_stage_2, logvar_stage_2) #embedding

        logit_stage_2 = self.cls_head(x_pathway_A_stage_2.float(), num_segs) #useless in stage 2

        mu_stage_2 = mu_stage_2[class_l.size(0):]
            # print(type(mu))
        logvar_stage_2 = logvar_stage_2[class_l.size(0):]
        y_samples_stage_2 = x_pathway_A_stage_2[:class_l.size(0)]
        div = club(mu_stage_2, logvar_stage_2, y_samples_stage_2)

        # Semantic consistency
        e = x_pathway_A_stage_2
        e1 = e[:class_l.size(0)]
        e2 = e[class_l.size(0):]
        dist = conditional_mmd_rbf(e1, e2, class_l, num_class=num_segs)
        
        stage_2_loss = dist + 0.1 * div
        step_2_loss = {'stage_2_loss': stage_2_loss}
        losses.update(step_2_loss)
        # step_1_loss = {'step_1_loss': loss_cls_stage_1}
        #if we update the cls loss the model falls in to the wrong place
     
        #losses.update(loss_self_supervised)
        return losses


    def train_step(self, data_batch, optimizer, **kwargs):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data_batch (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``,
                ``num_samples``.
                ``loss`` is a tensor for back propagation, which can be a
                weighted sum of multiple losses.
                ``log_vars`` contains all the variables to be sent to the
                logger.
                ``num_samples`` indicates the batch size (when the model is
                DDP, it means the batch size on each GPU), which is used for
                averaging the logs.
        """
        print(data_batch.keys())
        print(data_batch['imgs'].shape)
        imgs_slow = data_batch['imgs']
        label = data_batch['label']

        aux_info = {}
        for item in self.aux_info:
            assert item in data_batch
            aux_info[item] = data_batch[item]

        losses = self((imgs_slow), label, return_loss=True)

        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss,
            log_vars=log_vars,
            num_samples=len(next(iter(data_batch[0].values()))))

        return outputs

    def _do_test(self, imgs):
        """Defines the computation performed at every call when evaluation,
        testing and gradcam."""
        batches = imgs.shape[0]
        imgs = imgs.reshape((-1, ) + imgs.shape[2:])
        num_segs = imgs.shape[0] // batches

        x = self.extract_feat(imgs)
        x = nn.AdaptiveAvgPool2d(1)(x)
        x = x.squeeze()

        if self.backbone_from in ['torchvision', 'timm']:
            if len(x.shape) == 4 and (x.shape[2] > 1 or x.shape[3] > 1):
                # apply adaptive avg pooling
                x = nn.AdaptiveAvgPool2d(1)(x)
            x = x.reshape((x.shape[0], -1))
            x = x.reshape(x.shape + (1, 1))

        if self.with_neck:
            x = [
                each.reshape((-1, num_segs) +
                             each.shape[1:]).transpose(1, 2).contiguous()
                for each in x
            ]
            x, _ = self.neck(x)
            x = x.squeeze(2)
            num_segs = 1

        if self.feature_extraction:
            # perform spatial pooling
            avg_pool = nn.AdaptiveAvgPool2d(1)
            x = avg_pool(x)
            # squeeze dimensions
            print(x.shape)
            print('---------')
            x = x.reshape((batches, num_segs, -1))
            # temporal average pooling
            x = x.mean(axis=1)
            return x

        # When using `TSNHead` or `TPNHead`, shape is [batch_size, num_classes]
        # When using `TSMHead`, shape is [batch_size * num_crops, num_classes]
        # `num_crops` is calculated by:
        #   1) `twice_sample` in `SampleFrames`
        #   2) `num_sample_positions` in `DenseSampleFrames`
        #   3) `ThreeCrop/TenCrop/MultiGroupCrop` in `test_pipeline`
        #   4) `num_clips` in `SampleFrames` or its subclass if `clip_len != 1`

        # should have cls_head if not extracting features
        cls_score = self.cls_head(x.float(), num_segs)

        assert cls_score.size()[0] % batches == 0
        # calculate num_crops automatically
        cls_score = self.average_clip(cls_score,
                                      cls_score.size()[0] // batches)
        return cls_score

