import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from ..builder import LOSSES
from .base import BaseWeightedLoss
from torch.nn.functional import normalize





def kd_loss(logits_student, logits_teacher, temperature):
    log_pred_student = F.log_softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1).mean()
    loss_kd *= temperature**2
    # print(loss_kd)
    # print('loss_kd ^^^^')
    # print('loss_kd type', type(loss_kd))
    return loss_kd

@LOSSES.register_module()
class KD(BaseWeightedLoss):
    """Distilling the Knowledge in a Neural Network"""

    def __init__(self, loss_weight=1.0):
        super().__init__()
        self.loss_weight = loss_weight
        self.temperature = 4
       
        self.kd_loss_weight = 0.9

    def _forward(self, logits_teacher, logits_student):
     
        # losses
       
        loss_kd = self.kd_loss_weight * kd_loss(
            logits_student, logits_teacher, self.temperature
        )

        return {"KD_loss": loss_kd}


#------------------------------------------------------------

def _pdist(e, squared, eps):
    e_square = e.pow(2).sum(dim=1)
    prod = e @ e.t()
    res = (e_square.unsqueeze(1) + e_square.unsqueeze(0) - 2 * prod).clamp(min=eps)

    if not squared:
        res = res.sqrt()

    res = res.clone()
    res[range(len(e)), range(len(e))] = 0
    return res


def rkd_loss(f_s, f_t, squared=False, eps=1e-12, distance_weight=25, angle_weight=50):
    stu = f_s.view(f_s.shape[0], -1)
    tea = f_t.view(f_t.shape[0], -1)
    # print("stu = f_s.view(f_s.shape[0], -1)",stu.shape )
    # print("tea = f_t.view(f_t.shape[0], -1)", tea.shape)

    # RKD distance loss
    with torch.no_grad():
        t_d = _pdist(tea, squared, eps)
        mean_td = t_d[t_d > 0].mean()
        t_d = t_d / mean_td

    d = _pdist(stu, squared, eps)
    mean_d = d[d > 0].mean()
    d = d / mean_d
 

    loss_d = F.smooth_l1_loss(d, t_d).float()

    # RKD Angle loss
    with torch.no_grad():
        td = tea.unsqueeze(0) - tea.unsqueeze(1) *1e-3
        # print('td has NaN:', torch.isnan(td).any())
        # print('td has Inf:', torch.isinf(td).any())
        # td[torch.isnan(td)] = 0 # Replace NaN values with 0
        #norm_td = F.normalize(td, p=2, dim=2, eps=1e-8) # Add eps parameter
     
        norm_td = F.normalize(td, p=2, dim=2, eps=1e-6)
        t_angle = torch.bmm(norm_td, norm_td.transpose(1, 2)).view(-1)
        # if torch.any(torch.isnan(t_angle)):
        #     print('t_angle nan')



    sd = stu.unsqueeze(0) - stu.unsqueeze(1)
   

    sd = stu.unsqueeze(0) - stu.unsqueeze(1)* 1e-3
    # print('sd has NaN:', torch.isnan(sd).any())
    # print('sd has Inf:', torch.isinf(sd).any())
    # sd[torch.isnan(sd)] = 0 # Replace NaN values with 0
    #norm_sd = F.normalize(sd, p=2, dim=2, eps=1e-8) # Add eps parameter
    norm_sd = F.normalize(sd, p=2, dim=2, eps=1e-6)
    # norm_sd = F.normalize(sd, p=2, dim=2) 
    s_angle = torch.bmm(norm_sd, norm_sd.transpose(1, 2)).view(-1)
    # if torch.any(torch.isnan(s_angle)):
    #     print('s_angle nan')
    # print('s_angle', s_angle)
    # print('norm_sd', norm_sd)
    loss_a = F.smooth_l1_loss(s_angle, t_angle)
    if torch.isnan(loss_a):
        assert print("loss_a = F.smooth_l1_loss(s_angle, t_angle) return nan")

    # print(" loss_a = F.smooth_l1_loss(s_angle, t_angle) ",loss_a )
    loss = distance_weight * loss_d + angle_weight * loss_a
    return loss

@LOSSES.register_module()
class RKD(BaseWeightedLoss):
    """Relational Knowledge Disitllation, CVPR2019"""

    def __init__(self, loss_weight=1.0):
        super().__init__()
        self.distance_weight = 25
        self.angle_weight = 50
        self.feat_loss_weight = 1.0
        self.eps = 1.0e-12
        self.squared = False
        self.loss_weight=loss_weight

    def _forward(self, teacher_embeddings, student_embeddings):
   
        # for numeric stabability normalize the feature 
        # student_embeddings = F.normalize(student_embeddings, p=2, dim=1)
        # teacher_embeddings = F.normalize(teacher_embeddings, p=2, dim=1)

        # losses
        # print('teacher_embeddings ', f_t_norm.shape)
        # print('student_embeddings ', f_s_norm.shape)
        loss_rkd = self.feat_loss_weight * rkd_loss(
            teacher_embeddings,
            student_embeddings,
            self.squared,
            self.eps,
            self.distance_weight,
            self.angle_weight,
        )
    
        return {"RKD_loss": loss_rkd}

# f_s shape torch.Size([16, 256])
# stu = f_s.view(f_s.shape[0], -1) torch.Size([16, 256])
# td shape -  torch.Size([16, 16, 256])
# norm_td torch.Size([16, 16, 256])
# t_angle torch.Size([4096])

#------------------------------------------------------------




def single_stage_at_loss(f_s, f_t, p):
    f_s = F.normalize(f_s)
    f_t = F.normalize(f_t)

    return (f_s - f_t).pow(2).mean()


def at_loss(g_s, g_t, p):
    return sum([single_stage_at_loss(f_s, f_t, p) for f_s, f_t in zip(g_s, g_t)])

@LOSSES.register_module()
class AT(BaseWeightedLoss):
    """
    Paying More Attention to Attention: Improving the Performance of Convolutional Neural Networks via Attention Transfer
    src code: https://github.com/szagoruyko/attention-transfer
    """

    def __init__(self, loss_weight=1.0):
        super().__init__()
        self.loss_weight = loss_weight
        self.p =2
        self.feat_loss_weight = 1000.0

    def _forward(self, teacher_embeddings_list, student_embeddings_list):
        # losses
        loss_feat = self.feat_loss_weight * at_loss(
            teacher_embeddings_list, student_embeddings_list, self.p
        )
        # print('feature_student["feats"][1:]', len(feature_student["feats"][1:]))
        # print('feature_student["feats"][1]', feature_student["feats"][1].shape)
        return {"AT_loss": loss_feat}

