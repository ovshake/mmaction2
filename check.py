
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist


from torch.nn.functional import normalize





def single_stage_at_loss(f_s, f_t, p):
    f_s = F.normalize(f_s)
    f_t = F.normalize(f_t)

    return (f_s - f_t).pow(2).mean()


def at_loss(g_s, g_t, p):
    return sum([single_stage_at_loss(f_s, f_t, p) for f_s, f_t in zip(g_s, g_t)])


    def _forward(self, teacher_embeddings_list, student_embeddings_list):
        # losses
        loss_feat = self.feat_loss_weight * at_loss(
            teacher_embeddings_list, student_embeddings_list, self.p
        )
        # print('feature_student["feats"][1:]', len(feature_student["feats"][1:]))
        # print('feature_student["feats"][1]', feature_student["feats"][1].shape)
        return {"AT_loss": loss_feat}


loss_weight = 1.0
p =2
feat_loss_weight = 1000.0


#         # losses
teacher_embeddings = torch.rand((96,2048))
teacher_list = [teacher_embeddings,teacher_embeddings]
student_embeddings = torch.rand((96,2048))
student_list = [student_embeddings,student_embeddings]


loss = at_loss(teacher_list,student_list, p)
loss = feat_loss_weight * loss
print(loss)