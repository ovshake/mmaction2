import torch
import torch.nn as nn
from mmcv.cnn import normal_init
import torch.nn.functional as F
from einops import rearrange

from ..builder import HEADS
from .base import AvgConsensus, BaseHead
from ..builder import build_loss
import math
import numpy as np
import itertools





@HEADS.register_module()
class projection_MLP(nn.Module):
    def __init__(self,
                 feature_size,
                 num_segments,
                 init_std=0.001,
                 middle_layer_dim=2048,
                 img_dim=2048,
                 **kwargs):

        super().__init__()


        self.fc1 = nn.Linear(feature_size * num_segments , middle_layer_dim, bias=True)
        self.BN1 = nn.BatchNorm1d(middle_layer_dim)
        self.relu_1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(middle_layer_dim, img_dim, bias=True)
        self.BN2 = nn.BatchNorm1d(img_dim)
        self.relu_2 = nn.ReLU(inplace=True)
        self.fc3 = nn.Linear(img_dim, img_dim, bias=True)
        self.BN3 = nn.BatchNorm1d(img_dim)

        self.encoder = nn.Sequential(self.fc1, self.BN1, self.relu_1, self.fc2, self.BN2, self.relu_2, self.fc3,self.BN3)

        self.init_std = init_std
        self.num_segments = num_segments
        self.img_dim = img_dim
        self.init_weights()



    def forward(self, features):
        #print(features.size())
        features = rearrange(features, '(b c) e -> b (c e)', c=self.num_segments)
        #print(features.size())
        features = self.encoder(features)
        return features

    def init_weights(self):
        """Initiate the parameters from scratch."""
        normal_init(self.encoder, std=self.init_std)




#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

@HEADS.register_module()
class prediction_MLP(nn.Module):
    def __init__(self,
                 feature_size,
                 init_std=0.001,
                 middle_layer_dim=512,
                 img_dim=2048,
                 **kwargs):

        super().__init__()


        self.fc1 = nn.Linear(feature_size, middle_layer_dim, bias=True)
        self.BN1 = nn.BatchNorm1d(middle_layer_dim)
        self.relu_1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(middle_layer_dim, img_dim, bias=True)



        self.encoder = nn.Sequential(self.fc1, self.BN1, self.relu_1, self.fc2)

        self.init_std = init_std
        self.img_dim = img_dim
        self.init_weights()



    def forward(self, features):
        #features = rearrange(features, '(b c) e -> b (c e)', c=self.num_segments)
        features = self.encoder(features)
        return features

    def init_weights(self):
        """Initiate the parameters from scratch."""
        normal_init(self.encoder, std=self.init_std)



#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@



@HEADS.register_module()
class projection_MLP_bias_false(nn.Module):
    def __init__(self,
                 feature_size,
                 num_segments,
                 init_std=0.001,
                 middle_layer_dim=2048,
                 img_dim=2048,
                 **kwargs):

        super().__init__()


        self.fc1 = nn.Linear(feature_size * num_segments , middle_layer_dim, bias=False)
        self.BN1 = nn.BatchNorm1d(middle_layer_dim)
        self.relu_1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(middle_layer_dim, img_dim, bias=False)
        self.BN2 = nn.BatchNorm1d(img_dim)
        self.relu_2 = nn.ReLU(inplace=True)
        self.fc3 = nn.Linear(img_dim, img_dim, bias=False)
        self.BN3 = nn.BatchNorm1d(img_dim, affine=False)

        self.encoder = nn.Sequential(self.fc1, self.BN1, self.relu_1, self.fc2, self.BN2, self.relu_2, self.fc3, self.BN3)

        self.init_std = init_std
        self.num_segments = num_segments
        self.img_dim = img_dim
        self.init_weights()



    def forward(self, features):
        #print(features.size())
        features = rearrange(features, '(b c) e -> b (c e)', c=self.num_segments)
        #print(features.size())
        features = self.encoder(features)
        return features

    def init_weights(self):
        """Initiate the parameters from scratch."""
        normal_init(self.encoder, std=self.init_std)




#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

@HEADS.register_module()
class prediction_MLP_bias_false(nn.Module):
    def __init__(self,
                 feature_size,
                 init_std=0.001,
                 middle_layer_dim=512,
                 img_dim=2048,
                 **kwargs):

        super().__init__()


        self.fc1 = nn.Linear(feature_size, middle_layer_dim, bias=False)
        self.BN1 = nn.BatchNorm1d(middle_layer_dim)
        self.relu_1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(middle_layer_dim, img_dim)



        self.encoder = nn.Sequential(self.fc1, self.BN1, self.relu_1, self.fc2)

        self.init_std = init_std
        self.img_dim = img_dim
        self.init_weights()



    def forward(self, features):
        #features = rearrange(features, '(b c) e -> b (c e)', c=self.num_segments)
        features = self.encoder(features)
        return features

    def init_weights(self):
        """Initiate the parameters from scratch."""
        normal_init(self.encoder, std=self.init_std)

#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
@HEADS.register_module()
class projection_MLP_multi(nn.Module):
    def __init__(self,
                 feature_size,
                 num_segments,
                 init_std=0.001,
                 middle_layer_dim=2048,
                 img_dim=2048,
                 **kwargs):

        super().__init__()


        self.fc1 = nn.Linear(feature_size * num_segments , middle_layer_dim, bias=True)
        self.BN1 = nn.BatchNorm1d(middle_layer_dim)
        self.relu_1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(middle_layer_dim, img_dim, bias=True)
        self.BN2 = nn.BatchNorm1d(img_dim)
        self.relu_2 = nn.ReLU(inplace=True)
        self.fc3 = nn.Linear(img_dim, img_dim, bias=True)
        self.BN3 = nn.BatchNorm1d(img_dim)
        self.relu_3 = nn.ReLU(inplace=True)

        self.encoder = nn.Sequential(self.fc1, self.BN1, self.relu_1, self.fc2, self.BN2, self.relu_2, self.fc3,self.BN3,self.relu_3)

        self.init_std = init_std
        self.num_segments = num_segments
        self.img_dim = img_dim
        self.init_weights()



    def forward(self, features):
        #print(features.size())
        features = rearrange(features, '(b c) e -> b (c e)', c=self.num_segments)
        #print(features.size())
        features = self.encoder(features)
        return features

    def init_weights(self):
        """Initiate the parameters from scratch."""
        normal_init(self.encoder, std=self.init_std)




#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

@HEADS.register_module()
class prediction_MLP_multi(nn.Module):
    def __init__(self,
                 feature_size,
                 init_std=0.001,
                 middle_layer_dim=512,
                 img_dim=2048,
                 **kwargs):

        super().__init__()


        self.fc1 = nn.Linear(feature_size, middle_layer_dim, bias=True)
        self.BN1 = nn.BatchNorm1d(middle_layer_dim)
        self.relu_1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(middle_layer_dim, img_dim, bias=True)
        self.relu_2 = nn.ReLU(inplace=True)



        self.encoder = nn.Sequential(self.fc1, self.BN1, self.relu_1, self.fc2,self.relu_2)

        self.init_std = init_std
        self.img_dim = img_dim
        self.init_weights()



    def forward(self, features):
        #features = rearrange(features, '(b c) e -> b (c e)', c=self.num_segments)
        features = self.encoder(features)
        return features

    def init_weights(self):
        """Initiate the parameters from scratch."""
        normal_init(self.encoder, std=self.init_std)



#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@