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
class SlowFastContrastiveHead(nn.Module):
    def __init__(self,
                 feature_size,
                 num_segments,
                 middle_layer_dim=1024,
                 init_std=0.001,
                 img_dim=256,
                 **kwargs):

        super().__init__()
        self.fc1 = nn.Linear(feature_size * num_segments, middle_layer_dim, bias=True)
        self.relu_1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(middle_layer_dim, img_dim, bias=True)
        self.relu_2 = nn.ReLU(inplace=True)
        self.encoder = nn.Sequential(self.fc1, self.relu_1, self.fc2, self.relu_2)
        self.init_std = init_std
        self.num_segments = num_segments
        self.img_dim = img_dim
        self.init_weights()


    def forward(self, features):
        features = rearrange(features, '(b c) e -> b (c e)', c=self.num_segments)
        features = self.encoder(features)
        return features

    def init_weights(self):
        """Initiate the parameters from scratch."""
        normal_init(self.encoder, std=self.init_std)


@HEADS.register_module()
class ContrastiveHead(nn.Module):
    def __init__(self,
                 feature_size,
                 num_segments,
                 init_std=0.001,
                 middle_layer_dim=1024,
                 img_dim=512,
                 **kwargs):

        super().__init__()
        self.fc1 = nn.Linear(feature_size * num_segments, middle_layer_dim, bias=True)
        self.relu_1 = nn.ReLU(inplace=False)
        self.fc2 = nn.Linear(middle_layer_dim, img_dim, bias=True)
        self.relu_2 = nn.ReLU(inplace=False)
   
        self.encoder = nn.Sequential(self.fc1, self.relu_1, self.fc2, self.relu_2)
        

        
        
        self.init_std = init_std
        self.num_segments = num_segments
        self.img_dim = img_dim
        self.init_weights()



    def forward(self, features):
        # print('before rearrange',features.size())
        features = rearrange(features, '(b c) e -> b (c e)', c=self.num_segments)
        # print('after rearrange',features.size())
        features = self.encoder(features)
        return features

    def init_weights(self):
        """Initiate the parameters from scratch."""
        normal_init(self.encoder, std=self.init_std)


@HEADS.register_module()
class AugSelfHead(nn.Module):
    def __init__(self,
                 feature_size,
                 num_pathways=2,
                 num_segments=16,
                 init_std=0.001,
                 middle_layer_dim=1024,
                 img_dim=512,
                 **kwargs):

        super().__init__()
        self.num_pathways = num_pathways
        self.num_segments = num_segments
        self.fc1 = nn.Linear(feature_size * self.num_pathways * self.num_segments, middle_layer_dim, bias=True)
        self.relu_1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(middle_layer_dim, img_dim, bias=True)
        self.relu_2 = nn.ReLU(inplace=True)
        self.tanh_2 = nn.Tanh()
        self.encoder = nn.Sequential(self.fc1, self.relu_1, self.fc2, self.relu_2, self.tanh_2)
        self.init_std = init_std
        self.img_dim = img_dim
        self.init_weights()



    def forward(self, features):
        features = rearrange(features, '(b c) e -> b (c e)', c=self.num_segments)
        features = self.encoder(features)
        return features

    def init_weights(self):
        """Initiate the parameters from scratch."""
        normal_init(self.encoder, std=self.init_std)

@HEADS.register_module()
class TwoPathwayContrastiveHead(nn.Module):
    def __init__(self,
                 feature_size,
                 init_std=0.001,
                 **kwargs):

        super().__init__()
        self.fc1 = nn.Linear(feature_size, 512)
        self.relu_1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(512, 128)
        self.relu_2 = nn.ReLU(inplace=True)
        self.encoder = nn.Sequential(self.fc1, self.relu_1, self.fc2, self.relu_2)
        self.init_std = init_std
        self.init_weights()


    def forward(self, features):
        batch_size = features.shape[0]
        features = features.view(batch_size, -1)
        features = self.encoder(features.float())
        return features

    def init_weights(self):
        """Initiate the parameters from scratch."""
        normal_init(self.encoder, std=self.init_std)