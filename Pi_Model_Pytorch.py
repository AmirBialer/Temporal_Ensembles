import torch
import torch.nn.functional as F 
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.autograd import Variable, Function

import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

class Pi_Model(nn.Module):
    def __init__(self, num_features, num_classes, dropout=0.15, noise=0.15):
        super(Pi_Model, self).__init__()
        self.loss = nn.CrossEntropyLoss
        self.last_layer_activation = F.softmax
        self.last_layer_neurons = num_classes
        self.gn = GaussianNoise(noise)
        self.activation = nn.LeakyReLU(0.1)
        self.Dense1 = nn.Linear(num_features,10)
        self.Dropout = nn.Dropout(dropout)
        self.Dense2 = nn.Linear(10,8)
        self.Dense3 = nn.Linear(8,self.last_layer_neurons)
        self.num_features = num_features
        self.num_classes = num_classes

    def forward(self, x):
        x = self.gn(x, self.training)
        x = self.activation(self.Dense1(x))
        x=self.Dropout(x)
        x=self.activation(self.Dense2(x))
        x=self.Dropout(x)
        x=self.Dense3(x)
        return x


class GaussianNoise(nn.Module):

    def __init__(self, std):
        super(GaussianNoise, self).__init__()
        self.std = std

    def forward(self, x, is_training):
        if is_training:
            zeros_ = torch.zeros(x.size()).cuda()
            #n = Variable(torch.normal(zeros_, std=self.std).cuda())
            n = Variable(torch.normal(zeros_, std=self.std)).cuda()
            return x + n
        return x