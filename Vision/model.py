import torch
import torch.nn as nn
import torch.nn.functional as F


class FCL(nn.Module):
    def __init__(self,x_dim,h_dim,y_dim):
        super(FCL,self).__init__()
        self.x_dim = x_dim
        self.h_dim = h_dim
        self.y_dim = y_dim
        self.lin_1 = nn.Linear(self.x_dim,self.h_dim)
        self.lin_2 = nn.Linear(self.h_dim,self.y_dim)
        self.init_param()
    
    def forward(self,x):
        x = self.lin_2(F.relu(self.lin_1(x)))
        return x
    
    def init_param(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.kaiming_normal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        pass

    def forward(self,x):
        pass

    def init_param(self):
        pass

class VGG(nn.Module):
    def __init__(self):
        super(VGG,self).__init__()
        pass

    def forward(self,x):
        pass

    def init_param(self):
        pass