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
        super(VGG, self).__init__()

        self.relu = nn.ReLU(inplace=True)

        # Convolution Feature Extraction Part
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3_1   = nn.BatchNorm2d(256)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn3_2   = nn.BatchNorm2d(256)
        self.pool3   = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn4_1   = nn.BatchNorm2d(512)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn4_2   = nn.BatchNorm2d(512)
        self.pool4   = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn5_1   = nn.BatchNorm2d(512)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn5_2   = nn.BatchNorm2d(512)
        self.pool5   = nn.MaxPool2d(kernel_size=2, stride=2)
        '''======================== TO DO (1) ========================'''
        '''==========================================================='''


        # Fully Connected Classifier Part
        self.gap      = nn.AdaptiveAvgPool2d(1)
        self.fc1      = nn.Linear(512 * 1 * 1, 4096)
        self.dropout1 = nn.Dropout()

        '''==========================================================='''
        '''======================== TO DO (2) ========================'''
        self.fc2      = nn.Linear(4096,4096)
        self.dropout2 = nn.Dropout()

        self.fc3      = nn.Linear(4096,100)
        '''======================== TO DO (2) ========================'''
        '''==========================================================='''

    def forward(self, x):
        # Convolution Feature Extraction Part
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool2(x)

        x = self.conv3_1(x)
        x = self.bn3_1(x)
        x = self.relu(x)
        x = self.conv3_2(x)
        x = self.bn3_2(x)
        x = self.relu(x)
        x = self.pool3(x)

        x = self.conv4_1(x)
        x = self.bn4_1(x)
        x = self.relu(x)
        x = self.conv4_2(x)
        x = self.bn4_2(x)
        x = self.relu(x)
        x = self.pool4(x)

        x = self.conv5_1(x)
        x = self.bn5_1(x)
        x = self.relu(x)
        x = self.conv5_2(x)
        x = self.bn5_2(x)
        x = self.relu(x)

        # Fully Connected Classifier Part
        x = self.gap(x)
        x = torch.flatten(x,1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout2(x)

        x = self.fc3(x)
        return x