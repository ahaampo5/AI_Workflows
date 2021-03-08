import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets,transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
print('Pytorch version :',torch.__version__)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Device :',device)

train_data = datasets.CIFAR100(root='data',train=True,transform=transforms.ToTensor(),download=True)
test_data = datasets.CIFAR100(root='data',train=False,transform=transforms.ToTensor(),download=True)

BATCH_SIZE = 128
train_iter = DataLoader(train_data,shuffle=True,batch_size=BATCH_SIZE,num_workers=1)
test_iter = DataLoader(test_data,shuffle=True,batch_size=BATCH_SIZE,num_workers=1)

plt.imshow(train_data.__getitem__(1)[0].transpose(0,2).transpose(0,1))
plt.title(f'{train_data.__getitem__(1)[1]}')
plt.show()