import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets,transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from model import VGG
from utils import eval_func

print('Pytorch version :',torch.__version__)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Device :',device)

BATCH_SIZE = 64
lr = 1e-4
# transform
input_size = 224
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224,224)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])
])
# dataset
train_data = datasets.CIFAR100(root='data',train=True,transform=transforms,download=True)
test_data = datasets.CIFAR100(root='data',train=False,transform=transforms,download=True)
# from dataset to iter

train_iter = DataLoader(train_data,shuffle=True,batch_size=BATCH_SIZE,num_workers=1)
test_iter = DataLoader(test_data,shuffle=True,batch_size=BATCH_SIZE,num_workers=1)
# Check data in plt
# plt.imshow(train_data.__getitem__(1)[0].transpose(0,2).transpose(0,1))
# plt.title(f'{train_data.__getitem__(1)[1]}')
# plt.show()
# train function
def train(model,data_iter,device):
    EPOCH = 10
    model.train()
    for epoch in range(EPOCH):
        for batch_in, batch_label in data_iter:
            x = batch_in.to(device)
            y = batch_label.to(device)
            y_pred = model.forward(x)
            optimizer.zero_grad()
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
        train_acc = eval_func(model,train_iter,device)
        test_acc = eval_func(model,test_iter,device)
        print(f'EPOCH : {epoch}, train_acc : {train_acc}, test_acc : {test_acc}')

# model
model = VGG()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=lr)

train(model,train_iter,device)

