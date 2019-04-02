import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt


class MLP(nn.Module):                     # MLP의 기능은 layer를 만들고 activation function을 적용하는 것?
    def __init__(self):
        super(MLP,self).__init__()         # super ... ... 상속시 클래스의 특성을 온전히 전달하기 위함
        self.fc1 = nn.Linear(in_features=28 * 28, out_features=64)          #
        self.fc2 = nn.Linear(in_features=64, out_features=128)
        self.fc3 = nn.Linear(in_features=128, out_features=256)
        self.fc4 = nn.Linear(in_features=256, out_features=10)

    def forward(self, x):
        x = F.relu(self.fc1(x))            # Activation function  적용
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        # input shape : [Batch_size, 1, 28, 28]  channel, size(a*b)
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels=16, kernel_size=3, padding=1, stride=2)
        # output shape : [Batch_size,32, 14, 14]
        #(28-3+2*1)/2 +1 = 14
        self.conv2 = nn.Conv2d(in_channels = 16, out_channels=32, kernel_size=3, padding=1, stride=2)
        # output shape : [Batch_size,32, 7, 7]
        self.conv3 = nn.Conv2d(in_channels = 32, out_channels=64, kernel_size=3, padding=1, stride=2)
        # output shape : [Batch_size,32, 4, 4]
        self.conv4 = nn.Conv2d(in_channels = 64, out_channels=128, kernel_size=3, padding=1, stride=2)
        # output shape : [Batch_size,32, 2, 2]

        self.linear = nn.Linear(128 * 2 * 2, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(x.shape[0], -1) # [Batch_size, 128*2*2] #view?!??
        x = self.linear(x)
        return x
