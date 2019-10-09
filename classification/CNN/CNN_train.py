import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

class CNN(nn.Module):
    def __init__(self):  # init : weight를 정의
        super(CNN, self).__init__() # 부모 클래스의 init이 실행되도록
        self.input_layer = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1) # 16x28x28
        self.layer_1 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1, stride=2) # 32x14x14
        self.layer_2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, stride=2) # 64x7x7
        self.layer_3 = nn.AdaptiveAvgPool2d((1,1)) # 64x1x1
        self.layer_4 = nn.Linear(in_features=64, out_features=10) # 10 ; 클래스의 개수
        # 공간사이즈를 줄이고, channel, weight 를 늘림

    def forward(self, x):  # forward : weight를 계산
        x1 = F.relu(self.input_layer(x)) # 16x28x28
        x2 = F.relu(self.layer_1(x1)) # 32x14x14
        x3 = F.relu(self.layer_2(x2)) # 64x7x7
        x4 = self.layer_3(x3) # 64x1x1
        x5 = x4.view(x4.shape[0], 64) # x4.shape :[B,64,1,1] >> Bx64 ; view : 차원을 맞춰줌..
        output = self.layer_4(x5) # Bx10
        return output

if __name__ == '__main__':
    dataset = MNIST(root='./datasets', train=True, transform=ToTensor(), download=True) # ToTensor : 0~1 값으로 만들어줌
    data_loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0) # DataLoader(data, 배치사이즈, 셔플, 스레드)

    model = CNN() # CNN이라는 클래스의 인스턴스를 생성, CNN : 딥러닝 모델의 구조(층)를 정의

    criterion = nn.CrossEntropyLoss() # Loss 정의

    optim = torch.optim.Adam(model.parameters(), lr=0.001)  # parameters : 웨이트들의 값을 돌려줌
        # optim : 웨이트들의 값을 수정해 줌.
        # weight_new = weight_old - weight_gradient * lr
   # if os.path.isfile("./weight_dict.pt"):
   #    model_dict = torch.load('./weight_dict.pt')['model_weight']
   #    model.load_state_dict(model_dict)
   #    adam_dict = torch.load('./weight_dict.pt')['adam_weight']
   #    optim.load_state_dict(adam_dict)

    # 위 주석처리된 부분은 이미 만들어진 모델의 웨이트가 있다면 불러오는 구문..

    list_loss = []
    list_acc = []

    for epoch in range(1): # 1회 훈련
        for input, label in tqdm(data_loader):
            # label 32
            output = model(input) # 32*10
            loss = criterion(output, label) # 1

            optim.zero_grad() # 이전 단계에서 사용한 웨이트 초기화
            loss.backward()
            optim.step()
            # 웨이트를 적용시킴.
            list_loss.append(loss.detach().item()) # .detach : 파이토치에서 웨이트들과의 관계를 끊음 .item : 파이토치 실수->파이썬

            n_correct_answers = torch.sum(torch.eq(torch.argmax(output, dim=1), label)).item() # 정확도 평가
            print("Accuracy:", n_correct_answers / 32 *100) # 정확도 평가
            list_acc.append(n_correct_answers / 32 *100) # 정확도 평가

    weight_dict = {'model_weight': model.state_dict(), 'adam_weight': optim.state_dict()}  # 파이썬의 딕셔너리 형태로 웨이트를 저장
    torch.save(weight_dict, "./weight_dict.pt") # 웨이트 저장
    #torch.save(model.state_dict(), "./CNN_model.pt") # 파이썬의 딕셔너리 형태로 웨이트를 저장
    #torch.save(optim, "./adam.pt")

    plt.plot(list_loss)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.show()

    plt.plot(list_acc)
    plt.xlabel("Iteration")
    plt.ylabel("Acc")
    plt.show()
