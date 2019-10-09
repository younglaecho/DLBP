import torch
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from tqdm import tqdm
from CNN_train import CNN
import numpy as np

dataset = MNIST(root='./datasets', download=True, train=False, transform=ToTensor())
#    PIL Image : 파이썬의 이미지 자료형
#    -> torch.tensor : 파이토치의 자료형
data_loader = DataLoader(dataset, batch_size=32, shuffle=False) # 데이터를 batch 사이즈 32, 안섞고 불러옴

model = CNN() # 모델의 구조 가져옴
weight_dict = torch.load('./weight_dict.pt') # 훈련된 모델의 웨이트들을 가져옴

for k, _ in weight_dict.items():
    print(k)

model_weight = weight_dict['model_weight'] # 딥러닝 모델의 웨이트를 가져옴

for k, _ in model_weight.items(): # 웨이트를 가지는 층의 변수? 만 표시됨
    print(k)

model.load_state_dict(model_weight) # 가져온 웨이트를 적용

list_acc =[]
for input, label in tqdm(data_loader): # 불러옴 웨이트를 사용하여 테스트
    output = model(input)

    n_correct_answers = torch.sum(torch.eq(torch.argmax(output, dim=1), label)).item()
    list_acc.append(n_correct_answers / 32 * 100)

print("Acc: ", np.mean(list_acc))
