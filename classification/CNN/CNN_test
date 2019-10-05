import torch
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from tqdm import tqdm
from CNN_train import CNN
import numpy as np

dataset = MNIST(root='./datasets', download=True, train=False, transform=ToTensor()) # 데이터 정의, 0~1사이 값/파이토치가 다룰수있는 값으로
#    PIL Image : 파이썬의 이미지 자료형
#    -> torch.tensor : 파이토치의 자료형
data_loader = DataLoader(dataset, batch_size=32, shuffle=False) # 정의한 데이터 불러 오고

model = CNN() # 모델구조 가져오기
weight_dict = torch.load('./weight_dict.pt') # 모델 웨이트 가져오기

for k, _ in weight_dict.items():
    print(k)

model_weight = weight_dict['model_weight']

for k, _ in model_weight.items():
    print(k)

model.load_state_dict(model_weight)

list_acc =[]
for input, label in tqdm(data_loader):
    output = model(input)

    n_correct_answers = torch.sum(torch.eq(torch.argmax(output, dim=1), label)).item()
    list_acc.append(n_correct_answers / 32 * 100)

print("Acc: ", np.mean(list_acc))
