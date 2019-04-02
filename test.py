from torch.utils.data import DataLoader
import torchvision
from models import MLP     # models 안에 있는 MLP 클래스를 가져옴 (함수, 클래스 불러올 수 있다.)
import torch

# Input pipeline
transform =torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                          torchvision.transforms.Normalize(mean=[0.5], std=[0.5])]) #   []? []? ??
dataset = torchvision.datasets.MNIST(root='./datasets', train=False, download=True, transform=transform)
data_loader = DataLoader(dataset = dataset, batch_size = 16, shuffle=False, num_workers = 0)

# Define model
mlp = MLP()
trained_model = torch.load('mlp.pt')# mlp에서 trained model을 불러오는거
state_dict = trained_model.state_dict() # weight를 가져오는거
mlp.load_state_dict(state_dict) # mlp에서 trained weight로 바꿔주는 거

nb_correct_answers = 0
for data in data_loader: # data_loader는 2*
    input, label = data[0], data[1]
    input = input.view(input.shape[0], -1)
    classification_results = mlp(input)
    print(classification_results.shape)
    nb_correct_answers += torch.eq(classification_results.argmax(), label).sum() #argmax : 가장 큰 값을 가지는 원소의 인덱스, 맞????
print("Average acc. : {} %".format(float(nb_correct_answers)/ len(data_loader)*100))