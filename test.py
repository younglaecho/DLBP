import torch
import torch.nn as nn       # torch.nn
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt

class MLP(nn.Module):
    def __init__(self):
        super(MLP,self).__init__()         # super ... ... 상속시 클래스의 특성을 온전히 전잘하기 위함
        self.fc1 = nn.Linear(in_features=28 * 28, out_features=64)          #
        self.fc2 = nn.Linear(in_features=64, out_features=128)
        self.fc3 = nn.Linear(in_features=128, out_features=256)
        self.fc4 = nn.Linear(in_features=256, out_features=10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

transform =  torchvision.transforms.Compose([torchvision.transforms.ToTensor(),                                # Compose를 써서 여러 transform
                                             torchvision.transforms.Normalize(mean=[0.5],
                                                                             std=[0.5])])                     # ToTensor()는 torch의 텐서로 만들어준다.

dataset = torchvision.datasets.MNIST(root='./dataset', train=True, transform=transform, download=True)

data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=32, num_workers=1, shuffle=True)

mlp = MLP()
loss =nn.CrossEntropyLoss()
optim = torch.optim.SGD(mlp.parameters(), lr=2e-4)       # parameters : weight , Optim : 그래디언트 전달 , lr

EPOCHS = 1
total_step = 0
list_loss = list()
for epoch in range(EPOCHS):
    for i, data in enumerate(data_loader):
        total_step = total_step+1
        input, label = data[0], data[1]
        # input shape [32, 1, 28, 28] 첫번째 배치사이즈, channel, height, width
        input = input.view(input.shape[0], -1)  # batchsize, channel * height * width

        classfication_results = mlp.forward(input) # [bstch size, 10] #nn.module을 상속한 클래스는 forward를 생략해도된다.

        l = loss(classification_results, label)
        list_loss.append(l.detach().item())        # detach 그래디언트를 없애는 item : torch tensor를 파이썬의 plot이나 int로 바꿔줌

        optim.zero_grad()                          # 그래디언트를 초기화
        l.backward()                               # l을 출력하기 까지의 해당하는 각각의 그래디언트를 돌려줌
        optim.step()

plt.figure() #figure : 도화지
plt.plot(range(len(list_loss)), list_loss, linestyle='--')
plt.show()

