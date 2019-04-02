import torch                              # torch
import torch.nn as nn                     # torch.nn 이 뭐더라?
import torch.nn.functional as F           # torch.nn.functional 이 뭐더라? 함수를 만들어 주는 것인가?
import torchvision
import matplotlib.pyplot as plt

if __name__ == '__main__':
    MODEL = 'MLP'
    transform =  torchvision.transforms.Compose([torchvision.transforms.ToTensor(),                           # Compose를 써서 여러 transform??.??!!
                                             torchvision.transforms.Normalize(mean=[0.5],
                                                                             std=[0.5])])                     # ToTensor()는 torch의 텐서로 만들어준다.

    dataset = torchvision.datasets.MNIST(root='./dataset', train=True, transform=transform, download=True)    # 손글씨 데이터를 불러오는 작업  (총 6만개의 자료)
                                                                                                              # MNIST : 손글씨 분류 root : 다운로드할 디렉토리 설정(.은 현재 작업중인 디렉토리)
                                                                                                              # train = True(테스트 데이터와, 트레이닝 데이터 중 트레이닝 데이터 사용?)
                                                                                                              # download = True : 다운로드
    data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=32, num_workers = 0, shuffle=True)  # num_workers : 사용하는 쓰레드의 개수? 근데 왜 안되냐
                                                                                                              # batch_size : 한번에 처리할 데이터의 개수
                                                                                                              # shuffle = True : 자료를 섞을 것인지

    if MODEL == 'CNN':
        from models import CNN
        model = CNN()
    elif MODEL == 'MLP':
        from models import MLP
        model = MLP()
        print(10)
    else:
        raise NotImplementedError("You need to choose among [CNN, MLP].")


    loss = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=2e-4, betas=(0.5,0.99), eps=1e-8)       # parameters : wexight , optim : 그래디언트 전달 , lr: 계산되는 그래디언트의 값이 너무 크기 때문에 그래디언트에 붙이는 계수

    EPOCHS = 1                                                                 # 전체 데이터의 학습을 몇번 시킬 것인가?
    total_step = 0
    list_loss = list()
    for epoch in range(EPOCHS):
      for i, data in enumerate(data_loader):                                 # enumerate : 반복문에서 index를 입력할 때 사용
        total_step = total_step+1
        input, label = data[0], data[1]
        # input shape [32, 1, 28, 28] 첫번째 배치사이즈, channel, height, width
        input = input.view(input.shape[0], -1) if MODEL == 'MLP' else input   #
                                                    # batchsize, channel * height * width     왜 이렇게 하는거지 ? → 1차원으로 바꾸기 위해서
                                                    # view? reshape?  메모리주소?

        classification_results = model.forward(input) # [bstch size, 10] #nn.module을 상속한 클래스는 forward를 생략해도된다.

        l = loss(classification_results, label)
        list_loss.append(l.detach().item())         # detach 그래디언트를 없애는 item : torch tensor를 파이썬의 float이나 int로 바꿔줌

        optim.zero_grad()                           # 그래디언트를 초기화(
        l.backward()                                # l을 출력하기 까지의 해당하는 각각의 그래디언트를 돌려줌
        optim.step()                                # 이건 뭐냐 그래디언트를 각레이어로 전달해주는 거 맞나,,
        print(i)

    torch.save(model, '{}.pt'.format(MODEL))
    # 'MLP.pt', 'CNN.pt'
    plt.figure() #figure : 도화지
    plt.plot(range(len(list_loss)), list_loss, linestyle='--')
    plt.show()