import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from torch import nn, optim
import torch.nn.functional as F

# Нормируем данные
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])

# Загружаем тренировочный датасет
trainset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# Загружаем тестовый датасет
testset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)

#Расшифровка лейблов для подписей к картинкам
f_labels = ['T-shirt/top',
          'Trouser',
          'Pullover',
          'Dress',
          'Coat',
          'Sandal',
          'Shirt',
          'Sneaker',
          'Bag',
          'Ankle Boot']

image, label = next(iter(trainloader))
img_trans = np.reshape(image[0], (28,28) )
plt.imshow(img_trans,cmap='gray')
print(f_labels[label[0]])

class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 10)
        
    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.log_softmax(self.fc4(x), dim=1)

        return x

model = Classifier()
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.003)

epochs = 5

for e in range(epochs):
    running_loss = 0
    for images, labels in trainloader:
        log_ps = model(images)
        loss = criterion(log_ps, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    else:
        print(f"Epoch {e}. Training loss: {running_loss/len(trainloader)}")

#Вспомогательная функция для проверки сети
def view_classify(img, ps):
    ps = ps.data.numpy().squeeze()
    fig, (ax1, ax2) = plt.subplots(figsize=(6,9), ncols=2)
    ax1.imshow(img.resize_(1, 28, 28).numpy().squeeze(), cmap="gray")
    ax1.axis('off')
    ax2.barh(np.arange(10), ps)
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(10))
    ax2.set_yticklabels(f_labels, size='small');
    ax2.set_title('Class Probability')
    ax2.set_xlim(0, 1.1)
    plt.tight_layout()

dataiter = iter(testloader)
images, labels = dataiter.next()
img = images[0]
ps = torch.exp(model(img))
view_classify(img, ps)

import numpy as np

true_positive = np.zeros(10)
true_negative = np.zeros(10)
false_positive = np.zeros(10)
false_negative = np.zeros(10)
accuracy = 0
count = 0

for images, labels in iter(testloader):
    with torch.no_grad():
        labels_pred = model(images).max(dim=1)[1]
    for i in range(10):
        for pred, real in zip(labels_pred, labels):
            if real == i:
                if pred == real:
                    true_positive[i] += 1
                else:
                    false_negative[i] += 1
            else:
                if pred == i:
                    false_positive[i] += 1
                else:
                    true_negative[i] += 1
            
    accuracy += torch.sum(labels_pred == labels).item()
    count += len(labels)
print("Overall accuracy", accuracy / count)
print("Precision by class", true_positive / (true_positive + false_positive))
print("Recall by class", true_positive / (true_positive + false_negative))
print("Mean precision", np.mean(true_positive / (true_positive + false_positive)))
print("Mean pecall", np.mean(true_positive / (true_positive + false_negative)))

# CIFAR 10

import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from torch import nn, optim
import torch.nn.functional as F

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])
# Загружаем тренировочный датасет
trainset = datasets.CIFAR10('~/.pytorch/CIFAR10_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# Загружаем тестовый датасет
testset = datasets.CIFAR10('~/.pytorch/CIFAR10_data/', download=True, train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)
classes = ['plane', 
          'car', 
          'bird',
          'cat',
          'deer', 
          'dog',
          'frog',
          'horse',
          'ship',
          'truck']
def imshow(img, ax):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    ax.imshow(np.transpose(npimg, (1, 2, 0)))
fig, ax = plt.subplots(figsize=(9,5))
image, label = next(iter(trainloader))
imshow(image[0], ax)
plt.show()
print(classes[label[0]])

class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3*32*32, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 10)
        
    def forward(self, x):
        if len(x.shape) == 4:
            x = x.view(x.shape[0], -1)
        else:
            x = x.view(1,-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.log_softmax(self.fc4(x), dim=1)

        return x

model = Classifier()
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.003)

epochs = 5

for e in range(epochs):
    running_loss = 0
    for images, labels in trainloader:
        log_ps = model(images)
        loss = criterion(log_ps, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    else:
        print(f"Epoch {e}. Training loss: {running_loss/len(trainloader)}")

#Вспомогательная функция для проверки сети
def view_classify(img, ps):
    ps = ps.data.numpy().squeeze()
    fig, (ax1, ax2) = plt.subplots(figsize=(9,12), ncols=2)
    imshow(img.squeeze(), ax1)
    ax1.axis('off')
    ax2.barh(np.arange(10), ps)
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(10))
    ax2.set_yticklabels(classes, size='small');
    ax2.set_title('Class Probability')
    ax2.set_xlim(0, 1.1)
    plt.tight_layout()

dataiter = iter(testloader)
images, labels = dataiter.next()
img = images[0]
ps = torch.exp(model(img))
view_classify(img, ps)

import numpy as np

true_positive = np.zeros(10)
true_negative = np.zeros(10)
false_positive = np.zeros(10)
false_negative = np.zeros(10)
accuracy = 0
count = 0

for images, labels in iter(testloader):
    with torch.no_grad():
        labels_pred = model(images).max(dim=1)[1]
    for i in range(10):
        for pred, real in zip(labels_pred, labels):
            if real == i:
                if pred == real:
                    true_positive[i] += 1
                else:
                    false_negative[i] += 1
            else:
                if pred == i:
                    false_positive[i] += 1
                else:
                    true_negative[i] += 1
            
    accuracy += torch.sum(labels_pred == labels).item()
    count += len(labels)
print("Overall accuracy", accuracy / count)
print("Precision by class", true_positive / (true_positive + false_positive))
print("Recall by class", true_positive / (true_positive + false_negative))
print("Mean precision", np.mean(true_positive / (true_positive + false_positive)))
print("Mean pecall", np.mean(true_positive / (true_positive + false_negative)))
