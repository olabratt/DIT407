import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

batch_size = 32
img_size = 28

device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)
print(f"Using {device} device")

# Problem 1

train_data = MNIST("Assignment6/patrick", train=True, download=True, transform=ToTensor())
test_data = MNIST("Assignment6/patrick", train=False, download=True, transform=ToTensor())
train_dataloader = DataLoader(train_data, batch_size=batch_size, num_workers=1, pin_memory=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, num_workers=1, pin_memory=True)

# print(train_data[0])
# plt.imshow(train_data[0][0][0], cmap='Greys') # 5
# plt.show()
# plt.imshow(test_data[0][0][0], cmap='Greys')  # 7
# plt.show()

# Problem 2

def accuracy(network, dataloader):
    network.eval()
    
    total_correct = 0
    total_instances = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            classifications = torch.argmax(network(images), dim=1)

            correct_predictions = sum(classifications==labels).item()

            total_correct+=correct_predictions
            total_instances+=len(images)
    return round(total_correct/total_instances, 3)

def train(model, dataloader, loss_fn, optimizer):
    for data, target in dataloader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        target_pred = model(data)

        loss = loss_fn(target_pred, target)
        loss.backward()
        optimizer.step()

NUM_EPOCHS = 10

model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(img_size * img_size, 300),
    nn.ReLU(),
    nn.Linear(300, 10),
    nn.Softmax(dim=1)
).to(device)

loss_fn = nn.CrossEntropyLoss().to(device)
optimizer = optim.SGD(model.parameters(), lr = 0.1)

# for i in range(NUM_EPOCHS):
#     train(model, train_dataloader, loss_fn, optimizer)
#     print(accuracy(model, test_dataloader))

# Problem 3

NUM_EPOCHS = 40

model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(img_size * img_size, 500),
    nn.ReLU(),
    nn.Linear(500, 300),
    nn.ReLU(),
    nn.Linear(300, 10),
    nn.Softmax(dim=1)
).to(device)

loss_fn = nn.CrossEntropyLoss().to(device)
optimizer = optim.SGD(model.parameters(), lr = 0.1, weight_decay=1e-4)

for i in range(NUM_EPOCHS):
    train(model, train_dataloader, loss_fn, optimizer)
    print(accuracy(model, test_dataloader))

# Problem 4

NUM_EPOCHS = 40

loss_fn = nn.CrossEntropyLoss().to(device)
optimizer = optim.SGD(model.parameters(), lr = 0.1, weight_decay=1e-4)

model = nn.Sequential(
    nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Flatten(),
    nn.Linear(9216, 128),
    nn.ReLU(),
    nn.Linear(128, 10),
    nn.Softmax(dim=1)
).to(device)

for i in range(NUM_EPOCHS):
    train(model, train_dataloader, loss_fn, optimizer)
    print(accuracy(model, test_dataloader))