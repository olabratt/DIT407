from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

batch_size = 64

# Problem 1

train_data = MNIST("Assignment6/patrick", train=True, download=True, transform=ToTensor())
test_data = MNIST("Assignment6/patrick", train=False, download=True, transform=ToTensor())
train_dataloader = DataLoader(train_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

plt.imshow(train_data[0][0][0], cmap='Greys')
plt.show()
plt.imshow(test_data[0][0][0], cmap='Greys')
plt.show()

# Problem 2

