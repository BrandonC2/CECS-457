from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
from torchvision import datasets, transforms

from pathlib import Path
import matplotlib.pyplot as plt

# from torch_tensors_images.py assignment
def get_dataset(root: Path) -> datasets.CIFAR10:
    """
    Returns the CIFAR-10 training set.

    The first time you run this, it will download the dataset to `root`.
    """
    root.mkdir(parents=True, exist_ok=True)
    return datasets.CIFAR10(root=str(root), train=True, download=True, transform=torchvision.transforms.ToTensor())


# define the network
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # conv layer: 
        # in = 3 since CIFAR-10 images have 3 color channels (RGB)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        # fully connected layers
        self.fc1 = nn.Linear(64 * 2 * 2, 200)
        self.fc2 = nn.Linear(200, 10)

    def forward(self, x):  

        # used the photo's architecture as reference

        # conv1 to relu 
        x = F.relu(self.conv1(x))
        # maxpool(kernel=3, stride=3)
        x = F.max_pool2d(x, kernel_size = 3, stride = 3)

        # conv2 to relu 
        x = F.relu(self.conv2(x))
        # maxpool(kernel=2, stride=2)
        x = F.max_pool2d(x, kernel_size = 2, stride = 2)

        x = x.view(-1, 256) # flatten

        # fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def main() -> None:
    data_root = Path(__file__).resolve().parent / "data"

    # i copied the dataset into the root of the project
    # data_root = Path("./data") # Alt way since it kept saying __file__ is not defined in notebook
    dataset = get_dataset(data_root)

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

    images, labels = next(iter(data_loader))

    # create network
    net = Net()

    # forward pass
    output = net(images)

    print(f"Input tensor shape: {images.shape}")    # [64, 3, 32, 32]
    print(f"Output tensor shape: {output.shape}")   # [64, 10]

    optimizer = torch.optim.Adam(net.parameters(), lr = 0.001)
    criterion = nn.CrossEntropyLoss() # another name for loss function

    # training loop

    for epoch in range(5):

        for images, labels  in data_loader:
            optimizer.zero_grad()
            output = net(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch + 1}, loss: {loss.item():.4f}")


if __name__ == "__main__":
    main()