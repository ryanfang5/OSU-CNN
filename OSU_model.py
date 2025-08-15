import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import Dataset


class OSUModel(nn.Module):
    def __init__(self, in_channels=1, num_outputs=2):
        super(OSUModel, self).__init__()

        self.resnet = torchvision.models.resnet50(weights=True)

        for param in self.resnet.parameters():
            param.requires_grad = False

        self.resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3,
                                      bias=False)

        self.resnet.fc = nn.Linear(in_features=2048, out_features=num_outputs, bias=True)

    def forward(self, x):
        return self.resnet(x)
