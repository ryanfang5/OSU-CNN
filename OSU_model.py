import torch.nn as nn
import torchvision.models
import numpy as np
from torchvision.models import ResNet50_Weights


class OSUModel(nn.Module):
    """
    Slightly modified version of Resnet50
    """
    def __init__(self, in_channels=1, num_outputs=2):
        super(OSUModel, self).__init__()

        self.resnet = torchvision.models.resnet50(weights=ResNet50_Weights.DEFAULT)

        # Start by only training last layer

        for param in self.resnet.parameters():
            param.requires_grad = False

        # Grayscale images

        # self.resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        # 2 outputs for mouse coordinates

        self.resnet.fc = nn.Linear(in_features=2048, out_features=num_outputs, bias=True)

    def forward(self, x):
        return self.resnet(x)


class EarlyStopping:
    """
    Stop early if validation loss has not found new minimum within the number of patience epochs
    """

    def __init__(self, model, optimizer, patience=5):
        self.patience = patience
        self.counter = 0
        self.best_loss = np.inf
        self.early_stop = False
        self.model = model
        self.optimizer = optimizer
        self.checkpoint = {}

    def __call__(self, val_loss):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
