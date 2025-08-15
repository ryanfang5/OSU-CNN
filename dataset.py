import torch
from torch.utils.data import Dataset

import numpy as np

SCREEN_WIDTH = 2560
SCREEN_HEIGHT = 1440

class TestDataset(Dataset):
    def __init__(self, image_file, new_file, transform=None):
        self.images = np.load(image_file)
        self.new_data = np.load(new_file)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]

        if self.transform:
            image = self.transform(image)

        (x, y) = self.new_data[index]

        x /= SCREEN_WIDTH
        y /= SCREEN_HEIGHT

        output = torch.tensor((x, y), dtype=torch.float32)

        return image, output
