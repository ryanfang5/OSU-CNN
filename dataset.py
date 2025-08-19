import torch
from torch.utils.data import Dataset

import numpy as np


class OSUDataset(Dataset):
    def __init__(self, image_file, output_file, transform=None):
        """
        Initialize image data, output data and transform
        :param image_file:
        :param output_file:
        :param transform:
        """
        self.images = np.load(image_file)
        self.output_data = np.load(output_file)
        self.transform = transform

    def __len__(self):
        """
        return length of samples
        :return: number of data samples
        """
        return len(self.images)

    def __getitem__(self, index):
        """
        Get an image and normalized mouse coordinates for a single index
        :param index:
        :return: (image, output) in tensor form
        """
        image = self.images[index]

        if self.transform:
            image = self.transform(image)

        (x, y) = self.output_data[index]

        output = torch.tensor((x, y), dtype=torch.float32)

        return image, output
