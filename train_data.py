import torch.nn as nn
import torch.optim as optim

import torchvision.models
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch
from torch.utils.tensorboard import SummaryWriter

import numpy as np

from test_data import image_file, abs_file, rel_file, new_file

from dataset import TestDataset

from OSU_model import OSUModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
input_size = 36866
num_classes = 2
learning_rate = 0.0001
fine_tune_lr = 0.00001
batch_size = 64
num_epochs = 10
load_model = True

abs_data = np.load(abs_file)
image_data = np.load(image_file)
rel_data = np.load(rel_file)

writer = SummaryWriter(f'runs/OSU/test4')


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("Saving Checkpoint")
    torch.save(state, filename)


def load_checkpoint(model, optimizer, checkpoint):
    print("Loading Checkpoint")
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])


def check_validation_loss(loader, model, criterion):
    # Set the model to evaluation mode
    model.eval()
    total_loss = 0.0

    # We don't need to compute gradients for validation
    with torch.no_grad():
        for data, target in loader:
            data = data.to(device)
            targets = target.to(device)
            scores = model(data)

            loss = criterion(scores, targets)
            total_loss += loss.item() * data.size(0)

    avg_loss = total_loss / len(loader.dataset)
    # Set the model back to training mode
    return avg_loss


step = 0
if __name__ == '__main__':

    # Model

    model = torchvision.models.resnet50(weights=True)

    print(model)

    for param in model.parameters():
        param.requires_grad = False

    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                            bias=False)

    model.fc = nn.Linear(in_features=2048, out_features=num_classes, bias=True)

    model.to(device)

    # Load Data

    dataset = TestDataset(image_file=image_file, new_file=new_file, transform=transforms.ToTensor())

    train_set, test_set = torch.utils.data.random_split(dataset, [30000, 7000])

    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    if load_model:
        load_checkpoint(model, optimizer, torch.load("my_checkpoint.pth.tar"))

    for param in model.parameters():
        param.requires_grad = True

    for epoch in range(num_epochs):

        checkpoint = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}

        if epoch % 2:
            save_checkpoint(checkpoint)

        for batch_idx, (data, target) in enumerate(train_loader):
            model.train()
            data = data.to(device)
            targets = target.to(device)

            scores = model(data)

            loss = criterion(scores, targets)
            img_grid = torchvision.utils.make_grid(data)
            writer.add_image("osu images", img_grid, global_step=step)
            writer.add_histogram('fc1', model.fc.weight)

            writer.add_scalar("Training Loss", loss, global_step=step)
            step += 1

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            print("Loss: ", loss.item())
            print("Epoch:", epoch)

        testing_loss = check_validation_loss(test_loader, model, criterion)
        print("Validation AVG Loss:", testing_loss)
        writer.add_scalar("Testing Loss", testing_loss, global_step=epoch)

    writer.close()