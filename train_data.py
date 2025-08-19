import torch.nn as nn
import torch.optim as optim

import torchvision.models
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch
from torch.utils.tensorboard import SummaryWriter

from dataset import OSUDataset

from OSU_model import OSUModel, EarlyStopping

image_file = "image_data.npy"
output_file = "normalized_coords.npy"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
num_classes = 2
learning_rate = 0.0001
fine_tune_lr = 0.00001
batch_size = 64
num_epochs = 10
load_model = True

run = 1
logs = 0

writer = SummaryWriter(f'runs/OSU/{logs}')
model_name = f'osu_model_{run}.pth.tar'


def save_checkpoint(state, filename=model_name):
    print("Saving Checkpoint")
    torch.save(state, filename)


def load_checkpoint(model, optimizer, checkpoint):
    print("Loading Checkpoint")
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])


def train_model(model, train_loader, val_loader, optimizer, criterion,
                num_epochs=50, patience=5):
    early_stopper = EarlyStopping(model, optimizer, patience=patience)

    step = 0
    global_step = 0

    print("Training...")

    for epoch in range(num_epochs):

        checkpoint = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}

        if epoch % 2:
            save_checkpoint(checkpoint)

        model.train()
        train_loss = 0
        for data, target in train_loader:
            data = data.to(device)
            targets = target.to(device)

            optimizer.zero_grad()
            scores = model(data)

            loss = criterion(scores, targets)

            if step % 1000:
                img_grid = torchvision.utils.make_grid(data)
                writer.add_image("osu images", img_grid, global_step=step)

            step += 1

            loss.backward()

            optimizer.step()

            train_loss += loss.item()

        # --- Validation ---
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data, target in val_loader:
                data = data.to(device)
                targets = target.to(device)
                scores = model(data)
                loss = criterion(scores, targets)
                val_loss += loss.item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)

        writer.add_scalar("Training Loss", train_loss, global_step=global_step)
        writer.add_scalar("Validation Loss", val_loss, global_step=global_step)

        global_step += 1

        print(f"Epoch:", epoch,
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # Early stopping check
        early_stopper(val_loss)
        if early_stopper.early_stop:
            print("Early stopping triggered.")
            break

    writer.close()


if __name__ == '__main__':

    # Model

    model = OSUModel()

    model.to(device)

    # Load Data

    dataset = OSUDataset(image_file=image_file, output_file=output_file, transform=transforms.ToTensor())

    train_set, test_set = torch.utils.data.random_split(dataset, [43500, 4992])

    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.resnet.parameters(), lr=learning_rate)

    if load_model:
        load_checkpoint(model, optimizer, torch.load(model_name))

    # optimizer = optim.Adam(model.resnet.parameters(), lr=fine_tune_lr)

    # for param in model.resnet.parameters():
    #     param.requires_grad = True

    train_model(model, train_loader, test_loader, optimizer, criterion)
