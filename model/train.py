import torch

from test import test
from utils import get_optim, get_criterion, get_device
from data.dataset import train_loader, test_loader
from model import model


def train(epochs, model, train_loader, device, optim, criterion):
    print("Training starting successfully...")

    for epoch in range(epochs):
        training_loss = 0

        model.train()

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optim.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optim.step()

            training_loss += loss.item()

        print(f"Epoch: {epoch+1}/{epochs}, Loss: {training_loss/len(train_loader):.4f}")
        torch.save(model.state_dict(), "car_color_model.pth")


if __name__ == "__main__":
    num_epochs = 20
    optim = get_optim(model)
    criterion = get_criterion()
    device = get_device()

    print(f"Total params: {sum(p.numel() for p in model.parameters())}")
    train(num_epochs, model, train_loader, device, optim, criterion)
    test(model, test_loader, device)