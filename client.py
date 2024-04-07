import concurrent.futures
import os
from collections import OrderedDict

import pandas as pd
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms, datasets
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.datasets import CIFAR10

# pd.options.plotting.backend = "plotly"
from torch import nn
import flwr as fl

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
NUM_CLIENTS = 5
BATCH_SIZE = 32
data_dir = "./chest_xray"
TEST = 'test'
TRAIN = 'train'
VAL = 'val'


def data_transforms(phase=None):
    if phase == TRAIN:

        return transforms.Compose([

            transforms.Resize(size=(256, 256)),
            transforms.RandomRotation(degrees=(-20, +20)),
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    elif phase == TEST or phase == VAL:

        return transforms.Compose([

            transforms.Resize(size=(224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])


def load_data():
    trainset = datasets.ImageFolder(os.path.join(data_dir, TRAIN), transform=data_transforms(TRAIN))
    testset = datasets.ImageFolder(os.path.join(data_dir, TEST), transform=data_transforms(TEST))
    validset = datasets.ImageFolder(os.path.join(data_dir, VAL), transform=data_transforms(VAL))

    class_names = trainset.classes
    print(class_names)
    print(trainset.class_to_idx)
    trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
    validloader = DataLoader(validset, batch_size=64, shuffle=True)
    testloader = DataLoader(testset, batch_size=64, shuffle=True)
    images, labels = next(iter(trainloader))
    print(images.shape)
    print(labels.shape)
    return trainloader, trainloader, trainloader


def train(net, trainloader, epochs: int, verbose=False):
    """Train the network on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters())
    net.train()
    for epoch in range(epochs):
        correct, total, epoch_loss = 0, 0, 0.0
        for images, labels in trainloader:
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # Metrics
            epoch_loss += loss
            total += labels.size(0)
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
        epoch_loss /= len(trainloader.dataset)
        epoch_acc = correct / total
        if verbose:
            print(f"Epoch {epoch + 1}: train loss {epoch_loss}, accuracy {epoch_acc}")


def test(net, testloader):
    """Evaluate the network on the entire test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    net.eval()
    with torch.no_grad():
        for images, labels in testloader:
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    loss /= len(testloader.dataset)
    accuracy = correct / total
    return loss, accuracy


class Net(nn.Module):
    def __init__(self, num_classes=2):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=12)
        self.relu1 = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=20, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(in_channels=20, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(num_features=32)
        self.relu3 = nn.ReLU()
        self.fc = nn.Linear(in_features=32 * 112 * 112, out_features=num_classes)

        # Feed forwad function

    def forward(self, input):
        output = self.conv1(input)
        output = self.bn1(output)
        output = self.relu1(output)
        output = self.pool(output)
        output = self.conv2(output)
        output = self.relu2(output)
        output = self.conv3(output)
        output = self.bn3(output)
        output = self.relu3(output)
        output = output.view(-1, 32 * 112 * 112)
        output = self.fc(output)

        return output


class ChestXrayClient(fl.client.NumPyClient):
    def __init__(self, net, trainloader, validloader, testloader):
        self.trainloader = trainloader
        self.valloader = validloader
        self.testloader = testloader
        self.net = net

    def set_parameters(self, parameters):
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train(self.net, self.trainloader, epochs=1)
        return self.get_parameters(config={}), len(self.trainloader), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = test(self.net, self.testloader)
        return float(loss), len(self.valloader), {"accuracy": float(accuracy)}


def start_client(client_id):
    """
    Start a client with the given client ID.

    Parameters:
        client_id (int): The ID of the client.
    """
    trainloader, validloader, testloader = load_data()
    net = Net().to(DEVICE)
    client = ChestXrayClient(net, trainloader, validloader, testloader).to_client()
    print(f"Client {client_id} started.")
    fl.client.start_client(server_address="localhost:8080", client=client)


def start_clients(num_clients):
    """
    Start a specified number of clients concurrently.

    Parameters:
        num_clients (int): The number of clients to start.
    """
    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.map(start_client, range(1, num_clients + 1))


if __name__ == "__main__":
    # Load model and data
    start_clients(2)
