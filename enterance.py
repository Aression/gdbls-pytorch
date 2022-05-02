import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from sklearn.model_selection import train_test_split

from model.gdbls import GDBLS
from torchvision.datasets import CIFAR10, CIFAR100, SVHN

# configs
model_name = 'GDBLS'
dataset_name = "CIFAR10"
data_format = 'channels_first'
batch_size = 128
test_size = 0.05
epochs = 10
# config for gdbls cifar10
num_classes = 10
input_shape = [32, 32, 3]
overall_dropout = 0.5
filters = [128, 192, 256]
divns = [4, 4, 4]
dropout_rate = [0.1, 0.1, 0.1]
labelNames = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]
# save path
log_pth = 'recording/normal'
err_pth = 'recording/err'
saved_model_pth = 'saved_models/'

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


def get_data(dataset_name):
    # Download training data from open datasets.
    training_data = (eval(dataset_name))(
        root="datasets/" + dataset_name,
        train=True,
        download=True,
        transform=ToTensor(),
    )

    training_data, validation_data = train_test_split(training_data, test_size=test_size)

    # Download test data from open datasets.
    test_data = eval(dataset_name)(
        root="datasets/" + dataset_name,
        train=False,
        download=True,
        transform=ToTensor(),
    )

    # Create data loaders.
    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(validation_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    for X, y in test_dataloader:
        print(f"Shape of X [N, C, H, W]: {X.shape}")
        print(f"Shape of y: {y.shape} {y.dtype}")
        break

    return train_dataloader, val_dataloader, test_dataloader


def train(dataloader, model_name, loss_fn):
    """
    train the given model
    :param loss_fn:
    :param dataloader:
    :param model_name:
    :return:
    """
    model = eval(model_name)(
        num_classes=num_classes,
        input_shape=input_shape,
        overall_dropout=overall_dropout,
        filters=filters,
        divns=divns,
        dropout_rate=dropout_rate
    ).to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    return model


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return correct, test_loss


import sys
import time
from logger import Logger

if __name__ == '__main__':
    sys.stdout = Logger(log_pth, sys.stdout)
    sys.stderr = Logger(err_pth, sys.stderr)

    train_dataloader, val_dataloader, test_dataloader = get_data(dataset_name)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

    avgtime = 0
    besttime = 0
    bestacc = 0
    best_model = None
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")

        Stime = time.time()
        model = train(train_dataloader, model_name, loss_fn)
        correct, test_loss = test(test_dataloader, model, loss_fn)
        Etime = time.time()

        timecost = Etime - Stime
        if timecost < besttime:
            besttime = timecost

        if correct > bestacc:
            bestacc = correct
            best_model = model

    print(f"Done! The whole process costs {avgtime / epochs} seconds on average,\n"
          f"and the best best accuracy reaches {bestacc * 100}%!")

    Stime = time.time()
    corrects = 0
    whole = 0
    for x, y in test_dataloader:
        whole += 1
        with torch.no_grad():
            pred = best_model(x)
            predicted, actual = labelNames[pred[0].argmax(0)], labelNames[y]
            if predicted == actual:
                corrects+=1
            print(f'Predicted: "{predicted}", Actual: "{actual}"')
    Etime = time.time()

    print(f'Finished test in {Etime-Stime}seconds, the accuracy is {(corrects/whole)*100}%')

    torch.save(best_model.state_dict(), saved_model_pth + f"model-at-" +
               time.strftime("%m-%d %H:%M:%S") + ".pth")
    print("Saved PyTorch Model State to saved_models!")
