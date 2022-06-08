import ignite
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from ignite.engine import create_supervised_evaluator, create_supervised_trainer, Events
from ignite.metrics import Accuracy, Loss
from ignite.utils import setup_logger
from ignite.handlers import EarlyStopping
from sklearn.model_selection import StratifiedShuffleSplit

from model import gdbls
from torchvision.datasets import CIFAR10, CIFAR100, SVHN

from pyinstrument import Profiler
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd

cfg = {
    "overall_dropout": 0.5,
    "filters": [128, 192, 256],
    "divns": [2, 2, 2],
    "dropout_rate": [0.1, 0.1, 0.1],
    "batch_size": 128,
    "test_size": 0.05,
    "epochs": 100
}

# todo apply experiment on cifar100, svhn and record the performance
dataset_name = "CIFAR10"
num_classes = 10
input_shape = [3, 32, 32]
log_pth = "logs"
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

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


def get_data(dataset_name):
    cifar_norm_mean = (0.49139968, 0.48215827, 0.44653124)
    cifar_norm_std = (0.24703233, 0.24348505, 0.26158768)
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),  # this costs much time.
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(cifar_norm_mean, cifar_norm_std),
        transforms.RandomErasing(p=0.5)
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cifar_norm_mean, cifar_norm_std),
    ])

    trainset = eval(dataset_name)(root='datasets/' + dataset_name, train=True, download=True, transform=transform_train)
    validset = eval(dataset_name)(root='datasets/' + dataset_name, train=True, download=True, transform=transform_test)
    testset = eval(dataset_name)(root='datasets/' + dataset_name, train=False, download=True, transform=transform_test)

    labels = [trainset[i][1] for i in range(len(trainset))]
    ss = StratifiedShuffleSplit(n_splits=1, test_size=0.05, random_state=0)
    train_indices, valid_indices = list(ss.split(np.array(labels)[:, np.newaxis], labels))[0]
    trainset = torch.utils.data.Subset(trainset, train_indices)
    validset = torch.utils.data.Subset(validset, valid_indices)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=cfg["batch_size"],
                                              shuffle=False, drop_last=True, num_workers=4, pin_memory=True)
    validloader = torch.utils.data.DataLoader(validset, batch_size=cfg["batch_size"],
                                              shuffle=False, drop_last=True, num_workers=4, pin_memory=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=cfg["batch_size"],
                                             shuffle=False, drop_last=True, num_workers=4, pin_memory=True)
    print(f"load data complete")
    for X, y in testloader:
        print(f"Shape of X [N, C, H, W]: {X.shape}")
        print(f"Shape of y: {y.shape} {y.dtype}")
        break

    return trainloader, validloader, testloader


def score_function(engine):
    val_loss = engine.state.metrics['loss']
    # val_acc = engine.state.metrics['accuracy']
    return -val_loss


train_loader, val_loader, test_dataloader = get_data(dataset_name)
model = gdbls.GDBLS(
    num_classes=num_classes,
    input_shape=input_shape,
    overall_dropout=cfg["overall_dropout"],
    filters=cfg["filters"],
    divns=cfg["divns"],
    dropout_rate=cfg["dropout_rate"],
    batchsize=cfg["batch_size"]
).to(device)

# todo redirect logger output to recording files.
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, verbose=True, factor=0.2)

trainer = create_supervised_trainer(model, optimizer, loss_fn, device=device)
trainer.logger = setup_logger("trainer")

val_metrics = {"accuracy": Accuracy(), "loss": Loss(loss_fn)}
evaluator = create_supervised_evaluator(model, metrics=val_metrics, device=device)

early_stop_handler = EarlyStopping(patience=10, score_function=score_function, trainer=trainer)
evaluator.add_event_handler(Events.COMPLETED, early_stop_handler)  # set early stopping handler

evaluator.logger = setup_logger("evaluator")


@trainer.on(Events.EPOCH_STARTED)
def print_lr():
    lr = optimizer.param_groups[0]["lr"]
    trainer.logger.info(f"Current learning rate is {lr}")


# @trainer.on(Events.EPOCH_COMPLETED)
# def log_training_results(engine):
#     evaluator.run(train_loader)
#     metrics = evaluator.state.metrics
#     avg_accuracy = metrics["accuracy"]
#     avg_loss = metrics["loss"]
#     evaluator.logger.info(
#         f"Training Results - Epoch: {engine.state.epoch} Avg accuracy: {avg_accuracy:.4f} Avg loss: {avg_loss:.4f}"
#     )


@trainer.on(Events.EPOCH_COMPLETED)
def log_validation_results(engine):
    evaluator.run(val_loader)
    metrics = evaluator.state.metrics
    avg_accuracy = metrics["accuracy"]
    avg_loss = metrics["loss"]
    lr_scheduler.step(avg_loss)  # invoke lr_scheduler
    evaluator.logger.info(
        f"Validation Results - Epoch: {engine.state.epoch} Avg accuracy: {avg_accuracy:.4f} Avg loss: {avg_loss:.4f}"
    )


@trainer.on(Events.EPOCH_COMPLETED | Events.COMPLETED)
def log_time(engine):
    evaluator.logger.info(
        f"{trainer.last_event_name.name} took {trainer.state.times[trainer.last_event_name.name]} seconds")


@trainer.on(Events.COMPLETED)
def do_test(engine):
    evaluator.run(test_dataloader)
    metrics = evaluator.state.metrics
    avg_accuracy = metrics["accuracy"]
    avg_loss = metrics["loss"]
    evaluator.logger.info(
        f"Done, Test Results - Avg accuracy: {avg_accuracy:.4f} Avg loss: {avg_loss:.4f}"
    )

    y_pred = evaluator.state.output[0].cpu().numpy()
    y_pred = np.argmax(y_pred, axis=1)  # 选择max值进行输出0,或1
    y_true = evaluator.state.output[1].cpu().numpy()
    cf_matrix = confusion_matrix(y_pred, y_true, normalize='true')
    df_cm = pd.DataFrame(cf_matrix, index=[i for i in labelNames],
                         columns=[i for i in labelNames])
    plt.figure(figsize=(12, 7))
    sn.heatmap(df_cm, annot=True)
    plt.savefig('confusion_matrix.png')
    print(classification_report(y_true, y_pred, target_names=labelNames))


## pyinsnstrument profiler to analyse performance in time
# profiler = Profiler()
# profiler.start()

trainer.run(train_loader, max_epochs=cfg["epochs"])

# profiler.stop()
# print(profiler.output_text(unicode=True, color=True))
