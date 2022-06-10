import torch
import argparse
import numpy as np
from torch import nn

# handel data
from torchvision.transforms import transforms
from sklearn.model_selection import StratifiedShuffleSplit

# ignite tools
from ignite.engine import create_supervised_evaluator, create_supervised_trainer, Events
from ignite.metrics import Accuracy, Loss
from ignite.utils import setup_logger
from ignite.contrib.handlers.clearml_logger import *
from ignite.handlers import EarlyStopping, Checkpoint

# models and data-sets
from model import gdbls
from torchvision.datasets import CIFAR10, CIFAR100, SVHN

# analyse tools
from pyinstrument import Profiler
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd


def get_data(config):
    mean = tuple(config['mean'])
    std = tuple(config['std'])

    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4, padding_mode="reflect"),  # this costs much time.
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        transforms.RandomErasing(p=0.5)
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    dataset_name = config['dataset_name']
    trainset = eval(dataset_name)(root='datasets/' + dataset_name, train=True, download=True, transform=transform_train)
    validset = eval(dataset_name)(root='datasets/' + dataset_name, train=True, download=True, transform=transform_test)
    testset = eval(dataset_name)(root='datasets/' + dataset_name, train=False, download=True, transform=transform_test)

    label_names = list(trainset.classes)
    if config['cfg']['test_size'] != 0:
        labels = [trainset[i][1] for i in range(len(trainset))]
        ss = StratifiedShuffleSplit(n_splits=1, test_size=config['cfg']['test_size'])
        train_indices, valid_indices = list(ss.split(np.array(labels)[:, np.newaxis], labels))[0]
        trainset = torch.utils.data.Subset(trainset, train_indices)
        validset = torch.utils.data.Subset(validset, valid_indices)

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=config['cfg']["batch_size"],
                                                  shuffle=False, drop_last=True, num_workers=4, pin_memory=True)
        validloader = torch.utils.data.DataLoader(validset, batch_size=config['cfg']["batch_size"],
                                                  shuffle=False, drop_last=True, num_workers=4, pin_memory=True)
        testloader = torch.utils.data.DataLoader(testset, batch_size=config['cfg']["batch_size"],
                                                 shuffle=False, drop_last=True, num_workers=4, pin_memory=True)
    else:
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=config['cfg']["batch_size"],
                                                  shuffle=False, drop_last=True, num_workers=4, pin_memory=True)
        validloader = testloader = torch.utils.data.DataLoader(testset, batch_size=config['cfg']["batch_size"],
                                                               shuffle=False, drop_last=True, num_workers=4,
                                                               pin_memory=True)

    print(f"load data complete")
    for X, y in testloader:
        print(f"Shape of X [N, C, H, W]: {X.shape}")
        print(f"Shape of y [N, label]: {y.shape} {y.dtype}")
        print(y.min(), y.max())
        break

    return trainloader, validloader, testloader, label_names


def score_function(engine):
    val_loss = engine.state.metrics['loss']
    # val_acc = engine.state.metrics['accuracy']
    return -val_loss


def run(config, options, logger):
    torch.manual_seed(0)
    # Get cpu or gpu device for training.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    # we should load data first exactly.
    assert get_data is not None
    train_loader, val_loader, test_dataloader, label_names = get_data(config)

    model = gdbls.GDBLS(
        num_classes=config['num_classes'],
        input_shape=config['input_shape'],
        overall_dropout=config['cfg']["overall_dropout"],
        filters=config['cfg']["filters"],
        divns=config['cfg']["divns"],
        dropout_rate=config['cfg']["dropout_rate"],
        batchsize=config['cfg']["batch_size"]
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config['cfg']['init_lr'], weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True, factor=0.2)

    trainer = create_supervised_trainer(model, optimizer, loss_fn, device=device)
    trainer.logger = setup_logger("trainer")

    metrics = {"accuracy": Accuracy(), "loss": Loss(loss_fn)}
    # for training eval
    train_evaluator = create_supervised_evaluator(model, metrics=metrics, device=device)
    train_evaluator.logger = setup_logger("Train Evaluator")
    # for validation eval
    validation_evaluator = create_supervised_evaluator(model, metrics=metrics, device=device)
    validation_evaluator.logger = setup_logger("Val Evaluator")

    early_stop_handler = EarlyStopping(patience=6, score_function=score_function, trainer=trainer)
    validation_evaluator.add_event_handler(Events.COMPLETED, early_stop_handler)  # set early stopping handler

    clearml_logger = None
    if options['log_details']:
        # To utilize other loggers we need to change the object here
        clearml_logger = ClearMLLogger(project_name="examples", task_name="ignite")

        # Attach the logger to the trainer to log training loss
        clearml_logger.attach_output_handler(
            trainer,
            event_name=Events.ITERATION_COMPLETED(every=100),
            tag="training",
            output_transform=lambda loss: {"batchloss": loss},
        )
        # Attach the logger to log loss and accuracy for both training and validation
        for tag, evaluator in [("training metrics", train_evaluator), ("validation metrics", validation_evaluator)]:
            clearml_logger.attach_output_handler(
                evaluator,
                event_name=Events.EPOCH_COMPLETED,
                tag=tag,
                metric_names=["loss", "accuracy"],
                global_step_transform=global_step_from_engine(trainer),
            )
        # Attach the logger to the trainer to log optimizer's parameters, e.g. learning rate
        clearml_logger.attach_opt_params_handler(
            trainer, event_name=Events.EPOCH_COMPLETED(every=1), optimizer=optimizer
        )
        # # Attach the logger to the trainer to log model's weights as a scalar
        # clearml_logger.attach(trainer, log_handler=WeightsScalarHandler(model),
        #                       event_name=Events.EPOCH_COMPLETED(every=1))
        # # Attach the logger to the trainer to log model's gradients as a histogram
        # clearml_logger.attach(trainer, log_handler=GradsScalarHandler(model),
        #                       event_name=Events.EPOCH_COMPLETED(every=1))
        # save the best checkpoint
        handler = Checkpoint(
            {"model": model},
            ClearMLSaver(dirname=config['log_pth'] + '/saves', require_empty=False),
            n_saved=1,
            score_function=lambda e: e.state.metrics["accuracy"],
            score_name="val_acc",
            filename_prefix="best",
            global_step_transform=global_step_from_engine(trainer),
        )
        validation_evaluator.add_event_handler(Events.EPOCH_COMPLETED, handler)  # set Checkpoint handler

    @trainer.on(Events.EPOCH_STARTED)
    def print_lr():
        lr = optimizer.param_groups[0]["lr"]
        trainer.logger.info(f"Epoch[{trainer.state.epoch}]: Current learning rate is {lr}")

    @trainer.on(Events.EPOCH_COMPLETED)
    def compute_metrics(engine):
        train_evaluator.run(train_loader)
        validation_evaluator.run(val_loader)
        lr_scheduler.step(validation_evaluator.state.metrics["loss"])

    @trainer.on(Events.EPOCH_COMPLETED | Events.COMPLETED)
    def log_time(engine):
        print(f"{trainer.last_event_name.name} took {trainer.state.times[trainer.last_event_name.name]} seconds")

    @trainer.on(Events.COMPLETED)
    def do_test(engine):
        if options['log_details']:
            clearml_logger.close()

        # run test dataset
        validation_evaluator.run(test_dataloader)
        metrics = validation_evaluator.state.metrics
        avg_accuracy = metrics["accuracy"]
        avg_loss = metrics["loss"]
        print(
            f"Done, Test Results - Avg accuracy: {avg_accuracy:.4f} Avg loss: {avg_loss:.4f}"
        )

        y_pred = validation_evaluator.state.output[0].cpu().numpy()
        y_pred = np.argmax(y_pred, axis=1)  # 选择max值进行输出: 0或1
        y_true = validation_evaluator.state.output[1].cpu().numpy()

        if options['conclude_train']:
            # draw confusion matrix
            cf_matrix = confusion_matrix(y_pred, y_true, normalize='true')
            df_cm = pd.DataFrame(cf_matrix, index=label_names, columns=label_names)
            plt.figure(figsize=(20, 20))
            sn.heatmap(df_cm, annot=True)
            plt.savefig(config['log_pth'] + '/confusion_matrix.png')

            # print classification report
            print(classification_report(y_true, y_pred, target_names=label_names))

    if options['analyse_time']:  # use pyinsnstrument profiler to analyse performance in time
        profiler = Profiler()
        profiler.start()
        trainer.run(train_loader, max_epochs=config['cfg']["epochs"])
        profiler.stop()
        print(profiler.output_text(unicode=True, color=True))
    else:
        trainer.run(train_loader, max_epochs=config['cfg']["epochs"])
