from ignite_enterance import run

# todo use clear_ml to optimize these params

# configuration dictionary.
# cfg is hyper-parameters that are to be optimized.
cifar10_config = {
    'cfg': {
        "filters": [128, 192, 256],
        "divns": [2, 2, 2],
        "overall_dropout": 0.4,
        "dropout_rate": [0.1, 0.1, 0.1],
        "init_lr": 5e-4,
        "batch_size": 152,
        "test_size": 0.05,
        "epochs": 200
    },
    'dataset_name': "CIFAR10",
    'num_classes': 10,
    'input_shape': [3, 32, 32],
    'mean': (0.49139968, 0.48215827, 0.44653124),
    'std': (0.24703233, 0.24348505, 0.26158768),
    'log_pth': "logs/cifar10",
    'labelNames': [
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
}

cifar100_config = {
    'cfg': {
        "filters": [128, 192, 256],
        "divns": [4, 4, 4],
        "overall_dropout": 0.6,
        "dropout_rate": [0.2 for i in range(3)],
        "init_lr": 1e-3,
        "batch_size": 152,
        "test_size": 0,
        "epochs": 200
    },
    'dataset_name': "CIFAR100",
    'num_classes': 100,
    'input_shape': [3, 32, 32],
    'mean': (0.5074, 0.4867, 0.4411),
    'std': (0.2011, 0.1987, 0.2025),
    'log_pth': "logs/cifar100",
    'labelNames': [
        ('class-' + str(i)) for i in range(100)
    ]
}

SVHN_config = {
    'cfg': {
        "filters": [128, 192, 256],
        "divns": [2, 2, 2],
        "overall_dropout": 0.4,
        "dropout_rate": [0.1, 0.1, 0.1],
        "init_lr": 1e-4,
        "batch_size": 152,
        "test_size": 0.05,
        "epochs": 200
    },
    'dataset_name': "SVHN",
    'num_classes': 10,
    'input_shape': [3, 32, 32],
    'mean': (0.485, 0.456, 0.406),
    'std': (0.229, 0.224, 0.225),
    'log_pth': "logs/svhn",
    'labelNames': [
        'num-' + str(i) for i in range(10)
    ]
}

config = cifar100_config
options = {
    'analyse_time': False, 'log_details': True,
    'conclude_train': False
}

trainer = run(cifar100_config, options)
