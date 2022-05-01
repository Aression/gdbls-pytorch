from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

from torchvision.datasets import CIFAR10, CIFAR100, SVHN

dataset_name = 'CIFAR10'

# Download training data from open datasets.
training_data = (eval(dataset_name))(
    root='datasets/' + dataset_name,
    train=True,
    download=True,
    transform=ToTensor(),
)

# Download test data from open datasets.
test_data = eval(dataset_name)(
    root='datasets/' + dataset_name,
    train=False,
    download=True,
    transform=ToTensor(),
)

batch_size = 64

# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break
