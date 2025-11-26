import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_dataloaders(dataset_name="MNIST", batch_size=64, data_dir=".data"):
    """
    Returns train and test DataLoaders for a given TorchVision dataset.

    Args:
        dataset_name (str): Name of dataset ('MNIST', 'CIFAR10', 'FashionMNIST', etc.)
        batch_size (int): Batch size for DataLoader.
        data_dir (str): Directory to store/download datasets.

    Returns:
        (train_loader, test_loader): Torch DataLoaders.
    """

    # Define common transforms for each dataset
    if dataset_name.upper() == "MNIST":
        transform = transforms.Compose(
            [
                transforms.Pad(2),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )
        dataset_class = datasets.MNIST

    elif dataset_name.upper() == "FASHIONMNIST":
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        )
        dataset_class = datasets.FashionMNIST

    elif dataset_name.upper() == "CIFAR10":
        transform = transforms.Compose(
            [
                transforms.Resize(224),  # or transforms.Resize(227)
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        dataset_class = datasets.CIFAR10

    elif dataset_name.upper() == "CIFAR100":
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        dataset_class = datasets.CIFAR100

    elif dataset_name.upper() == "IMAGENET":
        transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        dataset_class = datasets.ImageNet

    else:
        raise ValueError(f"Dataset '{dataset_name}' is not supported!")

    # Create datasets
    train_dataset = dataset_class(
        root=data_dir, train=True, download=True, transform=transform
    )
    test_dataset = dataset_class(
        root=data_dir, train=False, download=True, transform=transform
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )

    return train_loader, test_loader
