from torchvision.datasets import CIFAR100, STL10, ImageFolder
from torch.utils.data import DataLoader
from preprocessing import preprocess_images_cifar_tinyimagenet, preprocess_images_stl_caltech


# Define a function to create a data loader for the CIFAR-100 dataset
def create_dataloader_cifar(dataset_path, batch_size):
    # Apply preprocessing
    transforms = preprocess_images_cifar_tinyimagenet()

    # Load dataset
    dataset = CIFAR100(root=dataset_path, download=True, transform=transforms)

    # Create data loader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader

# Define a function to create a data loader for the STL-10 dataset
def create_dataloader_stl(dataset_path, batch_size):
    # Apply preprocessing
    transforms = preprocess_images_stl_caltech()

    # Load dataset
    dataset = STL10(root=dataset_path, download=True, transform=transforms)

    # Create data loader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader


# Add data loaders for Tiny ImageNet, Caltech-101, and Caltech-256.
def create_dataloader(dataset_path, batch_size, preprocess_func):
    # Apply preprocessing
    transforms = preprocess_func()

    # Load dataset
    dataset = ImageFolder(root=dataset_path, transform=transforms)

    # Create data loader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader