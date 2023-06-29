# First, let's import the necessary libraries
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from cqr import helper
from nonconformist.nc import RegressorNc
from nonconformist.icp import IcpRegressor
from nonconformist.nc import AbsErrorErrFunc

# Let's define a function for loading a dataset
def load_dataset(dataset_name, train=True):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    if dataset_name == 'CIFAR10':
        dataset = datasets.CIFAR10(root='./data', train=train, download=True, transform=transform)
    elif dataset_name == 'CIFAR100':
        dataset = datasets.CIFAR100(root='./data', train=train, download=True, transform=transform)
    elif dataset_name == 'SVHN':
        dataset = datasets.SVHN(root='./data', split='train' if train else 'test', download=True, transform=transform)
    elif dataset_name == 'STL10':
        dataset = datasets.STL10(root='./data', split='train' if train else 'test', download=True, transform=transform)
    elif dataset_name == 'TinyImageNet':
        dataset = datasets.ImageFolder(root='./data/tiny-imagenet-200/train' if train else './data/tiny-imagenet-200/val', transform=transform)
    else:
        raise ValueError(f'Unknown dataset: {dataset_name}')
    return dataset

# Next, let's define a function for calculating confidences
def calculate_confidences(model, data_loader):
    model.eval()
    confidences = []
    with torch.no_grad():
        for inputs, _ in data_loader:
            outputs = model(inputs)
            softmax_outputs = torch.nn.functional.softmax(outputs, dim=1)
            topk_confidences, _ = torch.topk(softmax_outputs, 2, dim=1)
            confidences.append(topk_confidences)
    confidences = torch.cat(confidences)
    return confidences[:, 0], confidences[:, 1]

# Now, let's define a function for calculating metrics
def calculate_metrics(confidences):
    mean_confidence = confidences.mean().item()
    variance_confidence = confidences.var().item()
    return mean_confidence, variance_confidence

# Let's define a function for conformal quantile regression
def conformal_quantile_regression(X_train, y_train, X_test, y_test, alpha=0.1):
    model = GradientBoostingRegressor()
    nc = RegressorNc(model, AbsErrorErrFunc())
    icp = IcpRegressor(nc)
    icp.fit(X_train, y_train)
    prediction = icp.predict(X_test, significance=1-alpha)
    coverage = helper.compute_coverage(prediction, y_test)
    length = helper.compute_length(prediction)
    return coverage, length

# Now we can start our experiment
model = models.resnet50(pretrained=True)
model.to('cuda')

# Let's assume we have these in-distribution and out-of-distribution datasets
in_distribution_datasets = ['CIFAR10', 'CIFAR100', 'SVHN', 'STL10', 'TinyImageNet']
out_of_distribution_datasets = ['SVHN', 'STL10', 'TinyImageNet', 'CIFAR10', 'CIFAR100']

# Prepare plots for metrics
fig, axes = plt.subplots(nrows=len(in_distribution_datasets), ncols=2, figsize=(10, 5*len(in_distribution_datasets)))
axes = axes.flatten()

# Start experiments
for i, (in_dataset_name, out_dataset_name) in enumerate(zip(in_distribution_datasets, out_of_distribution_datasets)):
    # Load datasets
    in_dataset = load_dataset(in_dataset_name)
    out_dataset = load_dataset(out_dataset_name)
    in_data_loader = torch.utils.data.DataLoader(in_dataset, batch_size=64, shuffle=True)
    out_data_loader = torch.utils.data.DataLoader(out_dataset, batch_size=64, shuffle=True)

    # Calculate confidences
    in_confidences, _ = calculate_confidences(model, in_data_loader)
    out_confidences, _ = calculate_confidences(model, out_data_loader)

    # Calculate metrics
    in_mean_confidence, in_variance_confidence = calculate_metrics(in_confidences)
    out_mean_confidence, out_variance_confidence = calculate_metrics(out_confidences)

    # Print results
    print(f'In-distribution dataset: {in_dataset_name}')
    print(f'Out-of-distribution dataset: {out_dataset_name}')
    print(f'Difference of means: {in_mean_confidence - out_mean_confidence}')
    print(f'Difference of variances: {in_variance_confidence - out_variance_confidence}')

    # Plot confidence distributions
    axes[2*i].hist(in_confidences.cpu().numpy(), bins=100, alpha=0.5, label='In-distribution')
    axes[2*i].hist(out_confidences.cpu().numpy(), bins=100, alpha=0.5, label='Out-of-distribution')
    axes[2*i].legend()
    axes[2*i].set_title(f'Confidence distributions ({in_dataset_name} vs {out_dataset_name})')

    # Perform conformal quantile regression
    X_train, X_test, y_train, y_test = train_test_split(in_confidences.cpu().numpy().reshape(-1, 1), out_confidences.cpu().numpy(), test_size=0.2)
    coverage, length = conformal_quantile_regression(X_train, y_train, X_test, y_test)

    # Plot regression results
    axes[2*i+1].scatter(X_test, y_test, s=10, label='Test data')
    axes[2*i+1].plot(X_test, coverage, color='r', label='Conformal quantile regression')
    axes[2*i+1].legend()
    axes[2*i+1].set_title(f'Conformal quantile regression ({in_dataset_name} vs {out_dataset_name})')

# Show plots
plt.tight_layout()
plt.show()
