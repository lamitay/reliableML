from torchvision.transforms import Compose, Resize, ToTensor, Normalize

# Define a function for preprocessing images for CIFAR-100 and Tiny ImageNet
def preprocess_images_cifar_tinyimagenet():
    # Compose transforms
    transforms = Compose([
        Resize((224, 224)),  # Resize to match model's expected input size
        ToTensor(),
        Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))  # Normalize with CIFAR-100's mean and std
    ])
    return transforms

# Define a function for preprocessing images for STL-10 and Caltech datasets
def preprocess_images_stl_caltech():
    # Compose transforms
    transforms = Compose([
        Resize((224, 224)),  # Resize to match model's expected input size
        ToTensor(),
        Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to range [-1, 1]
    ])
    return transforms
