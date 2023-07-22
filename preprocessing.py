from torchvision.transforms import Compose, Resize, ToTensor, Normalize, CenterCrop
import torchvision.models as models

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


def preprocess_images_imagenet_r():
    # The values were taken from here: https://github.com/hendrycks/imagenet-r/blob/master/eval.py
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # Compose transforms
    transforms = Compose(
        [Resize(256),
         CenterCrop(224),
         ToTensor(),
         Normalize(mean, std)
         ])
    return transforms



# Define a function for preprocessing images for any dataset
def preprocess_images_any_dataset(model_name):
    if model_name == 'resnet50':
        return models.ResNet50_Weights.IMAGENET1K_V2.transforms()
    if model_name == 'resnet18':
        return models.ResNet18_Weights.IMAGENET1K_V1.transforms()
    if model_name == 'resnet34':
        return models.ResNet34_Weights.IMAGENET1K_V1.transforms()
    if model_name == 'resnet101':
        return models.ResNet101_Weights.IMAGENET1K_V1.transforms()
    if model_name == 'resnet152':
        return models.ResNet152_Weights.IMAGENET1K_V1.transforms()
    elif model_name == 'vgg16':
        return models.VGG16_Weights.IMAGENET1K_V1.transforms()
    elif model_name == 'vgg19':
        return models.VGG19_Weights.IMAGENET1K_V1.transforms()
    elif model_name == 'alexnet':
        return models.AlexNet_Weights.IMAGENET1K_V1.transforms()
    elif model_name == 'resnext':
        return models.ResNeXt101_32X8D_Weights.IMAGENET1K_V2.transforms()
    elif model_name == 'wide_resnet':
        return models.Wide_ResNet101_2_Weights.IMAGENET1K_V2.transforms()
    elif model_name == 'densenet121':
        return models.DenseNet121_Weights.IMAGENET1K_V1.transforms()
    elif model_name == 'googlenet':
        return models.GoogLeNet_Weights.IMAGENET1K_V1.transforms()
    elif model_name == 'mobilenet_v2':
        return models.MobileNet_V2_Weights.IMAGENET1K_V2.transforms()
    else:
        raise ValueError('Invalid model name')