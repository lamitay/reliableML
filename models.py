import torchvision.models as models

# Define a function for loading pre-trained models
def load_model(model_name):
    if model_name == 'resnet50':
        return models.resnet50(weights="IMAGENET1K_V2")
    if model_name == 'resnet18':
        return models.resnet18(pretrained=True)
    if model_name == 'resnet34':
        return models.resnet34(pretrained=True)
    if model_name == 'resnet101':
        return models.resnet101(pretrained=True)
    if model_name == 'resnet152':
        return models.resnet152(pretrained=True)
    if model_name == 'resnet50':
        return models.resnet50(pretrained=True)
    if model_name == 'resnet50':
        return models.resnet50(pretrained=True)
    elif model_name == 'vgg16':
        return models.vgg16(pretrained=True)
    elif model_name == 'vgg19':
        return models.vgg19(pretrained=True)
    elif model_name == 'alexnet':
        return models.alexnet(pretrained=True)
    elif model_name == 'resnext':
        return models.resnext101_32x8d(pretrained=True)
    elif model_name == 'wide_resnet':
        return models.wide_resnet101_2(pretrained=True)
    elif model_name == 'densenet121':
        return models.densenet121(pretrained=True)
    elif model_name == 'googlenet':
        return models.googlenet(pretrained=True)
    elif model_name == 'mobilenet_v2':
        return models.mobilenet_v2(pretrained=True)
    else:
        raise ValueError('Invalid model name')