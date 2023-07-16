import torchvision.models as models
 
# Define a function for loading pre-trained models
def load_model(model_name):
    weights = "IMAGENET1K_V2" if model_name in ['resnet50', 'resnext', 'wide_resnet', 'mobilenet_v2'] else "IMAGENET1K_V1"
    
    if model_name == 'resnet50':
        return models.resnet50(weights=weights)
    if model_name == 'resnet18':
        return models.resnet18(weights=weights)
    if model_name == 'resnet34':
        return models.resnet34(weights=weights)
    if model_name == 'resnet101':
        return models.resnet101(weights=weights)
    if model_name == 'resnet152':
        return models.resnet152(weights=weights)
    if model_name == 'resnet50':
        return models.resnet50(weights=weights)
    elif model_name == 'vgg16':
        return models.vgg16(weights=weights)
    elif model_name == 'vgg19':
        return models.vgg19(weights=weights)
    elif model_name == 'alexnet':
        return models.alexnet(weights=weights)
    elif model_name == 'resnext':
        return models.resnext101_32x8d(weights=weights)
    elif model_name == 'wide_resnet':
        return models.wide_resnet101_2(weights=weights)
    elif model_name == 'densenet121':
        return models.densenet121(weights=weights)
    elif model_name == 'googlenet':
        return models.googlenet(weights=weights)
    elif model_name == 'mobilenet_v2':
        return models.mobilenet_v2(weights=weights)
    else:
        raise ValueError('Invalid model name')