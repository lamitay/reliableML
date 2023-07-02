import torchvision.models as models

# Define a function for loading pre-trained models
def load_model(model_name):
    if model_name == 'resnet50':
        return models.resnet50(pretrained=True)
    elif model_name == 'vgg16':
        return models.vgg16(pretrained=True)
    elif model_name == 'densenet121':
        return models.densenet121(pretrained=True)
    elif model_name == 'googlenet':
        return models.googlenet(pretrained=True)
    elif model_name == 'mobilenet_v2':
        return models.mobilenet_v2(pretrained=True)
    else:
        raise ValueError('Invalid model name')
