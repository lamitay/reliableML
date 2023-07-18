import os

def get_val_dir(data_name):
    
    student_name = 'davidva' if 'david' in os.getcwd() else 'lamitay'
    
    if data_name == 'imagenet':
        val_dir = '/datasets/ImageNet/val/'
    elif data_name == 'imagenet-r':
        val_dir = '/home/' + student_name + '/datasets/imagenet-r'
    elif data_name == 'imagenet-a':
        val_dir = '/home/' + student_name + '/datasets/imagenet-a'
    elif data_name == 'imagenet-sketch':
        val_dir = '/home/' + student_name + '/datasets/imagenetsketch/sketch/'
    elif data_name == 'imagenet-vid-robust':
        val_dir = '/home/' + student_name + '/datasets/imagenet-a'
    elif data_name == 'imagenet-v2-top':
        val_dir = '/home/' + student_name + '/datasets/imagenetv2-top-images-format-val'
    elif data_name == 'imagenet-v2-threshold':
        val_dir = '/home/' + student_name + '/datasets/imagenetv2-threshold0.7-format-val'
    elif data_name == 'imagenet-v2-matched':
        val_dir = '/home/' + student_name + '/datasets/imagenetv2-matched-frequency-format-val'
    else:
        raise ValueError('Invalid dataset name')

    return val_dir