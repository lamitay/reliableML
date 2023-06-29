import argparse
from dataloader import create_dataloader_cifar, create_dataloader, preprocess_images_cifar_tinyimagenet, preprocess_images_stl_caltech
from models import load_model
from metrics import calculate_confidences, calculate_average_confidence, calculate_difference_of_confidences

def main(args):
    # Define the batch size
    batch_size = 64

    # Create data loaders
    cifar_dataloader = create_dataloader_cifar(args.cifar_path, batch_size)
    tiny_imagenet_dataloader = create_dataloader(args.tiny_imagenet_path, batch_size, preprocess_images_cifar_tinyimagenet)
    stl_dataloader = create_dataloader(args.stl_path, batch_size, preprocess_images_stl_caltech)
    caltech101_dataloader = create_dataloader(args.caltech101_path, batch_size, preprocess_images_stl_caltech)
    caltech256_dataloader = create_dataloader(args.caltech256_path, batch_size, preprocess_images_stl_caltech)

    # Load model
    model = load_model(args.model)

    # Calculate confidences
    confidences = calculate_confidences(model, cifar_dataloader)

    # Calculate metrics
    avg_confidence = calculate_average_confidence(confidences)
    print(f'Average Confidence: {avg_confidence}')

    # Calculate difference of confidences
    doc = calculate_difference_of_confidences(cifar_dataloader, tiny_imagenet_dataloader)
    print(f'Difference of Confidences: {doc}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--cifar_path', type=str, required=True)
    parser.add_argument('--tiny_imagenet_path', type=str, required=True)
    parser.add_argument('--stl_path', type=str, required=True)
    parser.add_argument('--caltech101_path', type=str, required=True)
    parser.add_argument('--caltech256_path', type=str, required=True)
    args = parser.parse_args()
    main(args)

    # Usage example
    # python main.py --model resnet50 --cifar_path data/CIFAR-100 --tiny_imagenet_path data/Tiny-ImageNet --stl_path data/STL-10 --caltech101_path data/Caltech-101 --caltech256_path data/Caltech-256

