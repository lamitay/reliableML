import torch
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime
import os
import argparse

def main(args):
    
    # Define device, batch size and directory path for ImageNet validation set
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    batch_size = args.batch_size
    val_dir = args.val_dir
    base_exp_dir = args.base_exp_dir
    exp_base_name = args.exp_name

    # Experiment name
    curr_time = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    exp_name = exp_base_name + '_' + curr_time

    exp_dir = os.path.join(base_exp_dir, exp_name)
    results_dir = os.path.join(exp_dir, 'results')

    # Create experiment directory
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # Define transformations for the input images
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load the ImageNet validation dataset with defined transformations
    val_dataset = ImageFolder(val_dir, transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Load pretrained ResNet-50 model
    model = models.resnet50(pretrained=True).to(device)
    model.eval()

    # Prepare DataFrame to store results
    results = []

    # Disable gradients for faster inference
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc=f"Processing {exp_name}"):
            images = images.to(device)
            labels = labels.to(device)

            # Run forward pass through the model and calculate probabilities
            output = model(images)
            probabilities = torch.nn.functional.softmax(output, dim=1)

            # Get top-5 predictions and confidences
            top5_prob, top5_classes = torch.topk(probabilities, 5)

            # Iterate over each image in the batch
            for i in range(len(images)):
                image_path = val_dataset.imgs[i][0]
                true_class = val_dataset.classes[labels[i].item()]
                predictions = {"image_path": image_path, "true_class": true_class}
                for j in range(5):                       
                    predictions[f"top{j+1}_prediction_class"] = val_dataset.classes[top5_classes[i, j].item()]
                    predictions[f"top{j+1}_confidence"] = top5_prob[i, j].item()
                results.append(predictions)

    # Convert results to DataFrame and save to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(results_dir, f"full_val_predictions_{exp_base_name}.csv"), index=False)

    # Calculate average and std of the top1 confidences across the val set
    avg_confidence = results_df["top1_confidence"].mean()
    std_confidence = results_df["top1_confidence"].std()

    # Save average and std confidences in a new DataFrame
    confidence_df = pd.DataFrame({
        "Average Confidence": [avg_confidence],
        "Standard Deviation Confidence": [std_confidence]
    })
    confidence_df.to_csv(os.path.join(results_dir, f"val_confidence_summary_{exp_base_name}.csv"), index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="ImageNet Validation Set Prediction")
    parser.add_argument('--batch_size', default=64, type=int, help='Batch size for dataloader')
    parser.add_argument('--val_dir', default='/datasets/ImageNet/val', type=str, help='Path to ImageNet validation dataset')
    parser.add_argument('--base_exp_dir', default='/home/lamitay/experiments/', type=str, help='Path to ImageNet validation dataset')
    parser.add_argument('--exp_name', default='predict_imagenet_resnet50', type=str, help='Experiment name')
    
    args = parser.parse_args()

    main(args)