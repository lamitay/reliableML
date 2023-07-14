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
import json
import urllib
from sklearn.manifold import TSNE
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.io as pio


def load_imagenet_labels():
    """
    Fetches the ImageNet class labels from a JSON file hosted online.

    Returns:
        list: A list of class labels where the index corresponds to the class ID in the ImageNet dataset.
    """
    classes_text = 'https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json'
    with urllib.request.urlopen(classes_text) as url:
        imagenet_labels = json.loads(url.read().decode())
    return imagenet_labels

# Register hook to capture the model's embeddings
def get_embedding_hook(batch_idx, batch_size, embed_dir):
    """
    Creates and returns a hook function to capture and save the model's embeddings.

    Returns:
        function: A hook function that can be registered to a PyTorch nn.Module.
    """
    def hook(module, input, output):
        output = output.detach().cpu().numpy()
        for i in range(output.shape[0]):
            # Calculate the overall index of the image in the dataset
            image_idx = batch_idx * batch_size + i
            np.save(os.path.join(embed_dir, f"{image_idx}_embeddings.npy"), output[i])
    return hook


def main(args):  
    # Define device, batch size and directory path for ImageNet validation set
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    batch_size = args.batch_size
    val_dir = args.val_dir
    base_exp_dir = args.base_exp_dir
    exp_base_name = args.exp_name
    model_name = args.model_name

    # Experiment name
    curr_time = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    exp_name = exp_base_name + '_' + model_name + '_' + curr_time

    exp_dir = os.path.join(base_exp_dir, exp_name)
    results_dir = os.path.join(exp_dir, 'results')
    embed_dir = os.path.join(results_dir, 'val_embeddings')

    # Create experiment directory
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(embed_dir, exist_ok=True)


    # Update according to model name
    if model_name == 'resnet50_V1':
        resize_im = 256
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1).to(device)
        # Register the hook at the avgpool layer of the model
        layer = model.avgpool
        
    elif model_name == 'resnet50_V2':
        resize_im = 232
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2).to(device)
    elif model_name == 'resnet101_V1':
        resize_im = 256
        model = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1).to(device)
        
    model.eval()

    # Define transformations for the input images
    transform = transforms.Compose([
        transforms.Resize(resize_im),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load the ImageNet validation dataset with defined transformations
    val_dataset = ImageFolder(val_dir, transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=6)

    # Load imagenet labels as real classes
    imagenet_labels = load_imagenet_labels()

    # Prepare DataFrame to store results
    results = []
    total_correct = 0

    # Disable gradients for faster inference
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(tqdm(val_loader, desc=f"Processing {exp_name}")):
            images = images.to(device)
            labels = labels.to(device)

            handle = layer.register_forward_hook(get_embedding_hook(batch_idx, batch_size, embed_dir))

            # Run forward pass through the model and calculate probabilities
            output = model(images)
            probabilities = torch.nn.functional.softmax(output, dim=1)

            # Get top-5 predictions and confidences
            top5_prob, top5_classes = torch.topk(probabilities, 5)

            # Iterate over each image in the batch
            for i in range(len(images)):
                # Calculate the overall index of the image in the dataset
                image_idx = batch_idx * batch_size + i
                image_path = val_dataset.imgs[image_idx][0]
                true_class = imagenet_labels[labels[i].item()]
                predictions = {"image_path": image_path, 
                            "embeddings_path":  os.path.join(embed_dir, f"{image_idx}_embeddings.npy"),
                            "true_class": true_class}
                
                for j in range(5):                       
                    prediction_class = imagenet_labels[top5_classes[i, j].item()]
                    predictions[f"top{j+1}_prediction_class"] = prediction_class
                    predictions[f"top{j+1}_confidence"] = top5_prob[i, j].item()

                    # Check if top-1 prediction is correct
                    if j == 0 and prediction_class == true_class:
                        total_correct += 1 

                results.append(predictions)

            handle.remove()  # Unregister the hook after processing the batch

    # Convert results to DataFrame and save to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(results_dir, f"full_val_predictions_{exp_base_name}.csv"), index=False)

    # Calculate average and std of the top1 confidences across the val set
    avg_confidence = results_df["top1_confidence"].mean()
    std_confidence = results_df["top1_confidence"].std()
    accuracy = total_correct / len(val_dataset)


    # Save average and std confidences in a new DataFrame
    confidence_df = pd.DataFrame({
        "Accuracy": [accuracy],
        "Average Confidence": [avg_confidence],
        "Standard Deviation Confidence": [std_confidence]
    })
    confidence_df.to_csv(os.path.join(results_dir, f"val_confidence_summary_{exp_base_name}.csv"), index=False)

    # Handle embeddings
    embeddings = np.array([np.load(file) for file in results_df['embeddings_path']])
    tsne = TSNE(n_components=2, random_state=0)

    np.save(os.path.join(results_dir, 'tot_embeddings_4D_np_array.npy'), embeddings)

    # assuming embeddings is a 4D numpy array
    embeddings = embeddings.reshape(embeddings.shape[0], -1)
    
    embeddings_2d = tsne.fit_transform(embeddings)

    # Create a new DataFrame for embeddings
    df_embeddings = pd.DataFrame(embeddings_2d, columns=['Dim1', 'Dim2'])
    df_embeddings['true_class'] = results_df['true_class']
    df_embeddings['predicted_class'] = results_df['top1_prediction_class']
    df_embeddings['image_path'] = results_df['image_path']

    # Save as CSV
    df_embeddings.to_csv(os.path.join(results_dir, f"val_embeddings_2d_{exp_base_name}.csv"), index=False)

    # Plot with Plotly
    fig = px.scatter(df_embeddings, x='Dim1', y='Dim2', color='true_class', hover_data=['predicted_class', 'image_path'])
    fig.show()

    # Save plot as PNG
    fig.write_image(os.path.join(results_dir, f"tSNE_embeddings_{exp_base_name}.png"))

    # Save plot as HTML
    pio.write_html(fig, os.path.join(results_dir, f"tSNE_embeddings_{exp_base_name}.html"))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="ImageNet Validation Set Prediction")
    parser.add_argument('--batch_size', default=128, type=int, help='Batch size for dataloader')
    parser.add_argument('--val_dir', default='/datasets/ImageNet/val', type=str, help='Path to ImageNet validation dataset')
    parser.add_argument('--base_exp_dir', default='/home/lamitay/experiments/', type=str, help='Path to ImageNet validation dataset')
    parser.add_argument('--exp_name', default='predict_imagenet', type=str, help='Experiment name')
    parser.add_argument('--model_name', default='resnet50_V1', type=str, help='Model name')

    args = parser.parse_args()

    main(args)