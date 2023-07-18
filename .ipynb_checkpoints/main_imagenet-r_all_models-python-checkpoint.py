import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'

from dataloader import *
from models import load_model
from metrics import *
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime
import scipy

import json
import urllib
from preprocessing import preprocess_images_any_dataset

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


def main():
    # Define device, batch size and directory path for ImageNet validation set
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    
    data_name = 'imagenet-r'
    in_Newton_flag = 'home' in os.getcwd()  # if it's true it means that I'm running the code on the server    
    base_exp_dir = '/home/davidva/experiments_David' if in_Newton_flag else 'C:/Users/David/PycharmProjects/reliableML/experiments_David'
    val_dir = '/home/davidva/datasets/imagenet-r' if in_Newton_flag else'E:/MLreliability_project/data/imagenet-r'
        
    # Define the batch size
    batch_size = 128

    # Load imagenet labels as real classes
    imagenet_labels = load_imagenet_labels()
    # get variables that are needed for adjusting the class difference between the two datasets
    from class_names import all_wnids, imagenet_r_wnids, imagenet_label_indices_in_imagenet_r
    imagenet_r_mask = [wnid in imagenet_r_wnids for wnid in all_wnids]
    imagenet_r_labels = np.array(imagenet_labels)[np.array(imagenet_r_mask)]

    model_names = ['resnet50', 'resnet18', 'resnet34', 'resnet101', 'resnet152', 'vgg16', 'vgg19', 'alexnet', 'resnext', 'wide_resnet', 'densenet121', 'googlenet', 'mobilenet_v2']
    
    for model_name in model_names:
        # Experiment name
        curr_time = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        exp_base_name = 'predict_' + data_name + '_' + model_name
        exp_name = exp_base_name + '_' + curr_time
        exp_dir = os.path.join(base_exp_dir, exp_name)
        results_dir = os.path.join(exp_dir, 'results')
        embed_dir = os.path.join(results_dir, 'val_embeddings')
    
        # Create experiment directory
        os.makedirs(exp_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(embed_dir, exist_ok=True)
            
        # Load the ImageNet validation dataset with defined transformations
        val_dataset = ImageFolder(val_dir, transform=preprocess_images_any_dataset(model_name))
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=5)

        # Load model
        model = load_model(model_name).to(device)
        
        # Register the hook at the penultimate layer of the model
        if model_name in ['densenet', 'mobilenet_v2']:
            layer = model.features  # TODO is it correct?
        else:
            layer = model.avgpool
        
        # Prepare DataFrame to store results
        results = []
        total_correct = 0
        batch_num = 0
        all_entropies = np.empty([0])
        
        # Disable gradients for faster inference
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(tqdm(val_loader, desc=f"Processing {exp_name}")):
                images = images.to(device)
                labels = labels.to(device)

                handle = layer.register_forward_hook(get_embedding_hook(batch_idx, batch_size, embed_dir))

                # Run forward pass through the model and calculate probabilities
                output = model(images)
                probabilities = torch.nn.functional.softmax(output, dim=1)
                # probabilities = probabilities[:, imagenet_r_mask]  # using only the classes that are in both datasets
                probabilities = probabilities * torch.tensor(imagenet_r_mask).to(device)  # using only the classes that are in both datasets
    
                # Get top-5 predictions and confidences
                top5_prob, top5_classes = torch.topk(probabilities, 5)
                entropies = scipy.stats.entropy(probabilities.detach().cpu().numpy(), axis=1)
                all_entropies = np.concatenate((all_entropies, entropies))  # might be really wasteful in terms of run time

                # # Iterate over each image in the batch
                # for i in range(len(images)):
                #     image_path = val_dataset.imgs[i][0]
                #     true_class = val_dataset.classes[labels[i].item()]
                #     predictions = {"image_path": image_path, "true_class": true_class}
                #     for j in range(5):
                #         predictions[f"top{j + 1}_prediction_class"] = val_dataset.classes[top5_classes[i, j].item()]
                #         predictions[f"top{j + 1}_confidence"] = top5_prob[i, j].item()
                #     results.append(predictions)
    
                # Iterate over each image in the batch
                for i in range(len(images)):
                    image_idx = batch_idx * batch_size + i
                    image_path = val_dataset.imgs[i + batch_num * batch_size][0]
                    # true_class = val_dataset.classes[labels[i].item()]
                    true_class = imagenet_r_labels[labels[i].item()]
                    predictions = {"image_path": image_path,
                                   "true_class": true_class,
                                   "embeddings_path":  os.path.join(embed_dir, f"{image_idx}_embeddings.npy")}
                    
                    for j in range(5):
                        # predictions[f"top{j+1}_prediction_class"] = val_dataset.classes[top5_classes[i, j].item()]
                        prediction_class = imagenet_labels[top5_classes[i, j].item()]
                        predictions[f"top{j+1}_prediction_class"] = prediction_class
                        predictions[f"top{j+1}_confidence"] = top5_prob[i, j].item()
    
                        # Check if top-1 prediction is correct
                        if j == 0 and prediction_class == true_class:
                            total_correct += 1
                    results.append(predictions)
                batch_num += 1
                handle.remove()  # Unregister the hook after processing the batch
        
        # Convert results to DataFrame and save to CSV
        results_df = pd.DataFrame(results)
        results_df.to_csv(os.path.join(results_dir, f"full_val_predictions_{exp_base_name}.csv"), index=False)
    
        # Calculate average and std of the top1 confidences across the val set2
        avg_confidence = results_df["top1_confidence"].mean()
        std_confidence = results_df["top1_confidence"].std()
        accuracy = total_correct / len(val_dataset)
        average_entropy = np.mean(all_entropies)
        
        # Save average and std confidences in a new DataFrame
        confidence_df = pd.DataFrame({
            "Accuracy": [accuracy],
            "Average Confidence": [avg_confidence],
            "Standard Deviation Confidence": [std_confidence],
            "Average Entropy": [average_entropy]
        })
        confidence_df.to_csv(os.path.join(results_dir, f"val_confidence_summary_{exp_base_name}.csv"), index=False)

        # Handle embeddings
        embeddings = np.array([np.load(file) for file in results_df['embeddings_path']])
        # assuming embeddings is a 4D numpy array
        embeddings = embeddings.reshape(embeddings.shape[0], -1)
        average, covariance = calculate_activation_statistics(embeddings)
        np.save(os.path.join(results_dir, 'embeddings_covariance.npy'), covariance)
        np.save(os.path.join(results_dir, 'embeddings_average.npy'), average)
        # np.save(os.path.join(results_dir, 'tot_embeddings_2D_np_array.npy'), embeddings)
        
        # Create a new DataFrame for embeddings
        # df_embeddings = pd.DataFrame(*******, columns=np.arange(0, embeddings.shape[1]))
        # df_embeddings['true_class'] = results_df['true_class']
        # df_embeddings['predicted_class'] = results_df['top1_prediction_class']
        # df_embeddings['image_path'] = results_df['image_path']
        # # Save as CSV
        # df_embeddings.to_csv(os.path.join(results_dir, f"val_embeddings_2d_{exp_base_name}.csv"), index=False)

if __name__ == '__main__':
    main()