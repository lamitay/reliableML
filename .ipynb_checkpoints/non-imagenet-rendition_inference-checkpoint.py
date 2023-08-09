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
import class_names
from utils import *

import json
import urllib
from preprocessing import preprocess_images_any_dataset

def main():
    # Define device, batch size and directory path for ImageNet validation set
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    
    base_exp_dir = '/home/davidva/experiments_David'
    
    # Set reproducibility variables
    seed = 42
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    np.random.seed(seed)
    
    # Define the batch size
    batch_size = 128

    # Load imagenet labels as real classes
    imagenet_labels = load_imagenet_labels()  # 'tench', 'goldfish', 'great white shark', 'tiger shark', 'hammerhead shark'...

    # data_names = ['imagenet', 'imagenetv2-matched-frequency-format-val', 'imagenetv2-threshold0.7-format-val', 'imagenetv2-top-images-format-val', 'imagenetsketch/sketch', 'imagenet-r', 'imagenet-a']
    data_names = ['speckle_noise', 'spatter', 'saturate', 'elastic_transform', 'pixelate', 'jpeg_compression', 'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur']

    for data_name in data_names:
        if data_name in ['imagenet', 'imagenetv2-matched-frequency-format-val', 'imagenetv2-threshold0.7-format-val', 'imagenetv2-top-images-format-val', 'imagenetsketch', 'gaussian_noise', 'shot_noise', 'gaussian_blur', 'impulse_noise', 'frost', 'snow', 'fog', 'contrast', 'brightness', 'speckle_noise', 'spatter', 'saturate', 'elastic_transform', 'pixelate', 'jpeg_compression', 'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur']:
            ds_specific_mask = [True] * 1000
            ds_specific_labels = imagenet_labels  # the classes of the specific dataset that we're working with
        elif data_name == 'imagenet-r':
            ds_specific_mask = class_names.imagenet_r_mask
            ds_specific_labels = class_names.imagenet_r_labels
        elif data_name == 'imagenet-a':  # note: if the hidden file .ipynb_checkpoints is inside of the folder of the imagenet-a data,
                                         # it should be removed from there in order to create the DataFolder object
            ds_specific_mask = class_names.imagenet_a_mask
            ds_specific_labels = class_names.imagenet_a_labels

        if data_name == 'imagenet':
            val_dir = '/datasets/ImageNet/val/'
        elif data_name == 'imagenetsketch':
            val_dir = '/home/davidva/datasets/imagenetsketch/sketch'
        elif data_name in ['gaussian_noise', 'shot_noise', 'gaussian_blur', 'impulse_noise', 'frost', 'snow', 'fog', 'contrast', 'brightness', 'speckle_noise', 'spatter', 'saturate', 'elastic_transform', 'pixelate', 'jpeg_compression', 'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur']:
            val_dir = '/home/davidva/datasets/' + data_name + '/3/'
        else:
            val_dir = '/home/davidva/datasets/' + data_name
        print('data_name is: ' + data_name + ' and val_dir is: ' + val_dir)
        
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
            model.eval()
            
            # Register the hook at the penultimate layer of the model
            if model_name in ['mobilenet_v2']:
                layer = model.classifier[0]
            elif model_name in ['densenet121']:
                layer = model.features  # model.features.norm5 sucks, features.denseblock3.denselayer24 is okay, features gives 1000-of-shape output, model.features.denseblock3.denselayer24.conv2 is okay
            elif model_name in ['vgg16', 'vgg19']:
                layer = model.classifier[-2]
            else:
                layer = model.avgpool
            
            # Prepare DataFrame to store results
            results = []
            total_correct = 0
            batch_num = 0
            all_entropies = np.empty([0])
            all_means = np.empty([0])
            all_variances = np.empty([0])
            all_skewnesses = np.empty([0])
            all_kurtosises = np.empty([0])
            all_top5_entropies = np.empty([0])
            all_top5_means = np.empty([0])
            all_top5_variances = np.empty([0])
            all_top5_skewnesses = np.empty([0])
            all_top5_kurtosises = np.empty([0])
    
            # Disable gradients for faster inference
            with torch.no_grad():
                for batch_idx, (images, labels) in enumerate(tqdm(val_loader, desc=f"Processing {exp_name}")):
                    images = images.to(device)
                    labels = labels.to(device)
    
                    # handle = layer.register_forward_hook(get_embedding_hook(batch_idx, batch_size, embed_dir))
    
                    # Run forward pass through the model
                    output = model(images)

                    # Create a tensor of the mask on the device
                    mask_tensor = torch.tensor(ds_specific_mask).to(device)

                    # Replace masked values in the output with negative infinity
                    if data_name in ['imagenet-a', 'imagenet-r', 'imagenet_vid_robust']:
                        output[:, ~mask_tensor] = float('-inf')

                    # Apply softmax
                    probabilities = torch.nn.functional.softmax(output, dim=1)

                    # Get top-5 predictions and confidences and some statistics
                    top5_prob, top5_classes = torch.topk(probabilities, 5)
                    _, _, top5_means, top5_variances, top5_skewnesses, top5_kurtosises = scipy.stats.describe(top5_prob.detach().cpu().numpy() , axis=1)
                    all_top5_means = np.concatenate((all_top5_means, top5_means))
                    all_top5_variances = np.concatenate((all_top5_variances, top5_variances))
                    all_top5_skewnesses = np.concatenate((all_top5_skewnesses, top5_skewnesses))
                    all_top5_kurtosises = np.concatenate((all_top5_kurtosises, top5_kurtosises))
                    top5_entropies = scipy.stats.entropy(top5_prob.detach().cpu().numpy(), axis=1)
                    all_top5_entropies = np.concatenate((all_top5_entropies, top5_entropies))
                    
                    _, _, means, variances, skewnesses, kurtosises = scipy.stats.describe(probabilities.detach().cpu().numpy() , axis=1)
                    all_means = np.concatenate((all_means, means))
                    all_variances = np.concatenate((all_variances, variances))
                    all_skewnesses = np.concatenate((all_skewnesses, skewnesses))
                    all_kurtosises = np.concatenate((all_kurtosises, kurtosises))
                    entropies = scipy.stats.entropy(probabilities.detach().cpu().numpy(), axis=1)
                    all_entropies = np.concatenate((all_entropies, entropies))
        
                    # Iterate over each image in the batch
                    for i in range(len(images)):
                        image_idx = batch_idx * batch_size + i
                        image_path = val_dataset.imgs[i + batch_num * batch_size][0]
                        # true_class = val_dataset.classes[labels[i].item()]
                        true_class = ds_specific_labels[labels[i].item()]  # e.g. labels[i].item()=0. true_class=goldfish
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
                    # handle.remove()  # Unregister the hook after processing the batch
            
            # Convert results to DataFrame and save to CSV
            results_df = pd.DataFrame(results)
            results_df.to_csv(os.path.join(results_dir, f"full_val_predictions_{exp_base_name}.csv"), index=False)
        
            # Calculate average and std of the top1 confidences across the val set2
            avg_confidence = results_df["top1_confidence"].mean()
            std_confidence = results_df["top1_confidence"].std()
            accuracy = total_correct / len(val_dataset)
            average_top5_mean = np.mean(all_top5_means)
            average_top5_variance = np.mean(all_top5_variances)
            average_top5_skewness = np.mean(all_top5_skewnesses)
            average_top5_kurtosis = np.mean(all_top5_kurtosises)
            average_top5_entropy = np.mean(all_top5_entropies)
            average_mean = np.mean(all_means)
            average_variance = np.mean(all_variances)
            average_skewness = np.mean(all_skewnesses)
            average_kurtosis = np.mean(all_kurtosises)
            average_entropy = np.mean(all_entropies)
            top2_grad_confidence = (results_df["top1_confidence"] - results_df["top2_confidence"]).mean()
    
            # Save average and std confidences in a new DataFrame and some more statistics
            confidence_df = pd.DataFrame({
                "Accuracy": [accuracy],
                "Average Confidence": [avg_confidence],
                "Standard Deviation Confidence": [std_confidence],
                "Average Entropy": [average_entropy],
                "Average mean": [average_mean],
                "Average variance": [average_variance],
                "Average skewness": [average_skewness],
                "Average kurtosis": [average_kurtosis],
                "Average top5 Entropy": [average_top5_entropy],
                "Average top5 mean": [average_top5_mean],
                "Average top5 variance": [average_top5_variance],
                "Average top5 skewness": [average_top5_skewness],
                "Average top5 kurtosis": [average_top5_kurtosis],
                "top2_grad_confidence": [top2_grad_confidence]
            })
            confidence_df.to_csv(os.path.join(results_dir, f"val_confidence_summary_{exp_base_name}.csv"), index=False)
    
            # # Handle embeddings
            # embeddings = np.array([np.load(file) for file in results_df['embeddings_path']])
            # # assuming embeddings is a 4D numpy array
            # embeddings = embeddings.reshape(embeddings.shape[0], -1)
            # average, covariance = calculate_activation_statistics(embeddings)
            # np.save(os.path.join(results_dir, 'embeddings_covariance.npy'), covariance)
            # np.save(os.path.join(results_dir, 'embeddings_average.npy'), average)
            # # np.save(os.path.join(results_dir, 'tot_embeddings_2D_np_array.npy'), embeddings)
            
            # Create a new DataFrame for embeddings
            # df_embeddings = pd.DataFrame(*******, columns=np.arange(0, embeddings.shape[1]))
            # df_embeddings['true_class'] = results_df['true_class']
            # df_embeddings['predicted_class'] = results_df['top1_prediction_class']
            # df_embeddings['image_path'] = results_df['image_path']
            # # Save as CSV
            # df_embeddings.to_csv(os.path.join(results_dir, f"val_embeddings_2d_{exp_base_name}.csv"), index=False)
            
            
if __name__ == '__main__':
    main()