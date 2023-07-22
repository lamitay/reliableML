import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
import argparse
# from dataloader import *
from models import load_model
from metrics import *
from dirs import *
from utils import *
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
import json
import urllib
from preprocessing import preprocess_images_any_dataset


def main():
    parser = argparse.ArgumentParser(description="Evaluate pretrained models.")
    parser.add_argument("--user", type=str, default='amitay',
                        help="The name of the dataset to use.")
    parser.add_argument("--data_name", type=str, default='imagenet',
                        help="The name of the dataset to use.")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Batch size for data loading.")
    # parser.add_argument("--model_names", type=str, nargs='+',
    #                     default=['resnet50', 'resnet18', 'resnet34', 'resnet101', 'resnet152', 'vgg16', 'vgg19', 'alexnet', 'resnext', 'wide_resnet', 'densenet121', 'googlenet', 'mobilenet_v2'],
    #                     help="The names of the models to evaluate.")
    parser.add_argument("--model_names", type=str, nargs='+',
                        default=['vgg16'],
                        help="The names of the models to evaluate.")
    
    args = parser.parse_args()

    # Set reproducibility variables
    seed = 42
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    np.random.seed(seed)

    user = args.user
    if user == 'amitay':
        base_exp_dir = '/home/lamitay/experiments/'
    else:
        base_exp_dir = '/home/davidva/experiments_David'

    data_name = args.data_name
    batch_size = args.batch_size
    model_names = args.model_names
    val_dir = get_val_dir(data_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    

    # Load imagenet labels as real classes
    imagenet_labels = load_imagenet_labels()  # 'tench', 'goldfish', 'great white shark', 'tiger shark', 'hammerhead shark'...

    if data_name in ['imagenet', 'imagenet-v2', 'imagenet-sketch']:
        ds_specific_mask = [True] * 1000
        ds_specific_labels = imagenet_labels  # the classes of the specific dataset that we're working with
    elif data_name == 'imagenet-r':
        ds_specific_r_mask = class_names.imagenet_r_mask
        ds_specific_labels = class_names.imagenet_r_labels
    elif data_name == 'imagenet-a':  # note: if the hidden file .ipynb_checkpoints is inside of the folder of the imagenet-a data,
                                     # it should be removed from there in order to create the DataFolder object
        ds_specific_mask = class_names.imagenet_a_mask
        ds_specific_labels = class_names.imagenet_a_labels
        
    # model_names = ['resnet50', 'resnet18', 'resnet34', 'resnet101', 'resnet152', 'vgg16', 'vgg19', 'alexnet', 'resnext', 'wide_resnet', 'densenet121', 'googlenet', 'mobilenet_v2']

    for model_name in tqdm(model_names):
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

        # # Define transformations for the input images
        # transform = transforms.Compose([
        #     transforms.Resize(256),
        #     transforms.CenterCrop(224),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        # ]) 

        # Load the ImageNet validation dataset with defined transformations
        val_dataset = ImageFolder(val_dir, transform=preprocess_images_any_dataset(model_name))
        # val_dataset = ImageFolder(val_dir, transform=transform)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=5)
        

        # Load model
        model = load_model(model_name).to(device)

        model.eval()
        
        # Register the hook at the penultimate layer of the model
        if model_name in ['mobilenet_v2']:
            layer = model.features
        elif model_name in ['densenet121']:
            layer = model.classifier
        elif model_name in ['vgg16', 'vgg19']:
            layer = model.classifier[-2]
        else:
            layer = model.avgpool
        
        # Prepare DataFrame to store results
        results = []
        total_correct = 0
        batch_num = 0
        all_entropies = np.empty([0])
        
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(tqdm(val_loader, desc=f"Processing {exp_name}")):
                images = images.to(device)
                labels = labels.to(device)

                # handle = layer.register_forward_hook(get_embedding_hook(batch_idx, batch_size, embed_dir))

                # Run forward pass through the model and calculate probabilities
                output = model(images)
                probabilities = torch.nn.functional.softmax(output, dim=1)
                probabilities = probabilities * torch.tensor(ds_specific_mask).to(device)  # using only the classes that are in both datasets
    
                # Get top-5 predictions and confidences
                top5_prob, top5_classes = torch.topk(probabilities, 5)
                entropies = scipy.stats.entropy(probabilities.detach().cpu().numpy(), axis=1)
                all_entropies = np.concatenate((all_entropies, entropies))  # might be really wasteful in terms of run time

                # Iterate over each image in the batch
                for i in range(len(images)):
                    image_idx = batch_idx * batch_size + i
                    image_path = val_dataset.imgs[i + batch_num * batch_size][0]
                    true_class = ds_specific_labels[labels[i].item()]  # e.g. labels[i].item()=0. true_class=goldfish
                    # true_class = imagenet_labels[labels[i].item()]  # e.g. labels[i].item()=0. true_class=goldfish
                    predictions = {"image_path": image_path,
                                   "true_class": true_class,
                                   "embeddings_path":  os.path.join(embed_dir, f"{image_idx}_embeddings.npy")}
                    
                    for j in range(5):
                        prediction_class = imagenet_labels[top5_classes[i, j].item()]
                        predictions[f"top{j+1}_prediction_class"] = prediction_class
                        predictions[f"top{j+1}_confidence"] = top5_prob[i, j].item()
    
                        # Check if top-1 prediction is correct
                        if j == 0 and prediction_class == true_class:
                            total_correct += 1

                        # if j==0:
                        #     if prediction_class == true_class:
                        #         total_correct += 1
                        #         print(f"Prediction CORRECTA. Prediction: {prediction_class} ---> True class: {true_class}")
                        #     else:
                        #         print(f"Prediction WRONGA. Prediction: {prediction_class} ---> True class: {true_class}")

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
        average_entropy = np.mean(all_entropies)
        top2_grad_confidence = (results_df["top1_confidence"] - results_df["top2_confidence"]).mean()

        
        # Save average and std confidences in a new DataFrame
        confidence_df = pd.DataFrame({
            "Accuracy": [accuracy],
            "Average Confidence": [avg_confidence],
            "Standard Deviation Confidence": [std_confidence],
            "Average Entropy": [average_entropy],
            "top2_grad_confidence": [top2_grad_confidence]
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
        # break

if __name__ == '__main__':
    main()

# Usage exapmle
# python main_predict_dataset_all_models.py --user 'david' --data_name 'imagenet-r' -- batch_size 128 --model_names resnet50 vgg16
