import torch
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast, AdamW
from datasets import load_dataset
import numpy as np
from sklearn.utils import resample
import matplotlib.pyplot as plt
import os
import argparse
from datetime import datetime
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, accuracy_score
from pandas import DataFrame
import pandas as pd


def calculate_metrics(y_true, y_score):
    auroc = roc_auc_score(y_true, y_score)
    mean_precision = average_precision_score(y_true, y_score)
    tn, fp, fn, tp = confusion_matrix(y_true, np.round(y_score)).ravel()
    accuracy = accuracy_score(y_true, np.round(y_score))
    return accuracy, auroc, mean_precision, tp, fn, fp, tn

class ReviewsDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def preprocess_data(data, tokenizer):
    encodings = tokenizer(data['text'], truncation=True, padding=True)
    labels = data['label']
    return ReviewsDataset(encodings, labels)

def check_class_balance(labels):
    unique, counts = np.unique(labels, return_counts=True)
    min_count = min(counts)
    indices_to_keep = []
    for class_label in unique:
        indices = np.where(np.array(labels) == class_label)[0]
        indices_to_keep.extend(np.random.choice(indices, size=min_count, replace=False))
    return indices_to_keep

def train(model, dataset, optimizer, device):
    loader = DataLoader(dataset, batch_size=16*torch.cuda.device_count(), shuffle=True)
    model = torch.nn.DataParallel(model)
    model.train()
    for batch in tqdm(loader, desc="Training"):
        optimizer.zero_grad()
        outputs = model(**{k: v.to(device) for k, v in batch.items()})
        loss = outputs[0].mean()  # Take the mean of the losses
        loss.backward()
        optimizer.step()

def get_confidences(model, dataset, device, num_samples_per_class=None):
    if num_samples_per_class is not None:
        class_0_indices = [i for i in range(len(dataset)) if dataset[i]['labels'].item() == 0]
        class_1_indices = [i for i in range(len(dataset)) if dataset[i]['labels'].item() == 1]

        # If there are not enough samples in any of the classes, raise an exception
        if num_samples_per_class > len(class_0_indices) or num_samples_per_class > len(class_1_indices):
            raise ValueError("Not enough samples in one or more classes")

        selected_indices = np.concatenate([np.random.choice(class_0_indices, size=num_samples_per_class, replace=False),
                                           np.random.choice(class_1_indices, size=num_samples_per_class, replace=False)])
        
        dataset = torch.utils.data.Subset(dataset, selected_indices)

    loader = DataLoader(dataset, batch_size=16*torch.cuda.device_count())
    model = torch.nn.DataParallel(model)
    model.eval()
    confidences = []
    y_true = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Inference"):
            outputs = model(**{k: v.to(device) for k, v in batch.items()})
            confidences.extend(outputs.logits.softmax(-1).max(-1).values.cpu().numpy())
            y_true.extend(batch['labels'].cpu().numpy())
    return confidences, y_true


def main(args):
    base_exp_dir = args.base_exp_dir
    exp_name = args.exp_name

    # Experiment name
    curr_time = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    exp_name = exp_name + '_' + curr_time
    exp_dir = os.path.join(base_exp_dir, exp_name)
    results_dir = os.path.join(exp_dir, 'results')
    models_dir = os.path.join(exp_dir, 'models')
    
    # Create experiment directory
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Load data
    imdb_dataset = load_dataset('imdb', split='train')
    amazon_dataset = load_dataset('amazon_polarity', split='train[:1%]')

    # Load tokenizer and model
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2).to(device)
    model = torch.nn.DataParallel(model).to(device) # Wrap model for multi-GPU use

    # Preprocessing IMDB dataset
    imdb_encodings = tokenizer(imdb_dataset['text'], truncation=True, padding=True)
    imdb_labels = imdb_dataset['label']
    imdb_dataset = ReviewsDataset(imdb_encodings, imdb_labels)

    # Split the data
    imdb_train_dataset, imdb_test_dataset = random_split(imdb_dataset, [int(0.8*len(imdb_dataset)), int(0.2*len(imdb_dataset))])

    # Now make them balanced
    imdb_train_indices = check_class_balance([imdb_train_dataset[i]['labels'].item() for i in range(len(imdb_train_dataset))])
    imdb_train_dataset = torch.utils.data.Subset(imdb_train_dataset, imdb_train_indices)

    imdb_test_indices = check_class_balance([imdb_test_dataset[i]['labels'].item() for i in range(len(imdb_test_dataset))])
    imdb_test_dataset = torch.utils.data.Subset(imdb_test_dataset, imdb_test_indices)

    # Preprocessing Amazon dataset
    amazon_indices = check_class_balance(amazon_dataset['label'])
    amazon_encodings = tokenizer(amazon_dataset['content'], truncation=True, padding=True)
    amazon_labels = amazon_dataset['label']
    amazon_dataset = ReviewsDataset({key: np.array(val)[amazon_indices] for key, val in amazon_encodings.items()}, np.array(amazon_labels)[amazon_indices])

    # Bootstrapping
    bootstrap_models = []
    for i in range(args.bootstrap_num):
        sampled_indices = resample(np.arange(len(imdb_train_dataset)), replace=True)
        bootstrap_dataset = torch.utils.data.Subset(imdb_train_dataset, sampled_indices)
        model_copy = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2).to(device)
        optimizer = AdamW(model_copy.parameters())
        train(model_copy, bootstrap_dataset, optimizer, device)
        bootstrap_models.append(model_copy)

        # Save model
        model_path = os.path.join(models_dir, f'model_bootstrap_{i}')
        model_copy.save_pretrained(model_path)

    # Calculate number of samples per class
    num_samples_per_class = min(len([i for i in range(len(imdb_test_dataset)) if imdb_test_dataset[i]['labels'].item() == 0]),
                                len([i for i in range(len(imdb_test_dataset)) if imdb_test_dataset[i]['labels'].item() == 1]),
                                len([i for i in range(len(amazon_dataset)) if amazon_dataset[i]['labels'].item() == 0]),
                                len([i for i in range(len(amazon_dataset)) if amazon_dataset[i]['labels'].item() == 1]))

    # Initialize DataFrames for storing metrics
    id_metrics_df = DataFrame(columns=['bootstrap_epoch', 'accuracy', 'AUROC', 'mean_precision', 'TP', 'FN', 'FP', 'TN', 'avg_confidence'])
    ood_metrics_df = DataFrame(columns=['bootstrap_epoch', 'accuracy', 'AUROC', 'mean_precision', 'TP', 'FN', 'FP', 'TN', 'avg_confidence'])

    # Initialize lists to store confidence differences for each bootstrap
    doc_list = []

    for i, model in enumerate(bootstrap_models):
        # Get confidences and true labels
        imdb_confidences, imdb_y_true = get_confidences(model, imdb_test_dataset, device, num_samples_per_class)
        amazon_confidences, amazon_y_true = get_confidences(model, amazon_dataset, device, num_samples_per_class)

        # Calculate average confidence and difference of confidences and append to the list
        id_avg_confidence = np.mean(imdb_confidences)
        ood_avg_confidence = np.mean(amazon_confidences)
        doc = id_avg_confidence - ood_avg_confidence
        doc_list.append(doc)

        # Compute metrics
        id_metrics = calculate_metrics(imdb_y_true, imdb_confidences)
        ood_metrics = calculate_metrics(amazon_y_true, amazon_confidences)

        # Create new DataFrames for this epoch's metrics and include average confidence
        id_metrics_df_epoch = DataFrame([dict(zip(id_metrics_df.columns, [i] + list(id_metrics) + [id_avg_confidence]))])
        ood_metrics_df_epoch = DataFrame([dict(zip(ood_metrics_df.columns, [i] + list(ood_metrics) + [ood_avg_confidence]))])

        # Concatenate new metrics DataFrames to the existing ones
        id_metrics_df = pd.concat([id_metrics_df, id_metrics_df_epoch], ignore_index=True)
        ood_metrics_df = pd.concat([ood_metrics_df, ood_metrics_df_epoch], ignore_index=True)

    # Save metrics to CSV
    id_metrics_df.to_csv(os.path.join(results_dir, 'id_bootstrap_metrics.csv'), index=False)
    ood_metrics_df.to_csv(os.path.join(results_dir, 'ood_bootstrap_metrics.csv'), index=False)


    # Plotting function
    def plot_metrics(df, title, filename):
        plt.figure(figsize=(10, 5))
        for metric in ['accuracy', 'AUROC', 'mean_precision', 'TP', 'FN', 'FP', 'TN']:
            plt.plot(df['bootstrap_epoch'], df[metric], label=metric)
        plt.xlabel('Bootstrap epoch')
        plt.ylabel('Metric value')
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, filename))

    # Plot metrics
    plot_metrics(id_metrics_df, 'In distribution bootstrap metrics', 'id_bootstrap_metrics_plot.png')
    plot_metrics(ood_metrics_df, 'Out of distribution bootstrap metrics', 'ood_bootstrap_metrics_plot.png')

    # Plotting
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.hist(doc_list, bins=30, edgecolor='black')
    plt.title('Histogram of Mean Difference of Confidences')
    plt.xlabel('Mean DoC')
    plt.ylabel('Frequency')

    plt.subplot(1, 2, 2)
    plt.boxplot(doc_list)
    plt.title('Boxplot of Mean Difference of Confidences')

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'DoC_distribution.png'))




    # for model in bootstrap_models:
    #     imdb_confidences.append(get_confidences(model, imdb_test_dataset, device, num_samples_per_class))
    #     amazon_confidences.append(get_confidences(model, amazon_dataset, device, num_samples_per_class))
    
    # # Calculate the mean confidence for each dataset per each bootstrap
    # imdb_mean_confidences = [np.mean(confidences) for confidences in imdb_confidences]
    # amazon_mean_confidences = [np.mean(confidences) for confidences in amazon_confidences]

    # # Calculate the difference of confidences
    # doc = np.array(imdb_mean_confidences) - np.array(amazon_mean_confidences)

    # # Plotting
    # plt.figure(figsize=(10, 5))

    # plt.subplot(1, 2, 1)
    # plt.hist(doc, bins=30, edgecolor='black')
    # plt.title('Histogram of Mean Difference of Confidences')
    # plt.xlabel('Mean DoC')
    # plt.ylabel('Frequency')

    # plt.subplot(1, 2, 2)
    # plt.boxplot(doc)
    # plt.title('Boxplot of Mean Difference of Confidences')

    # plt.tight_layout()
    # plt.savefig(os.path.join(results_dir, 'DoC_distribution.png'))

    # # Create a data frame to store metrics
    # id_metrics_df = DataFrame(columns=['bootstrap_epoch', 'AUROC', 'mean_precision', 'TP', 'FN', 'FP', 'TN'])
    # ood_metrics_df = DataFrame(columns=['bootstrap_epoch', 'AUROC', 'mean_precision', 'TP', 'FN', 'FP', 'TN'])


    # for i, model in enumerate(bootstrap_models):
    #     imdb_confidences = get_confidences(model, imdb_test_dataset, device, num_samples_per_class)
    #     amazon_confidences = get_confidences(model, amazon_dataset, device, num_samples_per_class)

    #     imdb_y_true = [imdb_test_dataset[i]['labels'].item() for i in range(len(imdb_test_dataset))]
    #     amazon_y_true = [amazon_dataset[i]['labels'].item() for i in range(len(amazon_dataset))]

    #     imdb_auroc, imdb_mean_precision, imdb_tp, imdb_fn, imdb_fp, imdb_tn = calculate_metrics(imdb_y_true, imdb_confidences)
    #     amazon_auroc, amazon_mean_precision, amazon_tp, amazon_fn, amazon_fp, amazon_tn = calculate_metrics(amazon_y_true, amazon_confidences)

    #     id_metrics_df = id_metrics_df.append({
    #         'bootstrap_epoch': i,
    #         'AUROC': imdb_auroc,
    #         'mean_precision': imdb_mean_precision,
    #         'TP': imdb_tp,
    #         'FN': imdb_fn,
    #         'FP': imdb_fp,
    #         'TN': imdb_tn
    #     }, ignore_index=True)

    #     ood_metrics_df = ood_metrics_df.append({
    #         'bootstrap_epoch': i,
    #         'AUROC': amazon_auroc,
    #         'mean_precision': amazon_mean_precision,
    #         'TP': amazon_tp,
    #         'FN': amazon_fn,
    #         'FP': amazon_fp,
    #         'TN': amazon_tn
    #     }, ignore_index=True)

    # id_metrics_df.to_csv(os.path.join(results_dir, 'id_bootstrap_metrics.csv'), index=False)
    # ood_metrics_df.to_csv(os.path.join(results_dir, 'id_bootstrap_metrics.csv'), index=False)

    # plt.figure(figsize=(10, 5))

    # for metric in ['AUROC', 'mean_precision', 'TP', 'FN', 'FP', 'TN']:
    #     plt.plot(id_metrics_df['bootstrap_epoch'], id_metrics_df[metric], label=metric)

    # plt.xlabel('Bootstrap epoch')
    # plt.ylabel('Metric value')
    # plt.title('In distribution bootstrap metrics')
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig(os.path.join(results_dir, 'id_bootstrap_metrics_plot.png'))

    # plt.figure(figsize=(10, 5))

    # for metric in ['AUROC', 'mean_precision', 'TP', 'FN', 'FP', 'TN']:
    #     plt.plot(ood_metrics_df['bootstrap_epoch'], ood_metrics_df[metric], label=metric)

    # plt.xlabel('Bootstrap epoch')
    # plt.ylabel('Metric value')
    # plt.title('Out of distribution bootstrap metrics')
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig(os.path.join(results_dir, 'ood_bootstrap_metrics_plot.png'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_exp_dir', type=str, default='/home/lamitay/experiments', help='Output directory')
    parser.add_argument('--exp_name', type=str, default='NLP_doc_multi_gpu', help='Experiment name')
    parser.add_argument('--bootstrap_num', type=int, default=5, help='Number of bootstrap iterations')
    args = parser.parse_args()
    main(args)
