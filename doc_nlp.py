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
    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    model.train()
    for batch in tqdm(loader, desc="Training"):
        optimizer.zero_grad()
        outputs = model(**{k: v.to(device) for k, v in batch.items()})
        loss = outputs.loss
        loss.backward()
        optimizer.step()

def get_confidences(model, dataset, device):
    loader = DataLoader(dataset, batch_size=16)
    model.eval()
    confidences = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Inference"):
            outputs = model(**{k: v.to(device) for k, v in batch.items()})
            confidences.extend(outputs.logits.softmax(-1).max(-1).values.cpu().numpy())
    return confidences

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

    # Get confidences on in-distribution (IMDB) and out-of-distribution (Amazon) data
    imdb_confidences = []
    amazon_confidences = []
    for model in bootstrap_models:
        imdb_confidences.append(get_confidences(model, imdb_test_dataset, device))
        amazon_confidences.append(get_confidences(model, amazon_dataset, device))

    # Get the difference of confidences
    doc = np.array(imdb_confidences) - np.array(amazon_confidences)

    # Plotting
    plt.boxplot(doc.transpose())
    plt.ylabel('Difference of Confidences')
    plt.savefig(os.path.join(results_dir, 'DoC_distribution.png'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_exp_dir', type=str, default='/home/lamitay/experiments', help='Output directory')
    parser.add_argument('--exp_name', type=str, default='NLP_doc', help='Experiment name')
    parser.add_argument('--bootstrap_num', type=int, default=5, help='Number of bootstrap iterations')
    args = parser.parse_args()
    main(args)
