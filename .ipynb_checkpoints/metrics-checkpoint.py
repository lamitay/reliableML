import torch
import numpy as np

def calculate_confidences(model, dataloader):
    model.eval()
    confidences = []
    with torch.no_grad():
        for images, _ in dataloader:
            outputs = torch.softmax(model(images), dim=1)
            max_confidences, _ = torch.max(outputs, dim=1)
            confidences.extend(max_confidences.cpu().numpy())
    return confidences

def calculate_average_confidence(confidences):
    return sum(confidences) / len(confidences)

def calculate_difference_of_confidences(in_confidences, out_confidences):
    return calculate_average_confidence(in_confidences) - calculate_average_confidence(out_confidences)

def calculate_MM_value(b):
    return np.mean(np.array(b), 0)