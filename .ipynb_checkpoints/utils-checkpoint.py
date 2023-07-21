import urllib
import json
import numpy as np
import os


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