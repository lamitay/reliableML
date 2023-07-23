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
# def get_embedding_hook(batch_idx, batch_size, embed_dir):
#     """
#     Creates and returns a hook function to capture and save the model's embeddings.

#     Returns:
#         function: A hook function that can be registered to a PyTorch nn.Module.
#     """
#     def hook(module, input, output):
#         output = output.detach().cpu().numpy()
#         for i in range(output.shape[0]):
#             # Calculate the overall index of the image in the dataset
#             image_idx = batch_idx * batch_size + i
#             np.save(os.path.join(embed_dir, f"{image_idx}_embeddings.npy"), output[i])
#     return hook

def get_embedding_hook(batch_idx, batch_size, embed_dir):
    """
    Creates and returns a hook function to capture and save the model's embeddings.

    Returns:
        function: A hook function that can be registered to a PyTorch nn.Module.
    """
    def hook(module, input, output):
        # output = output.detach().cpu().numpy()
        output = output.detach()  # detach from computation graph
        if len(output.shape) > 2:
            output = output.mean([2, 3])  # If spatial dimensions exist, average across them
        output = output.cpu().numpy()  # convert to numpy array

        for i in range(output.shape[0]):
            # Calculate the overall index of the image in the dataset
            image_idx = batch_idx * batch_size + i
            np.save(os.path.join(embed_dir, f"{image_idx}_embeddings.npy"), output[i])
    return hook


def combine_csvs():
    # combines CSV files of results and adds columns of model and data names
    
    experiment_list = os.listdir('/home/davidva/experiments_David')
    experiment_list.remove('.ipynb_checkpoints')
    header_exists_flag = 0  # indicates whether the combines CSV file has a header of "Dataset, Model, Accuracy..."
    with open('/home/davidva/vscode_projects/reliableML/all_experiments.csv', 'w') as f:
        wf = csv.writer(f, delimiter = ',')
        
        # note that the next line means that I don't use all the experiments because there's always one folder which doesn't
        # have the results, when the experimenation is still running. That's why I used the index 4
        for folder in experiment_list[4:]:
            results_path = '/home/davidva/experiments_David/' + folder + '/results'
            file_list = os.listdir(results_path)
            end_of_data_name_index = folder.index('_')
            with open(results_path + '/' + file_list[2], mode="r") as csv_file:
                reader = csv.reader(csv_file) #this is the reader object
                line_count = 0
                for row_num, row in enumerate(reader):
                    if header_exists_flag == 0 and row_num == 0:
                        header_exists_flag = 1
                        row = ['Dataset', 'Model'] + row
                        wf.writerow(row)
                    if row_num == 1:
                        sub_strings = folder.split('_')
                        model_name = sub_strings[2] + '_resnet' if 'wide' in sub_strings[2] else sub_strings[2]
                        data_name = sub_strings[1]
                        row = [data_name, model_name] + row
                        wf.writerow(row)