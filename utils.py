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

def create_table_of_differences():
    df = pd.read_csv('all_experiments_excluding_imagenet_c.csv')
    datasets = df['Dataset'].unique()  # list of dataset names
    models = df['Model'].unique()  # list of dataset names
    
    for dataset in datasets:
        if dataset in ['imagenet', 'imagenet_with_labels_of_imagenet-r', 'imagenet_with_labels_of_imagenet-a']:
            continue
        for model in models:
            for i, col in enumerate(list(df.columns[2:])):
                shifted_col = i + 2
                ID_dataset_name = 'imagenet_with_labels_of_' + dataset if dataset in ['imagenet_with_labels_of_imagenet-r', 'imagenet_with_labels_of_imagenet-a'] else 'imagenet'
                # the line of the data of a specific OOD dataset and a specific model
                ID_value = df[(df['Dataset'] == ID_dataset_name) & (df['Model'] == model)].iloc[:, shifted_col].to_numpy()
                # the line of the data of a imagenet and a specific model
                OOD_value = df[(df['Dataset'] == dataset) & (df['Model'] == model)].iloc[:, shifted_col].to_numpy()
                # subtracting between the lines
                diff = ID_value - OOD_value
                # assigning the values to the line of OOD dataset
                df.loc[(df['Dataset'] == dataset) & (df['Model'] == model), df.columns[shifted_col]] = diff
    df = df[(df['Dataset'] != 'imagenet') & (df['Dataset'] != 'imagenet_with_labels_of_imagenet-a') & (df['Dataset'] != 'imagenet_with_labels_of_imagenet-r')]
    
    df.to_csv(path_or_buf = "all_experiments_excluding_imagenet_c_differences.csv")


def datasets_folders_names_change():
    # The function changes the names of some folders so it's easier later to create a table of results for the linear regressor while using the csv files in the folders to create the table
    
    folder_path = "/home/davidva/experiments_David"

    data_names = ['gaussian_noise', 'shot_noise', 'gaussian_blur', 'impulse_noise', 'speckle_noise', 'elastic_transform', 'jpeg_compression', 'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur']
    
    for data_name in data_names:
        for folder_name in os.listdir(folder_path):
            if data_name in folder_name:
                new_folder_name = folder_name.replace(data_name, data_name.replace("_", "-"))
                os.rename(os.path.join(folder_path, folder_name), os.path.join(folder_path, new_folder_name))


def create_results_tablbe():
    # the function creates a table of results that is used to fit a linear regressor. It's created using the csv files
    # that are inside of each results folder. Each folder corresponds to a model-dataset pair
    
    experiment_list = os.listdir('/home/davidva/experiments_David')
    experiment_list.remove('.ipynb_checkpoints')
    header_exists_flag = 0  # indicates whether the combined CSV file has a header of "Dataset, Model, Accuracy..."
    # print(experiment_list)
    with open('/home/davidva/vscode_projects/reliableML/all_experiments_excluding_imagenet_c.csv', 'w') as f:
        wf = csv.writer(f, delimiter = ',')
        for folder in experiment_list:
            results_path = '/home/davidva/experiments_David/' + folder + '/results'
            file_list = os.listdir(results_path)
            end_of_data_name_index = folder.index('_')
            if 'robust' in folder:
                continue
            index = 3 if ('imagenet-a' in folder or 'imagenet-r' in folder) else 2  # making sure that we take the correct file
            # ... in some folders the index of the file is different
            with open(results_path + '/' + file_list[index], mode="r") as csv_file:
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


def create_differential_results_table():
    # the function creates a table of results that is used to fit a linear regressor. The values in the table are
    # the differences between the in-distribution and out-of-distribution values of each model-dataset pair and for each
    # parameter (DoE, DoC, kurtosis, variance etc.). The table is created using the non-differential results table that
    # is created by the function create_results_tablbe()
    
    df = pd.read_csv('all_experiments_including_imagenet_c.csv')
    datasets = df['Dataset'].unique()  # list of dataset names
    models = df['Model'].unique()  # list of dataset names
    
    for dataset in datasets:
        if dataset in ['imagenet', 'imagenet_with_labels_of_imagenet-r', 'imagenet_with_labels_of_imagenet-a']:
            continue
        for model in models:
            for i, col in enumerate(list(df.columns[2:])):
                shifted_col = i + 2
                ID_dataset_name = 'imagenet_with_labels_of_' + dataset if dataset in ['imagenet_with_labels_of_imagenet-r', 'imagenet_with_labels_of_imagenet-a'] else 'imagenet'
                # the line of the data of a specific OOD dataset and a specific model
                ID_value = df[(df['Dataset'] == ID_dataset_name) & (df['Model'] == model)].iloc[:, shifted_col].to_numpy()
                # the line of the data of a imagenet and a specific model
                OOD_value = df[(df['Dataset'] == dataset) & (df['Model'] == model)].iloc[:, shifted_col].to_numpy()
                # subtracting between the lines
                diff = ID_value - OOD_value
                # assigning the values to the line of OOD dataset
                df.loc[(df['Dataset'] == dataset) & (df['Model'] == model), df.columns[shifted_col]] = diff
    df = df[(df['Dataset'] != 'imagenet') & (df['Dataset'] != 'imagenet_with_labels_of_imagenet-a') & (df['Dataset'] != 'imagenet_with_labels_of_imagenet-r')]
    
    df.to_csv(path_or_buf = "all_experiments_excluding_imagenet_c_differences.csv")