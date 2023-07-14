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


def main(args):  
    # Define device, batch size and directory path for ImageNet validation set
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results_dir = args.exp_res_dir_path

    # Handle embeddings

    # embeddings = np.array([np.load(file) for file in results_df['embeddings_path']])
    # tsne = TSNE(n_components=2, random_state=0)

    embeddings = np.load(os.path.join(results_dir, 'tot_embeddings_4D_np_array.npy'))
    df_embeddings = pd.read_csv(os.path.join(results_dir, 'val_embeddings_2d_predict_imagenet.csv'), index_col=False)
   
    # # assuming embeddings is a 4D numpy array
    # embeddings = embeddings.reshape(embeddings.shape[0], -1)
    
    # embeddings_2d = tsne.fit_transform(embeddings)

    # # Create a new DataFrame for embeddings
    # df_embeddings = pd.DataFrame(embeddings_2d, columns=['Dim1', 'Dim2'])
    # df_embeddings['true_class'] = results_df['true_class']
    # df_embeddings['predicted_class'] = results_df['top1_prediction_class']
    # df_embeddings['image_path'] = results_df['image_path']

    # # Save as CSV
    # df_embeddings.to_csv(os.path.join(results_dir, f"val_embeddings_2d_{exp_base_name}.csv"), index=False)

    # Plot with Plotly
    fig = px.scatter(df_embeddings, x='Dim1', y='Dim2', color='true_class', hover_data=['predicted_class', 'image_path'])
    # fig.update_traces(marker_size=1, selector=dict(type='scatter'))
    fig.update_traces(marker={'size': 1})

    # fig = px.scatter(df_embeddings, x='Dim1', y='Dim2', color='true_class', 
    #              hover_data=['predicted_class', 'image_path'], 
    #              marker={'size': 1})
    
    # fig.show()

    # Save plot as PNG
    fig.write_image(os.path.join(results_dir, f"tSNE_embeddings_scatter.png"))

    # Save plot as HTML
    pio.write_html(fig, os.path.join(results_dir, f"tSNE_embeddings_scatter.html"))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="ImageNet Validation Set Prediction")
    parser.add_argument('--exp_res_dir_path', default='/home/lamitay/experiments/predict_imagenet_resnet50_V1_14-07-2023_17-26-10/results/', type=str, help='exp_res_dir_path')

    args = parser.parse_args()

    main(args)