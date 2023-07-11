import pandas as pd
import os

full_predictions_path = '/home/lamitay/experiments/predict_imagenet_resnet50_11-07-2023_17-20-53/results/full_val_predictions_predict_imagenet_resnet50.csv'
summary_confidence_path = '/home/lamitay/experiments/predict_imagenet_resnet50_11-07-2023_17-20-53/results/val_confidence_summary_predict_imagenet_resnet50.csv'

pred_df = pd.read_csv(full_predictions_path, index_col=False)
conf_df = pd.read_csv(summary_confidence_path, index_col=False)

pred_df.head()
conf_df.head()
