import matplotlib.pyplot as plt
import pandas as pd
import argparse
import os
import matplotlib.cm as cm
import numpy as np
import seaborn as sns



def main(args):
    
    results_dir = os.path.join(args.exp_path, 'results')
    results_df_path = os.path.join(results_dir, 'diff_summary_stats.csv')
    
    # Get the results df
    results = pd.read_csv(results_df_path, index_col=False)
    
    # Drop the unnecessary columns
    results = results.drop(columns=['iteration', 'first_moment_difference'])
    results = results.rename(columns={'Unnamed: 0': 'metrics'})

    # Assuming results is your DataFrame
    mean_results = results[results['metrics'] == 'mean']  # Filter to only include 'mean' row
    mean_results = mean_results.drop(['metrics'], axis=1)  # Remove specified columns
    # diff_stats_cols = mean_results.columns.to_list() 
    diff_stats_names = ['Accuracy', 'AUROC', 'F1', 'precision', 'recall', 'DoC', 'DoE', '2nd_mom', '3rd_mom', '4th_mom']
    # Transpose the DataFrame so that the column names become the index
    mean_results = mean_results.transpose()

    colors = cm.rainbow(np.linspace(0, 1, len(diff_stats_names)))  # Generate a rainbow color palette

    # Plot the bar chart
    fig, ax = plt.subplots(figsize=(14, 10))
    #mean_results.plot(kind='bar', legend=True, color=plt.cm.gist_rainbow(np.linspace(0, 1, len(mean_results))))
    ax.bar(diff_stats_names, mean_results[1].values, color=colors, width=0.9)
    plt.title('Mean of metrics differences')  # Set the title
    # plt.xlabel('Metrics')  # Set x-axis label
    plt.xticks(rotation='vertical')  # Make x-axis labels vertical
    plt.ylabel('Values')  # Set y-axis label
    # plt.grid(False)  # Remove grid
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'mean_bar_plot.png'))


    std_results = results[results['metrics'] == 'std']  # Filter to only include 'mean' row
    std_results = std_results.drop(['metrics'], axis=1)  # Remove specified columns
    # Transpose the DataFrame so that the column names become the index
    std_results = std_results.transpose()

    # Plot the bar chart
    fig, ax = plt.subplots(figsize=(14, 10))
    #mean_results.plot(kind='bar', legend=True, color=plt.cm.gist_rainbow(np.linspace(0, 1, len(mean_results))))
    ax.bar(diff_stats_names, std_results[2].values, color=colors, width=0.9)
    plt.title('Standard deviation of metrics differences')  # Set the title
    # plt.xlabel('Metrics')  # Set x-axis label
    plt.xticks(rotation='vertical')  # Make x-axis labels vertical
    plt.ylabel('Values')  # Set y-axis label
    # plt.grid(False)  # Remove grid
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'standard_deviation_bar_plot.png'))
 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_path', type=str, default='/home/lamitay/experiments/UCI_adult_income_bootstrap_30-07-2023_14-49-05/', help='Experiment name')
    args = parser.parse_args()
    main(args)