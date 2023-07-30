import matplotlib.pyplot as plt
import pandas as pd
import argparse
import os


def main(args):
    
    results_dir = os.path.join(args.exp_path, 'results')
    results_df_path = os.path.join(results_dir, 'bootstrap_results.csv')
    
    # Get the results df
    results = pd.read_csv(results_df_path, index_col=False)
    
    # Calculate the differences for all metrics
    difference_metrics = results.copy()

    for metric in ['accuracy', 'auroc', 'f1', 'precision', 'recall', 'avg_confidence', 'entropy', 'first_moment', 'second_moment', 'third_moment', 'fourth_moment']:
        difference_metrics[f'{metric}_difference'] = difference_metrics[f'id_{metric}'] - difference_metrics[f'ood_{metric}']

    # Select only the difference metrics
    difference_metrics = difference_metrics[[col for col in difference_metrics.columns if 'difference' in col]]

    # Plot histograms for each difference metric
    fig, axs = plt.subplots(3, 4, figsize=(10, len(difference_metrics.columns)*5))

    for ax, column in zip(axs.flatten(), difference_metrics.columns):
        ax.hist(difference_metrics[column], bins=30, alpha=0.5, label=column)
        ax.set_xlabel('Difference')
        ax.set_ylabel('Frequency')
        ax.set_title(f'Histogram of {column}')
        ax.legend()
        plt.savefig(os.path.join(results_dir, f'{column}_histogram.png'))
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f'all_histograms.png'))

    # # Create a DataFrame for differences
    # results_diff = pd.DataFrame()

    # # Calculate differences for each measure
    # for measure in ['accuracy', 'auroc', 'f1', 'precision', 'recall', 'avg_confidence', 'entropy', 'first_moment', 'second_moment', 'third_moment', 'fourth_moment']:
    #     results_diff[measure] = results[f'id_{measure}'] - results[f'ood_{measure}']

    # # Calculate the mean and standard deviation of each difference
    # mean_std_diff = results_diff.agg(['mean', 'std']).transpose()

    # Calculate the mean and standard deviation of each difference
    mean_std_diff = difference_metrics.agg(['mean', 'std']).transpose()
    

    mean_std_diff.to_csv(os.path.join(results_dir, 'diff_summary_mean_std.csv'), index=False)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_path', type=str, default='/home/lamitay/experiments/tabular_bootstrap_29-07-2023_17-04-07', help='Experiment name')
    # parser.add_argument('--dataset_path', type=str, default='/home/lamitay/datasets/UCI_adult_income_dataset/adult.data', help='Tabular data path')
    # parser.add_argument('--bootstrap_num', type=int, default=5, help='Number of bootstrap iterations')
    # parser.add_argument('--num_epochs', type=int, default=10, help='Number of bootstrap iterations')
    args = parser.parse_args()
    main(args)