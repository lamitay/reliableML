import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score
from scipy.stats import entropy, moment
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from tqdm import tqdm
import os
import argparse
import matplotlib.cm as cm


def main(args):
    
    """
    The main function for the script. It performs the following steps:
    
    1. Sets up the experiment directory based on the current time and provided base directory.
    2. Loads the dataset from the provided path and preprocesses it (drops missing values, applies one-hot encoding to categorical variables, normalizes numerical variables, and label encodes the target variable).
    3. Splits the data into in-distribution (ID) and out-of-distribution (OOD) sets based on the 'native-country' feature.
    4. Initializes a RandomForestClassifier and a DataFrame to store the results.
    5. Performs 100 bootstrap iterations, where for each iteration it resamples the training data, fits the classifier to the resampled data, makes predictions on the ID and OOD test sets, calculates various metrics (accuracy, AUROC, F1 score, precision, recall, average confidence, entropy, and moments), and adds these metrics to the results DataFrame.
    6. Calculates the difference of confidences (DoC), difference of entropies (DoE), difference of accuracies (DoAcc), and differences of moments for each bootstrap iteration and adds these to the results DataFrame.
    7. Saves the results to a CSV file.
    8. Calculates the differences for all metrics and saves these as a separate CSV file.
    9. Plots histograms and line plots for each difference metric and saves these plots as PNG files.
    10. Plots the DoC, DoE, DoAcc, and differences of moments across bootstrap iterations and saves this plot as a PNG file.
    11. Plots histograms of DoC and DoE values and saves these plots as PNG files.
    12. Plots the mean and standard deviation of the difference metrics and saves these plots as PNG files.

    Parameters:
    args (argparse.Namespace): The command-line arguments. It should have the following attributes:
        - base_exp_dir (str): The base directory for the experiment.
        - exp_name (str): The name of the experiment.
        - dataset_path (str): The path to the dataset.
        - bootstrap_num (int): The number of bootstrap iterations to perform.
    """

    base_exp_dir = args.base_exp_dir
    exp_name = args.exp_name
    data_path = args.dataset_path

    # Experiment name
    curr_time = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    exp_name = exp_name + '_' + curr_time
    exp_dir = os.path.join(base_exp_dir, exp_name)
    results_dir = os.path.join(exp_dir, 'results')
    
    # Create experiment directory
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # Load the data
    df = pd.read_csv(data_path, header=None)
    df.columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']

    print('Loaded data.')

    # Drop missing values
    df.replace(' ?', np.nan, inplace=True)
    df.dropna(inplace=True)

    # Apply one-hot encoding to categorical variables
    df = pd.get_dummies(df, columns=['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex'])

    # Normalize numerical variables
    scaler = StandardScaler()
    df[['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']] = scaler.fit_transform(df[['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']])

    # Label encode the target variable
    le = LabelEncoder()
    df['income'] = le.fit_transform(df['income'])

    # Split the data into in-distribution (ID) and out-of-distribution (OOD)
    id_data = df[df['native-country'] == ' United-States']
    ood_data = df[df['native-country'] != ' United-States']
    ood_data = ood_data.drop(columns=['native-country'])

    # Split the ID data into train and test sets
    train_data, id_test_data = train_test_split(id_data, test_size=len(ood_data), stratify=id_data['income'])

    # Initialize the classifier
    clf = RandomForestClassifier()

    # Initialize a DataFrame to store the results
    results = pd.DataFrame(columns=['iteration', 'id_accuracy', 'id_auroc', 'id_f1', 'id_precision', 'id_recall', 'id_avg_confidence', 'id_entropy', 'id_first_moment', 'id_second_moment', 'id_third_moment', 'id_fourth_moment', 'ood_accuracy', 'ood_auroc', 'ood_f1', 'ood_precision', 'ood_recall', 'ood_avg_confidence', 'ood_entropy', 'ood_first_moment', 'ood_second_moment', 'ood_third_moment', 'ood_fourth_moment'])

    print('start bootstrap now')

    # Perform 100 bootstrap iterations
    for i in tqdm(range(100)):
        # Resample the training data
        bootstrap_train_data = resample(train_data)

        # Separate the features and target
        X_train = bootstrap_train_data.drop(columns=['income', 'native-country'])
        y_train = bootstrap_train_data['income']
        X_id_test = id_test_data.drop(columns=['income', 'native-country'])
        y_id_test = id_test_data['income']
        X_ood_test = ood_data.drop(columns=['income'])
        y_ood_test = ood_data['income']

        # Fit the model to the bootstrap sample and make predictions on the ID and OOD test sets
        clf.fit(X_train, y_train)
        id_test_probs = clf.predict_proba(X_id_test)[:, 1]
        id_test_preds = clf.predict(X_id_test)
        ood_test_probs = clf.predict_proba(X_ood_test)[:, 1]
        ood_test_preds = clf.predict(X_ood_test)

        # Calculate metrics and add them to the results DataFrame
        id_accuracy = accuracy_score(y_id_test, id_test_preds)
        id_auroc = roc_auc_score(y_id_test, id_test_probs)
        id_f1 = f1_score(y_id_test, id_test_preds)
        id_precision = precision_score(y_id_test, id_test_preds)
        id_recall = recall_score(y_id_test, id_test_preds)
        id_avg_confidence = np.mean(id_test_probs)
        id_entropy = entropy(id_test_probs)
        id_first_moment = moment(id_test_probs, moment=1)
        id_second_moment = moment(id_test_probs, moment=2)
        id_third_moment = moment(id_test_probs, moment=3)
        id_fourth_moment = moment(id_test_probs, moment=4)

        ood_accuracy = accuracy_score(y_ood_test, ood_test_preds)
        ood_auroc = roc_auc_score(y_ood_test, ood_test_probs)
        ood_f1 = f1_score(y_ood_test, ood_test_preds)
        ood_precision = precision_score(y_ood_test, ood_test_preds)
        ood_recall = recall_score(y_ood_test, ood_test_preds)
        ood_avg_confidence = np.mean(ood_test_probs)
        ood_entropy = entropy(ood_test_probs)
        ood_first_moment = moment(ood_test_probs, moment=1)
        ood_second_moment = moment(ood_test_probs, moment=2)
        ood_third_moment = moment(ood_test_probs, moment=3)
        ood_fourth_moment = moment(ood_test_probs, moment=4)

        results.loc[i] = [i, id_accuracy, id_auroc, id_f1, id_precision, id_recall, id_avg_confidence, id_entropy, id_first_moment, id_second_moment, id_third_moment, id_fourth_moment,
                        ood_accuracy, ood_auroc, ood_f1, ood_precision, ood_recall, ood_avg_confidence, ood_entropy, ood_first_moment, ood_second_moment, ood_third_moment, ood_fourth_moment]

    # Calculate the difference of confidences (DoC) for each bootstrap iteration
    results['doc'] = results['id_avg_confidence'] - results['ood_avg_confidence']
    results['doe'] = results['id_entropy'] - results['ood_entropy']
    results['doacc'] = results['id_accuracy'] - results['ood_accuracy']
    results['do2m'] = results['id_second_moment'] - results['ood_second_moment']
    results['do3m'] = results['id_third_moment'] - results['ood_third_moment']
    results['do4m'] = results['id_fourth_moment'] - results['ood_fourth_moment']

    # Save the results to a CSV file
    results.to_csv(os.path.join(results_dir, 'bootstrap_results.csv'), index=False)

    # Calculate the differences for all metrics
    difference_metrics = results.copy()

    for metric in ['accuracy', 'auroc', 'f1', 'precision', 'recall', 'avg_confidence', 'entropy', 'first_moment', 'second_moment', 'third_moment', 'fourth_moment']:
        difference_metrics[f'{metric}_difference'] = difference_metrics[f'id_{metric}'] - difference_metrics[f'ood_{metric}']

    # Select only the difference metrics
    difference_metrics = difference_metrics[[col for col in difference_metrics.columns if 'difference' in col]]

    # Add the 'iteration' column as the first column in 'difference_metrics'
    difference_metrics.insert(0, 'iteration', results['iteration'])
    
    # Save the difference in results to a CSV file
    difference_metrics.to_csv(os.path.join(results_dir, 'bootstrap_differences_results.csv'), index=False)

    # Create a copy of difference_metrics and drop the 'iteration' column
    difference_metrics_copy = difference_metrics.copy()
    difference_metrics_copy.drop(columns=['iteration', 'first_moment_difference'])

    # Calculate the mean, standard deviation, min, 25th percentile, median, 75th percentile, and max of each difference
    stats_diff = difference_metrics_copy.describe()

    # Save these as a df table
    stats_diff.to_csv(os.path.join(results_dir, 'diff_summary_stats.csv'))


    # General plots configurations
    colors = cm.rainbow(np.linspace(0, 1, len(difference_metrics.columns)))
    # Create a list of the columns to plot (excluding 'first_moment' and 'iteration')
    columns_to_plot = [col for col in difference_metrics.columns if col not in ['first_moment_difference', 'iteration']]

    # Plot histograms for each difference metric
    fig, axs = plt.subplots(3, 4, figsize=(14, 7))
    axs = axs.flatten()  # Flatten the axes array
    for i in range(len(columns_to_plot)):
        # Skip unnecessary sub-plots
        if i >= len(columns_to_plot):
            ax.axis('off')  # Hide unnecessary sub-plots
            continue
        column = columns_to_plot[i]
        ax = axs[i]
        color = colors[i]
        # ax.hist(difference_metrics[column], bins=30, alpha=0.5, label=column, color=color)
        ax.hist(difference_metrics[column], bins=30, alpha=0.5, color=color)
        ax.set_xlabel('Difference')
        ax.set_ylabel('Frequency')
        ax.set_title(f'Histogram of {column}')
        # ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f'all_histograms.png'))

    # Plot line plots for each difference metric
    fig, axs = plt.subplots(3, 4, figsize=(14, 7))
    axs = axs.flatten()  # Flatten the axes array
    for i in range(len(columns_to_plot)):
        # Skip unnecessary sub-plots
        if i >= len(columns_to_plot):
            ax.axis('off')  # Hide unnecessary sub-plots
            continue
        column = columns_to_plot[i]
        ax = axs[i]
        color = colors[i]
        # ax.plot(difference_metrics[column], color=color, label=column)
        ax.plot(difference_metrics[column], color=color)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Difference')
        ax.set_title(f'Line plot of {column}')
        # ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f'all_line_plots.png'))

    # Plot all difference metrics in a single plot
    plt.figure(figsize=(14, 7))
    for i in range(len(columns_to_plot)):
        column = columns_to_plot[i]
        color = colors[i]
        plt.plot(difference_metrics['iteration'], difference_metrics[column], color=color, label=column)
    plt.xlabel('Iteration')
    plt.ylabel('Difference')
    plt.title('Line plot of all difference metrics')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f'all_metrics_single_plot.png'))

    # Plot the difference of confidences (DoC) across bootstrap iterations
    plt.figure(figsize=(14, 7))
    plt.plot(results['iteration'], results['doc'], label='DoC')
    plt.plot(results['iteration'], results['doe'], label='DoE')
    plt.plot(results['iteration'], results['doacc'], label='DoAcc')
    plt.plot(results['iteration'], results['do2m'], label='Do2ndMom')
    plt.plot(results['iteration'], results['do3m'], label='Do3rdMom')
    plt.plot(results['iteration'], results['do4m'], label='Do4thMom')
    plt.xlabel('Bootstrap Iteration')
    plt.ylabel('Difference')
    plt.title('Difference of Accuracies, Confidences, Entropies & Moments Across Bootstrap Iterations')
    plt.legend()
    plt.savefig(os.path.join(results_dir, 'Interesting_differences_per_bootstrap_iter.png'))

    # Plot the histogram of DoC values
    plt.figure(figsize=(14, 7))
    plt.hist(results['doc'], bins=30, alpha=0.5, color = colors[1], label='DoC')
    plt.xlabel('Difference')
    plt.ylabel('Frequency')
    plt.title('Histogram of Differences of Confidences')
    plt.legend()
    # plt.show()
    plt.savefig(os.path.join(results_dir, 'DoC_histogram.png'))

    # Plot the histogram of DoE values
    plt.figure(figsize=(14, 7))
    plt.hist(results['doe'], bins=30, alpha=0.5, color = colors[4], label='Doe')
    plt.xlabel('Difference')
    plt.ylabel('Frequency')
    plt.title('Histogram of Differences of Entropies')
    plt.legend()
    # plt.show()
    plt.savefig(os.path.join(results_dir, 'DoE_histogram.png'))


    # Plot mean and STD of stats_diff
    results = stats_diff.copy()
    
    # Drop the unnecessary columns
    results = results.drop(columns=['iteration', 'first_moment_difference'])
    results = results.reset_index().rename(columns={'index': 'metrics'})

    # Filter the results to get only the mean coloumns, prepare for plotting
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
    ax.bar(diff_stats_names, mean_results[1].values, color=colors, width=0.8)
    plt.title('Mean of metrics differences')  # Set the title
    # plt.xlabel('Metrics')  # Set x-axis label
    plt.xticks(rotation='vertical', fontsize=14)  # Make x-axis labels vertical
    plt.ylabel('Values')  # Set y-axis label
    # plt.grid(True)  # Remove grid
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'mean_bar_plot.png'))


    # Filter the results to get only the STD coloumns, prepare for plotting
    std_results = results[results['metrics'] == 'std']  # Filter to only include 'mean' row
    std_results = std_results.drop(['metrics'], axis=1)  # Remove specified columns
    # Transpose the DataFrame so that the column names become the index
    std_results = std_results.transpose()

    # Plot the bar chart
    fig, ax = plt.subplots(figsize=(14, 10))
    #mean_results.plot(kind='bar', legend=True, color=plt.cm.gist_rainbow(np.linspace(0, 1, len(mean_results))))
    ax.bar(diff_stats_names, std_results[2].values, color=colors, width=0.7)
    plt.title('Standard deviation of metrics differences', fontsize=14)  # Set the title
    # plt.xlabel('Metrics')  # Set x-axis label
    plt.xticks(rotation='vertical', fontsize=14)  # Make x-axis labels vertical
    plt.ylabel('Values', fontsize=14)  # Set y-axis label
    # plt.grid(False)  # Remove grid
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'standard_deviation_bar_plot.png'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_exp_dir', type=str, default='/home/lamitay/experiments', help='Output directory')
    parser.add_argument('--exp_name', type=str, default='UCI_adult_income_bootstrap', help='Experiment name')
    parser.add_argument('--dataset_path', type=str, default='/home/lamitay/datasets/UCI_adult_income_dataset/adult.data', help='Tabular data path')
    parser.add_argument('--bootstrap_num', type=int, default=5, help='Number of bootstrap iterations')
    args = parser.parse_args()
    main(args)
