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


def main(args):
    base_exp_dir = args.base_exp_dir
    exp_name = args.exp_name
    data_path = args.dataset_path
    # num_epochs = args.num_epochs

    # Experiment name
    curr_time = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    exp_name = exp_name + '_' + curr_time
    exp_dir = os.path.join(base_exp_dir, exp_name)
    results_dir = os.path.join(exp_dir, 'results')
    # models_dir = os.path.join(exp_dir, 'models')
    
    # Create experiment directory
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    # os.makedirs(models_dir, exist_ok=True)

    # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

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

    # Save the results to a CSV file
    results.to_csv(os.path.join(results_dir, 'bootstrap_results.csv'), index=False)

    # Plot the difference of confidences (DoC) across bootstrap iterations
    plt.figure(figsize=(14, 7))
    plt.plot(results['iteration'], results['doc'], label='DoC')
    plt.plot(results['iteration'], results['doe'], label='DoE')
    plt.xlabel('Bootstrap Iteration')
    plt.ylabel('Difference')
    plt.title('Difference of Confidences and Entropies Across Bootstrap Iterations')
    plt.legend()
    # plt.show()
    plt.savefig(os.path.join(results_dir, 'DoC_DoE_per_bootstrap_iter.png'))


    # Plot the histogram of DoC values
    plt.figure(figsize=(14, 7))
    plt.hist(results['doc'], bins=30, alpha=0.5, label='DoC')
    plt.xlabel('Difference')
    plt.ylabel('Frequency')
    plt.title('Histogram of Differences of Confidences')
    plt.legend()
    # plt.show()
    plt.savefig(os.path.join(results_dir, 'DoC_histogram.png'))

    # Plot the histogram of DoE values
    plt.figure(figsize=(14, 7))
    plt.hist(results['doe'], bins=30, alpha=0.5, label='DoC')
    plt.xlabel('Difference')
    plt.ylabel('Frequency')
    plt.title('Histogram of Differences of Entropies')
    plt.legend()
    # plt.show()
    plt.savefig(os.path.join(results_dir, 'DoE_histogram.png'))

    # Plot the difference in accuracy across bootstrap iterations
    plt.figure(figsize=(14, 7))
    plt.plot(results['iteration'], results['id_accuracy'] - results['ood_accuracy'], label='Accuracy difference')
    plt.xlabel('Bootstrap Iteration')
    plt.ylabel('Difference')
    plt.title('Difference of Accuracy Across Bootstrap Iterations')
    plt.legend()
    # plt.show()
    plt.savefig(os.path.join(results_dir, 'difference_of_accuracy_per_bootstrap_iter.png'))


    # Plot the difference in AUROC across bootstrap iterations
    plt.figure(figsize=(14, 7))
    plt.plot(results['iteration'], results['id_auroc'] - results['ood_auroc'], label='AUROC difference')
    plt.xlabel('Bootstrap Iteration')
    plt.ylabel('Difference')
    plt.title('Difference of AUROC Across Bootstrap Iterations')
    plt.legend()
    # plt.show()
    plt.savefig(os.path.join(results_dir, 'difference_of_AUROC_per_bootstrap_iter.png'))


    # Plot the difference in F1 score across bootstrap iterations
    plt.figure(figsize=(14, 7))
    plt.plot(results['iteration'], results['id_f1'] - results['ood_f1'], label='F1 score difference')
    plt.xlabel('Bootstrap Iteration')
    plt.ylabel('Difference')
    plt.title('Difference of F1 Score Across Bootstrap Iterations')
    plt.legend()
    # plt.show()
    plt.savefig(os.path.join(results_dir, 'difference_of_f1_per_bootstrap_iter.png'))


    # Plot the difference in Precision across bootstrap iterations
    plt.figure(figsize=(14, 7))
    plt.plot(results['iteration'], results['id_precision'] - results['ood_precision'], label='Precision difference')
    plt.xlabel('Bootstrap Iteration')
    plt.ylabel('Difference')
    plt.title('Difference of Precision Across Bootstrap Iterations')
    plt.legend()
    # plt.show()
    plt.savefig(os.path.join(results_dir, 'difference_of_precision_per_bootstrap_iter.png'))


    # Plot the difference in Recall across bootstrap iterations
    plt.figure(figsize=(14, 7))
    plt.plot(results['iteration'], results['id_recall'] - results['ood_recall'], label='Recall difference')
    plt.xlabel('Bootstrap Iteration')
    plt.ylabel('Difference')
    plt.title('Difference of Recall Across Bootstrap Iterations')
    plt.legend()
    plt.show()
    plt.savefig(os.path.join(results_dir, 'difference_of_recall_per_bootstrap_iter.png'))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_exp_dir', type=str, default='/home/lamitay/experiments', help='Output directory')
    parser.add_argument('--exp_name', type=str, default='tabular_bootstrap', help='Experiment name')
    parser.add_argument('--dataset_path', type=str, default='/home/lamitay/datasets/UCI_adult_income_dataset/adult.data', help='Tabular data path')
    parser.add_argument('--bootstrap_num', type=int, default=5, help='Number of bootstrap iterations')
    # parser.add_argument('--num_epochs', type=int, default=10, help='Number of bootstrap iterations')
    args = parser.parse_args()
    main(args)
