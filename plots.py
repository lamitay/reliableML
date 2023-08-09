import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import scipy

def plot_all_regressions():
    df = pd.read_csv('all_experiments_including_imagenet_c.csv')
    datasets = df['Dataset'].unique()  # list of dataset names
    
    # colors for the scatter plot. each dataset gets a different color
    colors = ['black', 'grey', 'gold', 'brown', 'red', 'blue', 'navy', 'orange', 'rosybrown', 'salmon', 'yellow', 'pink', 'peru', 'green', 'pink', 'lightgreen', 'olive', 'coral', 'khaki', 'sienna', 'azure', 'wheat', 'forestgreen', 'seagreen', 'cyan', 'teal', 'honeydew', 'tan', 'chocolate', 'aquamarine', 'lime', 'black', 'grey', 'gold', 'brown', 'red', 'blue', 'navy', 'orange', 'rosybrown', 'salmon', 'yellow', 'pink', 'peru', 'green', 'pink', 'lightgreen', 'olive', 'coral', 'khaki', 'sienna', 'azure', 'wheat', 'forestgreen', 'seagreen', 'cyan', 'teal', 'honeydew', 'tan', 'chocolate', 'aquamarine', 'lime']
    
    font = {'family' : 'sans-serif',
            'weight' : 'bold',
            'size'   : 20}
    
    matplotlib.rc('font', **font)
    
    for column in df.columns[3:]:  # skipping the first three columns where there aren't any response vectors
        X = df[column].values.reshape(-1, 1)
        y = df['Accuracy'].values.reshape(-1, 1)
        nan_mask = np.isnan(X)  # used for the case of the kurtosis column where there are some rows without values
        if np.sum(nan_mask > 0):
            X = X[~nan_mask].reshape(-1, 1)
            y = y[~nan_mask].reshape(-1, 1)
        reg = LinearRegression().fit(X, y)
        y_pred = reg.predict(X)
        
        R2_score = r2_score(y, y_pred)
        # print('R2_score: ' + str(R2_score))
        pearson_correlation_coeff = scipy.stats.pearsonr(X.flatten(), y.flatten()).statistic
        # print('Pearson correlation coefficient: ' + str(pearson_correlation_coeff))
        
        plt.plot(X, y_pred, color='blue', linewidth=3)
        
        for i, dataset in enumerate(datasets):
            X = df[df['Dataset'] == dataset][column].values.reshape(-1, 1)
            y = df[df['Dataset'] == dataset]['Accuracy'].values.reshape(-1, 1)
            plt.scatter(X, y, color=colors[i], label=dataset)
        
        plt.title('Linear Regression Model \n' + 'Pearson correlation coefficient: \n' + str(R2_score))
        plt.xlabel(column)
        plt.ylabel('Accuracy')
        
        plt.show()


def plot_graphs_of_differences():
    df = pd.read_csv('all_experiments_including_imagenet_c_differences.csv')
    df = df[df['Dataset'] != 'imagenet-a']
    df = df[df['Dataset'] != 'imagenet-r']
    datasets = df['Dataset'].unique()  # list of dataset names
    
    font = {'family' : 'sans-serif',
            'weight' : 'bold',
            'size'   : 20}
    
    # colors for the scatter plot. each dataset gets a different color
    colors = ['black', 'grey', 'gold', 'brown', 'red', 'blue', 'navy', 'orange', 'rosybrown', 'salmon', 'yellow', 'pink', 'peru', 'green', 'pink', 'lightgreen', 'olive', 'coral', 'khaki', 'sienna', 'azure', 'wheat', 'forestgreen', 'seagreen', 'cyan', 'teal', 'honeydew', 'tan', 'chocolate', 'aquamarine', 'lime', 'black', 'grey', 'gold', 'brown', 'red', 'blue', 'navy', 'orange', 'rosybrown', 'salmon', 'yellow', 'pink', 'peru', 'green', 'pink', 'lightgreen', 'olive', 'coral', 'khaki', 'sienna', 'azure', 'wheat', 'forestgreen', 'seagreen', 'cyan', 'teal', 'honeydew', 'tan', 'chocolate', 'aquamarine', 'lime']
    
    for column in df.columns[3:]:  # skipping the first three columns where there aren't any response vectors
        X = df[column].values.reshape(-1, 1)
        y = df['Accuracy'].values.reshape(-1, 1)
        nan_mask = np.isnan(X)  # used for the case of the kurtosis column where there are some rows without values
        if np.sum(nan_mask > 0):
            X = X[~nan_mask].reshape(-1, 1)
            y = y[~nan_mask].reshape(-1, 1)
        reg = LinearRegression().fit(X, y)
        y_pred = reg.predict(X)
        
        R2_score = r2_score(y, y_pred)
        # print('R2_score: ' + str(R2_score))
        pearson_correlation_coeff = scipy.stats.pearsonr(X.flatten(), y.flatten()).statistic
        # print('Pearson correlation coefficient: ' + str(pearson_correlation_coeff))
        
        plt.plot(X, y_pred, color='blue', linewidth=3)
        
        for i, dataset in enumerate(datasets):
            X = df[df['Dataset'] == dataset][column].values.reshape(-1, 1)
            y = df[df['Dataset'] == dataset]['Accuracy'].values.reshape(-1, 1)
            plt.scatter(X, y, color=colors[i], label=dataset)
        
        plt.title('Pearson correlation coefficient: \n' + str(pearson_correlation_coeff))
        plt.xlabel('Difference of ' + column)
        plt.ylabel('Accuracy Gap')
        # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        
        plt.show()