# Import libraries

import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split

# Fucntion 1: plot data points
def correlation_plot (data, measured, predicted, title):
    
    m, n = np.polyfit(measured, predicted, 1)
    
    # compute the correlation coefficient
    r_value, _ = pearsonr(measured, predicted)
    
    xlabel = measured.name if hasattr(measured, 'name') else 'Measured'
    ylabel = predicted.name if hasattr(predicted, 'name') else 'Predicted'
    
    plt.figure(figsize=(6,6))
    sns.set_style(style='white')
    
    sns.scatterplot(x=measured, 
                    y=predicted, 
                    data=data, s=50, color='blue', 
                    alpha=0.7, label='Data Points')
    
    plt.plot(measured, m*measured + n, color='red', linewidth=2, label="Regression Line")
    
    plt.xlabel(xlabel, fontsize=20)
    plt.ylabel(ylabel, fontsize=20)
    
    if title is not None:
        plt.title(title, fontsize=20)
    
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)

    
    plt.legend(fontsize=15, loc='upper left', frameon=True, edgecolor='black')
    
    plt.text(min(measured) + 0.1, 
             max(predicted) - 1000, f"r = {r_value:.2f}", 
             fontsize=15, color="black", 
             bbox=dict(facecolor='white', 
                       alpha=0.8))
    
    plt.show()
    

# Function 2: Standard Normal Variate (SNV) transformation

def snv_transform(X):
    """
    Perform Standard Normal Variate (SNV) transformation on the input data.
    
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        The input data.
        
    Returns
    -------
    X_snv : array-like of shape (n_samples, n_features)
        The SNV transformed data.
    """
    # Calculate the mean of each sample
    X_mean = np.mean(X, axis=1, keepdims=True)
    
    # Calculate the standard deviation of each sample
    X_std = np.std(X, axis=1, keepdims= True)
    
    # Perform SNV transformation
    X_snv = (X - X_mean) / X_std
    
    return X_snv


# Step 3: Split the data into training and testing sets
def split_data(X, y, test_size=0.2, random_state=None):
    """
    Split the input data into training and testing sets.
    
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        The input data.
        
    y : array-like of shape (n_samples,)
        The target values.
        
    test_size : float, default=0.2
        The proportion of the dataset to include in the test split.
        
    random_state : int, default=None
        Controls the shuffling applied to the data before applying the split.
        
    Returns
    -------
    X_train : array-like of shape (n_train_samples, n_features)
        The training input data.
        
    X_test : array-like of shape (n_test_samples, n_features)
        The testing input data.
        
    y_train : array-like of shape (n_train_samples,)
        The training target values.
        
    y_test : array-like of shape (n_test_samples,)
        The testing target values.
    """
    #from sklearn.model_selection import train_test_split
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    return X_train, X_test, y_train, y_test 