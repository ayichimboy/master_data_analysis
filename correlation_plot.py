# Import libraries

import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import pearsonr

"""

This function creates a correlation plot between the measured and predicted values of a dataset. 
The function takes in the following parameters:
measured: the measured values of the dataset
predicted: the predicted values of the dataset
title: the title of the plot
           
"""

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
    
    