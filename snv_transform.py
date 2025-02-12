import numpy as np

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