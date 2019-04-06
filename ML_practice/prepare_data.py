from sklearn.linear_model import LogisticRegression
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

def iris_datasets():
    # Load the dataset iris
    iris = datasets.load_iris()
    X = iris.data[:, [2, 3]]
    y = iris.target
    
    # shuffle data
    indices = np.arange(X.shape[0])
    np.random.seed(0) 
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]
    
    # split data into train and test set
    nb_validation_samples = int(0.3 * X.shape[0])
    X_train = X[nb_validation_samples:]
    X_test = X[0:nb_validation_samples]
    
    y_train = y[nb_validation_samples:]
    y_test = y[0:nb_validation_samples]
    
    #from sklearn.cross_validation import train_test_split
    ## X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    
    return X_train, X_test, y_train, y_test


def standardize(X_train, X_test):
    # standardization
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)

    return X_train_std, X_test_std

