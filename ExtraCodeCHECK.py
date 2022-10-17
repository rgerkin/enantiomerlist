import os
import string
import sys
import requests
import io

import numpy as np
import pandas as pd
import scipy as py
import matplotlib.pyplot as plt
import matplotlib as mpl  # Need this line
from sklearn.ensemble import RandomForestRegressor
from IPython.display import display
from scipy.stats import pearsonr
from scipy.stats import spearmanr
import seaborn as sns
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import LeaveOneOut
from sklearn.svm import SVR
from tqdm.auto import tqdm

def model(whole_dataset: pd.DataFrame, features_dataset: pd.DataFrame):
    """Generates a Random Forest Model for the dataframe of enantiomers and their features

    Args:
        whole_dataset (pd.DataFrame): The dataframe of enantiomers and their features
        features_dataset (pd.DataFrame): The dataframe with the features columns that do not have null values

    Returns:
        [Numpy Array]: returns the mean, standarad deviation, and histogram for the model
    """
    dataset = ""
    # Tells us which features dataset we used
    if whole_dataset.size < 400000:
        dataset = "Mordred Features"
    elif whole_dataset.size < 2000000:
        dataset = "Morgan Features"
    else:
        dataset = "Both Morgan and Mordred Features"

    # The Y variable holds all the correct classification values
    # The X variable holds all the data that will be used learn the classification problem
    Y = whole_dataset["log_abs"].values
    X = whole_dataset[features_dataset].astype(float).values  
    # This will help create a random split for training and testing data
    rs = np.zeros(100)
    ss = ShuffleSplit(n_splits=len(rs), random_state=0)

    counting_carvone = 0
    counting_glutamate = 0
    for i, (train, test) in enumerate(ss.split(X)):
        rfr = RandomForestRegressor(n_estimators=100, max_features=25)
        rfr.fit(X[train, :], Y[train])
        predicted = rfr.predict(X[test, :])
        rs[i] = np.corrcoef(predicted, Y[test])[0, 1]
        counting_carvone += predicted[1]
        counting_glutamate  += predicted[9]
    print(counting_carvone/100)
    print(counting_glutamate/100)
    print("The mean is ", np.mean(rs), "The Standard Error is ", np.std(rs)/np.sqrt(len(rs)))
    plt.hist(rs, alpha=0.5, label=dataset)
    plt.title("Correlation of Predicted Odor Divergence")
    plt.xlabel("Correlational Value (r) ")
    plt.ylabel("Number of Enantiomeric Pairs")
    plt.legend()
    return rs

def model_average(whole_dataset1: pd.DataFrame, features_dataset1: pd.DataFrame, whole_dataset2: pd.DataFrame, features_dataset2: pd.DataFrame):
    """Generates a Random Forest Model on the average of the predicted values of the mordred and morgan dataframes

    Args:
        whole_dataset1 (pd.DataFrame): The dataframe with the enantiomer and their mordred(or mogran) features
        features_dataset1 (pd.DataFrame): The dataframe with the features (mordred or morgan) columns that do not have null values
        whole_dataset2 (pd.DataFrame): The dataframe with the enantiomer and their morgan(or mordred) features
        features_dataset2 (pd.DataFrame): The dataframe with the features (mordred or morgan) columns that do not have null values

    Returns:
        [Numpy Array]: returns the mean, standarad deviation, and histogram for the model
    """
    # The Y variable holds all the correct classification values
    # The X variable holds all the data that will be used learn the classification problem
    Y = whole_dataset1["log_abs"].values
    X = whole_dataset1[features_dataset1].astype(float).values

    y = whole_dataset2["log_abs"].values
    x = whole_dataset2[features_dataset2].astype(float).values
    # This will help create a random split for training and testing data
    rs = np.zeros(100)
    ss = ShuffleSplit(n_splits=len(rs), random_state=0)

    for i, (train, test) in enumerate(ss.split(X)):
        rfr1 = RandomForestRegressor(n_estimators=100, max_features=25)
        rfr1.fit(X[train, :], Y[train])
        rfr2 = RandomForestRegressor(n_estimators=100, max_features=25)
        rfr2.fit(x[train, :], y[train])
        predicted = rfr1.predict(X[test, :])
        predicted2 = rfr2.predict(x[test, :])
        averaged = (predicted+predicted2)/2
        rs[i] = np.corrcoef(averaged, Y[test])[0, 1]
    print("The mean is ", np.mean(rs), " The Standard Error is ", py.stats.sem(rs))
    plt.hist(rs, alpha=0.5, label="The average of Mordred and Morgan")
    plt.title("Correlation of Predicted Odor Divergence")
    plt.xlabel("Correlational Value (r) ")
    plt.ylabel("Number of Enantiomeric Pairs")
    plt.legend()
    return rs 


def cumulativeHistogram(predicted_array: np.array, color: str, label_val: str):
    """Creates a cumulative histogram of all of the results

    Args:
        predicted_array (np.array): The predicted results in a numpy array
        color (str): Color of the line
        label_val (str): Tells use which test results each line is \
    
    return: No return because it will show the plots as the function is called
    """
    values, base = np.histogram(predicted_array)
    cumulative = np.cumsum(values)
    plt.plot(base[:-1], cumulative, c=color, label=label_val)
    plt.title("Correlation of Predicted Odor Divergence")
    plt.xlabel("Correlational Value (r) ")
    plt.ylabel("Number of Enantiomeric Pairs")
    plt.legend()
    return

def leave_one_out(whole_dataset: pd.DataFrame, features_dataset: pd.DataFrame):
    """Generates a Random Forest Model for the dataframe of enantiomers and their features

    Args:
        whole_dataset (pd.DataFrame): The dataframe of enantiomers and their features
        features_dataset (pd.DataFrame): The dataframe with the features columns that do not have null values

    Returns:
        [Numpy Array]: returns the mean, standarad deviation, and histogram for the model
    """
    dataset = ""
    # Tells us which features dataset we used
    if whole_dataset.size < 400000:
        dataset = "Mordred Features"
    elif whole_dataset.size < 2000000:
        dataset = "Morgan Features"
    else:
        dataset = "Both Morgan and Mordred Features"

    # The Y variable holds all the correct classification values
    # The X variable holds all the data that will be used learn the classification problem
    Y = whole_dataset["log_abs"].values
    X = whole_dataset[features_dataset].astype(float).values  
    # This will help create a random split for training and testing data
    predictedValues = np.zeros(X.shape[0])
    loo = LeaveOneOut()

    for i, (train, test) in enumerate(tqdm(loo.split(X))):
        rfr = RandomForestRegressor(n_estimators=100, max_features=25)
        rfr.fit(X[train, :], Y[train])
        predictedValues[test[0]] = rfr.predict(X[test, :])
    correlationCoefficient = np.corrcoef(predictedValues, Y)[0,1]
    plt.hist(correlationCoefficient, alpha=0.5, label=dataset)
    # plt.title("Correlation of Predicted Odor Divergence")
    # plt.xlabel("Correlational Value (r) ")
    # plt.ylabel("Number of Enantiomeric Pairs")
    plt.legend()
    return correlationCoefficient, predictedValues
