# Bounnoy Phanthavong (ID: 973081923)
# Homework 5
#
# This is a machine learning program that uses Gaussian's Naive Bayes
# and Logistic Regression to classify spam.
#
# This program was built in Python 3.

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, roc_curve
import matplotlib.pyplot as plt
import numpy as np

# Function to compute probability of a specified label in a Numpy array.
def prob(data, label):
    unique, count = np.unique(data, return_counts = True)
    A = dict(zip(unique, count))
    total = 0.0
    for value in A.values():
        total += value

    return A[label] / total


if __name__ == '__main__':

    # Load data.
    fName = "spambase/spambase.data"
    fileTrain = Path(fName)

    if not fileTrain.exists():
        sys.exit(fName + " not found")

    trainData = np.genfromtxt(fName, delimiter=",")

    # Split input data into X and labels into Y.
    X = trainData[:,0:-1]
    Y = trainData[:,-1]

    # Split X and Y by 50% for training and testing.
    # Keep proportion of labels same across both sets.
    XTrain, XTest, YTrain, YTest = train_test_split(X, Y, test_size = 0.5, stratify = Y)

    # Compute probability of spam and non-spam.
    print("Spam probability:", prob(YTrain, 1))
    print("Non-spam probability:", prob(YTrain, 0))
