# Bounnoy Phanthavong (ID: 973081923)
# Homework 5
#
# This is a machine learning program that uses Gaussian's Naive Bayes
# and Logistic Regression to classify spam.
#
# This program was built in Python 3.

from pathlib import Path
import math
import csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score
import numpy as np
import random

# Function to compute probability of a specified label in a Numpy array.
def prob(data, label):
    unique, count = np.unique(data, return_counts = True)
    A = dict(zip(unique, count))
    total = 0.0
    for value in A.values():
        total += value

    return A[label] / total

# Function to compute the mean and standard deviation of a specified label in a Numpy array.
def computeMS(dataX, dataY):
    data = np.concatenate((dataX, np.vstack(dataY)), axis = 1)  # Combine for sorting.
    data = data[data[:,(len(data[0])-1)].argsort()]             # Sort by class.

    unique, count = np.unique(dataY, return_counts = True)
    features = len(dataX[0])   # Number of features.
    classes = len(unique)   # Number of classes.

    # List of mean, standard deviation, and class.
    ms = np.zeros((classes, features, 3))
    epsilon = 0.0001

    # Loop through each class in the training data.
    for i in range(classes):

        # Loop through each attribute.
        for x in range(features):

            # Offset from a to b. Used for slicing magic.
            a = 0 if (i - 1 < 0) else np.sum(count[:i])
            b = a + count[i]

            # Calculate our mean and standard deviation.
            mean = np.sum(data[a:b,x], dtype=float) / count[i]
            std = math.sqrt(np.sum( (data[a:b,x] - mean)**2, dtype=float ) / count[i])
            std += epsilon

            # Add the mean, standard deviation, and class to msList.
            ms[i][x][0] = mean
            ms[i][x][1] = std
            ms[i][x][2] = unique[i]

            print("Class", '%d' % unique[i] + ", attribute", '%d' % (x+1) + ", mean =", '%.2f' % mean + ", std =", '%.2f' % std)
    return ms

# This class computes the gaussian for each test data and returns the predicted class and probability.
def gaussian(row, target, ms, pc):
    pred = 0            # Highest prediction
    prob = 0.0          # Gaussian of predicted class.
    acc = 0.0           # Accuracy of our prediction.
    ties = 1            # Count of how many ties for the predicted class.
    correct = 0         # Flag that marks our prediction as correct or not.
    tgaus = 1.0e-250    # Total gaussian after calculating for all classes.

    # Loop through classes.
    for i in range(len(ms)):
        gaus = 0.0                    # Current gaussian.

        # Loop through attributes.
        for j in range(len(ms[0])):
            mean = ms[i][j][0]        # Extract the mean from our training list.
            std = ms[i][j][1]         # Extract the standard deviation from our training list.

            x = (row[j] - mean) / std                                   # Subcalculation for exponent.
            calc = math.exp(-x*x/2.0) / (math.sqrt(2.0*math.pi * std * std))  # Calculate full gaussian.
            gaus += math.log(max(calc, 1.0e-250))#0.00000000000000000000000000000000000000000000000000000000000000000001))# if calc > 0 else 0.0                # Log results to prevent underflow.

        gaus += math.log(pc[i]) # Add probability(class).
        gaus = math.exp(gaus)   # Raise the results back so they're non-negative.
        tgaus += gaus           # Add to our total gaussian counter.

        # Check if new gaussian is about the same as the highest predicted gaussian.
        if math.isclose(prob, gaus):
            ties += 1
            pred = random.choice([pred, ms[i][0][2]]) # Randomly pick a prediction between the ties.

        else:
            prob = max(prob, gaus)
            # If the new gaussian is highest, make it the new predicted class.
            # Else, keep old prediction.
            pred = ms[i][0][2] if math.isclose(prob, gaus) else pred

        if int(target) == int(pred):
            correct = 1
        else:
            correct = 0

    # If no ties and correct, 1.
    # If no ties and incorrect, then 0.
    # If ties and correct, 1/n.
    # If ties and incorrect, 0
    if ties == 0 and correct > 0:
        acc = 1
    elif ties > 0 and correct > 0:
        acc = 1/ties
    else:
        acc = 0

    return pred, prob/tgaus, acc

def confuse(predict, actual, accuracy):
    unique, _ = np.unique(actual, return_counts = True)
    matrix = np.zeros((len(unique), len(unique)))

    for i in range(len(predict)):
        matrix[ int(predict[i]) ][ int(actual[i]) ] += 1

    np.set_printoptions(suppress = True)
    print("\nConfusion Matrix")
    print(matrix, "\n")

    with open('results.csv', 'a') as csvFile:
        w = csv.writer(csvFile)
        w.writerow([])
        w.writerow(["Confusion Matrix"])
        for k in range(len(unique)):
            w.writerow(matrix[k,:])
        w.writerow(["Final Accuracy"] + [accuracy])
        w.writerow([])

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
    pc = np.zeros(2)
    pc[0] = prob(YTrain, 0)
    pc[1] = prob(YTrain, 1)
    print("Spam probability in training data:", pc[1])
    print("Non-spam probability in training data:", pc[0])

    # Compute mean and standard deviation for all features in spam and non-spam.
    ms = computeMS(XTrain, YTrain)

    # Classify test data.
    accuracy = 0

    YPredict = np.array([])

    # Rows in Test Data
    for i in range(len(YTest)):
        pred, prob, acc = gaussian(XTest[i], YTest[i], ms, pc)
        truth = YTest[i]
        accuracy += acc
        print("ID=" + '%5d' % (i+1) + ", predicted=" + '%3d' % pred + ", probability =", '%.4f' % prob + ", true=" + '%3d' % truth + ", accuracy=" + '%4.2f' % acc)

        YPredict = np.append(YPredict, pred)

    print("classification accuracy=" + '%6.4f' % (accuracy/len(XTest)*100))

    precision = precision_score(YTest, YPredict)
    recall = recall_score(YTest, YPredict)

    print('precision: %f' % precision)
    print('recall: %f' % recall)

    confuse(YPredict, YTest, accuracy)
