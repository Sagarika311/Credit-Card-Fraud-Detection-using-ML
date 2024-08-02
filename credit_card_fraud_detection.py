# Importing all the necessary Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import gridspec
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score, matthews_corrcoef
from sklearn.metrics import confusion_matrix

# Loading the Data
# Load the dataset from the csv file using pandas
data = pd.read_csv("credit.csv")

# Understanding the Data
# Grab a peek at the data
print(data.head())

# Describing the Data
# Print the shape of the data
print(data.shape)
print(data.describe())

# Imbalance in the data
# Determine number of fraud cases in dataset
fraud = data[data['Class'] == 1]
valid = data[data['Class'] == 0]
outlierFraction = len(fraud) / float(len(valid))
print(outlierFraction)
print('Fraud Cases: {}'.format(len(fraud)))
print('Valid Transactions: {}'.format(len(valid)))

# Print the amount details for Fraudulent Transaction
print("Amount details of the fraudulent transaction")
print(fraud.Amount.describe())

# Print the amount details for Normal Transaction
print("Details of valid transaction")
print(valid.Amount.describe())

# Plotting the Correlation Matrix
# Correlation matrix
corrmat = data.corr()
plt.figure(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True)
plt.title("Correlation Matrix")
plt.show()

# Separating the X and the Y values
# Dividing the data into input parameters and output value
X = data.drop(['Class'], axis=1)
Y = data["Class"]
print(X.shape)
print(Y.shape)

# Training and Testing Data Bifurcation
# Using Scikit-learn to split data into training and testing sets
xData = X.values
yData = Y.values
xTrain, xTest, yTrain, yTest = train_test_split(xData, yData, test_size=0.2, random_state=42)

# Building a Random Forest Model using scikit-learn
# Random forest model creation
rfc = RandomForestClassifier()
rfc.fit(xTrain, yTrain)
# Predictions
yPred = rfc.predict(xTest)

# Building all kinds of evaluating parameters
# Evaluating the classifier
print("The model used is Random Forest classifier")

acc = accuracy_score(yTest, yPred)
print("The accuracy is {}".format(acc))

prec = precision_score(yTest, yPred)
print("The precision is {}".format(prec))

rec = recall_score(yTest, yPred)
print("The recall is {}".format(rec))

f1 = f1_score(yTest, yPred)
print("The F1-Score is {}".format(f1))

MCC = matthews_corrcoef(yTest, yPred)
print("The Matthews correlation coefficient is {}".format(MCC))

# Visualizing the Confusion Matrix
# Printing the confusion matrix
LABELS = ['Normal', 'Fraud']
conf_matrix = confusion_matrix(yTest, yPred)
plt.figure(figsize=(12, 12))
sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d")
plt.title("Confusion Matrix")
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.show()