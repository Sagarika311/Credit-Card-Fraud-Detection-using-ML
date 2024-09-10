# Credit Card Fraud Detection using ML
The aim of this project is to develop a machine learning model using the Random Forest algorithm to detect fraudulent transactions in a credit card dataset. 

## 1.	Importing Libraries:

* The code starts by importing necessary libraries like numpy, pandas, matplotlib, seaborn, and various functions from sklearn (Scikit-learn).
* NumPy is used for efficient numerical operations on arrays and matrices, while Pandas facilitates data manipulation and analysis with its DataFrame structure. Matplotlib provides a wide range of plotting functions for creating static and interactive visualizations, and Seaborn, built on Matplotlib, offers a high-level interface for attractive statistical graphics. Additionally, the gridspec module from Matplotlib allows for flexible arrangement of subplots, enabling the creation of complex visual layouts. Together, these libraries form a powerful toolkit for data analysis and visualization in machine learning projects.
* These libraries provide functionality for data manipulation, visualization, and machine learning.

## 2.	Loading the Data:
•	The dataset is loaded from a CSV file named "credit.csv" using pd.read_csv() from the pandas library.

## 3.	Understanding the Data:
•	data.head() is used to print the first few rows of the dataset, providing a quick overview of the data structure.

## 4.	Describing the Data:
•	The shape of the dataset is printed using data.shape.
•	data.describe() is used to generate descriptive statistics for the dataset, such as count, mean, standard deviation, minimum, and maximum values for each column.

## 5.	Imbalance in the Data:
•	The code identifies the number of fraudulent and valid transactions in the dataset.
•	It calculates the outlier fraction, which is the ratio of fraudulent transactions to valid transactions.
•	The number of fraud cases and valid transactions are printed.

## 6.	Printing Amount Details:
•	The code prints the descriptive statistics for the Amount column separately for fraudulent and valid transactions using fraud.Amount.describe() and valid.Amount.describe(), respectively.

## 7.	Plotting the Correlation Matrix:
•	A correlation matrix is calculated using data.corr().
•	The correlation matrix is visualized using sns.heatmap() from the seaborn library.
•	The resulting heatmap shows the correlation between features in the dataset.

## 8.	Separating X and Y:
•	The features (X) are extracted by dropping the 'Class' column from the dataset using data.drop().
•	The target variable (Y) is extracted by selecting the 'Class' column from the dataset.
•	The shapes of X and Y are printed to verify the dimensions.

## 9.	Training and Testing Data Bifurcation:
•	The feature matrix (X) and target vector (Y) are converted to NumPy arrays using values.
•	train_test_split() from sklearn.model_selection is used to split the data into training and testing sets.
•	The test_size parameter is set to 0.2, indicating that 20% of the data will be used for testing, and the remaining 80% for training.
•	random_state is set to 42 for reproducibility.

## 10.	Building a Random Forest Model:
•	A Random Forest Classifier is created using RandomForestClassifier() from sklearn.ensemble.
•	The classifier is trained on the training data using rfc.fit(xTrain, yTrain).
•	Predictions are made on the test data using rfc.predict(xTest), and the predicted labels are stored in yPred.

## 11.	Evaluating the Classifier:
•	Various evaluation metrics are calculated using functions from sklearn.metrics:
•	accuracy_score: Calculates the overall accuracy of the model.
•	precision_score: Calculates the precision of the model.
•	recall_score: Calculates the recall of the model.
•	f1_score: Calculates the F1-score, which is the harmonic mean of precision and recall.
•	matthews_corrcoef: Calculates the Matthews correlation coefficient, which is a balanced measure of classification performance.
•	The evaluation metrics are printed to assess the performance of the Random Forest Classifier.

## 12.	Visualizing the Confusion Matrix:
•	A confusion matrix is calculated using confusion_matrix() from sklearn.metrics.
•	The confusion matrix is visualized using sns.heatmap() from the seaborn library.
•	The labels for the confusion matrix are defined as LABELS = ['Normal', 'Fraud'].
•	The resulting heatmap shows the number of true positives, true negatives, false positives, and false negatives for the classification task.
In summary, this code loads a credit card transaction dataset, explores its characteristics, handles imbalanced data, calculates correlations, splits the data into training and testing sets, builds a Random Forest Classifier, evaluates its performance using various metrics, and visualizes the confusion matrix to gain insights into the model's predictions.
