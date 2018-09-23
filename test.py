import pandas as pd
import os

# For Reference go to:
# https://github.com/rasbt/python-machine-learning-book-2nd-edition/blob/master/code/ch03/ch03.ipynb

# Load the data
#df = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data', header=None)
os.chdir('E:\\Programming\\Data Mining\\Machine Learning\\Assignments')
df = pd.read_csv('breast_data.csv')
df.columns = ['Sample code number', 'Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape',
              'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli',
              'Mitoses', 'target']
df.head()

# Remove 'Sample code number' (identifier, unique for each sample)
# Implement Me
df = df.drop(['Sample code number'], axis=1)
df.head()

# Remove Rows with Missing Values
import numpy as np
print('Number of rows before removing rows with missing values: ' + str(df.shape[0]))

# Replace ? with np.NaN
df.replace('?', np.NaN, inplace=True)

# Remove rows with np.NaN
# Implement me
df.dropna(inplace=True)
print('Number of rows after removing rows with missing values: ' + str(df.shape[0]))

# Get the feature vector
# Implement me
X = df.iloc[:, :-1].values

# Get the target vector
# Implement me
y = df['target'].values

from sklearn.model_selection import train_test_split

# Randomly choose 30% of the data for testing (set randome_state as 0 and stratify as y)
# Implement me
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)

from sklearn.preprocessing import StandardScaler

std_scaler = StandardScaler()

# Standardize the training data
# Implement me
X_train = std_scaler.fit_transform(X_train)

# Standardize the testing data
# Implement me
X_test = std_scaler.transform(X_test)

# Slow Logistic Regression Classifier

from sklearn.metrics import accuracy_score


class MySlowLogisticRegression():
    """The slow logistic regression classifier (implemented heavily by list)"""

    def __init__(self, eta=0.01, n_iter=100):
        # Initialize the learning rate
        self.eta = eta
        # Initialize the number of iterations
        self.n_iter = n_iter

    def fit(self, X, y):
        """
        The fit function

        Parameters
        ----------
        X : the feature vector
        y : the target vector
        """

        # The dictionary of the weights
        self.w_ = {}
        # For each class label
        for class_ in np.unique(y):
            # Initialize the weight for each feature (and the dummy feature, x0)
            self.w_[class_] = np.zeros(1 + X.shape[1])

        # For each iteration
        for _ in range(self.n_iter):
            # For each class label
            for class_ in self.w_.keys():
                # Initialize the update (of the weight) for each feature (and the dummy feature, x0)
                delta_w = np.zeros(1 + X.shape[1])

                # For each sample
                for i in range(X.shape[0]):
                    # Get the net_input
                    # Implement me
                    z = self.net_input(X, class_, i)

                    # Get the logistic sigmoid activation
                    # Implement me
                    prob = self.activation(z)

                    # Get the error
                    # Implement me
                    error = (y[i] - prob)

                    # Get the update (of the weight) for each feature
                    for j in range(1, X.shape[1] + 1):
                        delta_w[j] += self.eta * error * X[i][j - 1]

                    # Get the update (of the weight) for the dummy feature, x0
                    delta_w[0] += self.eta * error

                # Update the weight for each feature (and the dummy feature, x0)
                # Implement me
                self.w_[class_] += delta_w

    def net_input(self, X, class_, i):
        """
        Get the net input

        Parameters
        ----------
        X : the feature vector
        class_ : a class label of the target
        i : the ith sample

        Returns
        ----------
        The net input

        """

        # Initialize the weighted sum (i.e., the net input)
        weighted_sum = self.w_[class_][0]

        # For each feature
        for j in range(1, X.shape[1] + 1):
            # Implement me
            weighted_sum += X[i][j - 1] * self.w_[class_][j]

        return weighted_sum

    def activation(self, z):
        """
        Get the logistic sigmoid activation
        Reference: the function is from the "Python Machine Learning (2nd edition)" book code repository and info resource
        https://github.com/rasbt/python-machine-learning-book-2nd-edition
        """
        return 1. / (1. + np.exp(-np.clip(z, -250, 250)))

    def predict(self, X):
        """
        The predict function

        Parameters
        ----------
        X : the feature vector

        Returns
        ----------
        The predicted class labels of the target
        """

        # The predicted class labels
        y_pred = []

        # For each sample
        for i in range(X.shape[0]):
            # The list of [probability, class]
            prob_classes = []

            # For each class label
            for class_ in self.w_.keys():
                # Get the net_input
                # Implement me
                z = self.net_input(X, class_, i)

                # Get the logistic sigmoid activation
                # Implement me
                prob = self.activation(z)

                # Update prob_classes
                prob_classes.append([prob, class_])

            # Sort prob_classes in descending order of probability
            prob_classes = sorted(prob_classes, key=lambda x: x[0], reverse=True)

            # Get the predicted class label (the one with the largest probability)
            # Implement me
            pred_class = np.argmax(prob_classes)  # prob_classes[-1]

            # Update y_pred
            y_pred.append(pred_class)

        return y_pred

    def score(self, X, y, sample_weight=None):
        """
        Parameters
        ----------
        X : the feature vector
        y : the target vector
        sample_weight : sample weight (None by default)

        Returns
        ----------
        The mean accuracy
        """

        return accuracy_score(y, self.predict(X), sample_weight=sample_weight)

# Evaluate the slow logistic regression model

import time

# The start time
start = time.time()

slow_lr = MySlowLogisticRegression()

# Train the model
slow_lr.fit(X_train, y_train)
score = str(slow_lr.score(X_test, y_test))
# Print the accuracy
print('Accuracy: ' + score)

# The end time
end = time.time()

# Print the Run time
print('Run time: ' + str(end - start))

# Results should be around:
# Accuracy: 0.9609756097560975
#
# Run time: 1.591623067855835

class MyFastLogisticRegression():
    """The fast logistic regression classifier (implemented largely by numpy array)"""

    def __init__(self, eta=0.01, n_iter=100):
        # Initialize the learning rate
        self.eta = eta
        # Initialize the number of iterations
        self.n_iter = n_iter

    def fit(self, X, y):
        """
        The fit function

        Parameters
        ----------
        X : the feature vector
        y : the target vector
        """

        # Get the (unique) class labels
        self.classes_ = np.unique(y)

        # Get the number of (unique) class labels
        class_num = len(self.classes_)

        # Encode the target using one-hot-encoding
        ys = np.zeros((X.shape[0], class_num))
        # For each class label
        for j in range(class_num):
            ys[np.where(y == self.classes_[j]), j] = 1

        # Initialize the weight for each feature (and the dummy feature, x0)
        self.w_ = np.zeros((1 + X.shape[1], class_num))

        # For each iteration
        for _ in range(self.n_iter):
            # Get the net_input
            # Implement me
            net_input = self.net_input(X)

            # Get the output
            # Implement me
            output = self.activation(net_input)

            # Get the errors
            # Implement me
            errors = (y[_] - output)

            # Get the update (of the weight) for each feature
            self.w_[1:, :] += self.eta * X.T.dot(errors)

            # Get the update (of the weight) for the dummy feature, x0
            self.w_[0, :] += self.eta * errors.sum()

    def net_input(self, X):
        """
        Get the net input

        Parameters
        ----------
        X : the feature vector
        class_ : a class label of the target

        Returns
        ----------
        The net input

        """

        return np.matmul(X, self.w_[1:, :]) + self.w_[0, :]

    def activation(self, z):
        """
        Get the logistic sigmoid activation
        Reference: the function is from the "Python Machine Learning (2nd edition)" book code repository and info resource
        https://github.com/rasbt/python-machine-learning-book-2nd-edition
        """
        return 1. / (1. + np.exp(-np.clip(z, -250, 250)))

    def predict(self, X):
        """
        The predict function

        Parameters
        ----------
        X : the feature vector

        Returns
        ----------
        The predicted class label of the target
        """

        # Get the net_input
        # Implement me
        net_input = self.net_input(X)

        # Get the output
        # Implement me
        output = self.activation(net_input)

        return np.asarray([self.classes_[np.argmax(probs)] for probs in output])

    def score(self, X, y, sample_weight=None):
        """
        Parameters
        ----------
        X : the feature vector
        y : the target vector
        sample_weight : sample weight (None by default)

        Returns
        ----------
        The mean accuracy
        """

        return accuracy_score(y, self.predict(X), sample_weight=sample_weight)

# The start time
start = time.time()

fast_lr = MyFastLogisticRegression()

# Train the model
fast_lr.fit(X_train, y_train)

# Print the accuracy
print('Accuracy: ' + str(fast_lr.score(X_test, y_test)))

# The end time
end = time.time()

# Print the Run time
print('Run time: ' + str(end - start))

# Results should be around:
# Accuracy: 0.9658536585365853
#
# Run time: 0.00993204116821289

from sklearn.linear_model import LogisticRegression

# The start time
start = time.time()

sklearn_lr = LogisticRegression()

# Train the model
sklearn_lr.fit(X_train, y_train)

# Print the accuracy
print('Accuracy: ' + str(sklearn_lr.score(X_test, y_test)))

# The end time
end = time.time()

# Print the Run time
print('Run time: ' + str(end - start))

# Load the data
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data', header=None)
df.columns = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'target']
df.head()

from sklearn.preprocessing import LabelEncoder

# Get the feature vector
# Implement me
X = df.iloc[:, :-1].values

# Encode the feature vector using one-hot-encoding
# Implement me
X = LabelEncoder().fit_transform(X.flatten()).reshape(X.shape)

# Get the target vector
# Implement me
y = df['target'].values

# Encode the target vector
# Implement me
le = LabelEncoder()
y = le.fit_transform(y)

# Randomly choose 30% of the data for testing (set randome_state as 0 and stratify as y)
# Implement me
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)

