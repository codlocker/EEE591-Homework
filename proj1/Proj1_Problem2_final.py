# ## PART-2 APPLY ML ALGORITHMS


import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from collections import defaultdict
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import StandardScaler


# Read the dataset
heart_df = pd.read_csv('./heart1.csv')


# Split the data X and Y. Y being the target variable.
# X being a collection of dependent and independent variables which needs to be analyzed
X = heart_df.values[:, 0:13]
Y = heart_df.values[:, 13]
validation_size = 0.2
# Split data into training and testing sets
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=42)
print("Shape of X_train after splitting: ", X_train.shape)
print("Shape of X_test after splitting: ", X_test.shape)


# Apply a standard scaler algorithm to scale all 
# features in a fixed range
scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

print("After scaling....")
print(X_train)
# ### 1. PERCEPTRON

print("Running Perceptron Algorithm...")


clf = Perceptron(tol=1, random_state=0)
clf.fit(X_train, Y_train)
print(f'Accuracy : {round(clf.score(X_test, Y_test) * 100, 4)}%') 


# ### 2. LOGISTIC REGRESSION

print("Running Logistic Regression...")

clf = LogisticRegression(
    solver='lbfgs', 
    random_state=0)

clf.fit(X_train, Y_train)
print(f"Accuracy : {round(clf.score(X_test, Y_test) * 100, 4)}%")


# ### 3. SUPPORT VECTOR MACHINE

print("Running Support Vector Machine...")

clf = SVC(kernel='rbf')
clf.fit(X_train, Y_train)
print(f'Accuracy : {round(clf.score(X_test, Y_test) * 100, 4)}%')


# ### 4. DECISION TREE LEARNING
print("Running Decision Tree Algorithm...")

clf = DecisionTreeClassifier(criterion='entropy')
clf.fit(X_train, Y_train)


print(f"Accuracy : {round(clf.score(X_test, Y_test) * 100, 4)}%")


# ### 5. RANDOM FOREST CLASSIFIER

print("Running Random Forest Classifier Algorithm...")
clf = RandomForestClassifier(max_depth=4, random_state=4)
clf.fit(X_train, Y_train)


print(f"Accuracy : {round(clf.score(X_test, Y_test) * 100, 4)}%")


# ### 6. K-NEAREST NEIGHBOR

print("Running K-NEAREST NEIGHBOR Algorithm...")
knn_accuracies = defaultdict(int)
clf = KNeighborsClassifier(algorithm='auto', n_neighbors=13)
clf.fit(X_train, Y_train)
print(f"Accuracy : {round(clf.score(X_test, Y_test) * 100, 4)}% for k = {13}")