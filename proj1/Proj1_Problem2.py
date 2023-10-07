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
tol_accuracy_map = {1: 0, 1e-1: 0, 1e-2: 0, 1e-3: 0, 1e-4: 0}


for key in tol_accuracy_map.keys():
    clf = Perceptron(tol=key, random_state=0)
    clf.fit(X_train, Y_train)
    tol_accuracy_map[key] = round(clf.score(X_test, Y_test) * 100, 4) 

print("Accuracy for different tolerance values (%): ", tol_accuracy_map)


# ### 2. LOGISTIC REGRESSION

print("Running Logistic Regression...")
solver_accuracy_map = {'lbfgs': 0, 'liblinear': 0, 'newton-cg': 0, 'sag': 0, 'saga': 0}


for solver in solver_accuracy_map.keys():
    clf = LogisticRegression(solver=solver,
            random_state=0)
    clf.fit(X_train, Y_train)
    solver_accuracy_map[solver] = round(clf.score(X_test, Y_test) * 100, 4)


print("Accuracy for different solvers (%): ", solver_accuracy_map)


# ### 3. SUPPORT VECTOR MACHINE

print("Running Support Vector Machine...")
kernel_accuracy_map = {'linear': 0, 'sigmoid': 0, 'rbf': 0}


for key in kernel_accuracy_map.keys():
    clf = SVC(kernel=key)
    clf.fit(X_train, Y_train)
    kernel_accuracy_map[key] = round(clf.score(X_test, Y_test) * 100, 4)


print("Accuracy for different kernels (%): ", kernel_accuracy_map)


# ### 4. DECISION TREE LEARNING
print("Running Decision Tree Algorithm...")

clf = DecisionTreeClassifier(criterion='entropy')
clf.fit(X_train, Y_train)


print("Accuracy (%): ", round(clf.score(X_test, Y_test) * 100, 4))


# ### 5. RANDOM FOREST CLASSIFIER

print("Running Random Forest Classifier Algorithm...")
clf = RandomForestClassifier(max_depth=4, random_state=4)
clf.fit(X_train, Y_train)


print("Accuracy (%) :", round(clf.score(X_test, Y_test) * 100, 4))


# ### 6. K-NEAREST NEIGHBOR

print("Running K-NEAREST NEIGHBOR Algorithm...")
knn_accuracies = defaultdict(int)
for k in range(1, 100):
    clf = KNeighborsClassifier(algorithm='auto', n_neighbors=k)
    clf.fit(X_train, Y_train)
    knn_accuracies[k] = round(clf.score(X_test, Y_test) * 100, 4)
    
max_accuracies = max(knn_accuracies, key=lambda x: knn_accuracies[x])
kmax = f"No. of neighbors={max_accuracies} with accuracy of {knn_accuracies[max_accuracies]}%"
print(kmax)



min_accuracies = min(knn_accuracies, key=lambda x: knn_accuracies[x])
kmin = f"No. of neighbors={min_accuracies} with accuracy of {knn_accuracies[min_accuracies]}%"
print(kmin)