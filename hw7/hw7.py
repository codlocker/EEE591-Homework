# Problem 1: 
# 
# - Read the database in from this heart1.csv file and analyze the data.
# - Your analysis should include a statistical study of each variable: correlation of each variable, dependent  or independent, with all the other variables. Determine which variables are most highly correlated with each other and also which are highly correlated with the variable you wish to predict. 
# - Create a cross covariance matrix to show which variables are not independent of each other and which ones are best predictors of heart disease. Create a pair plot.
# - Based on this analysis you must determine what you think you will be able to do and which variables you think are most likely to play a significant roll in predicting the dependent variable, in this case occurrence of heart disease. 
# 
# - Your management at AMAPE want to be kept constantly updated on your progress. Write one paragraph based on these results indicating what you have learned from this analysis. We are looking for specific observations

# ### Project 1 part 2 results


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
from sklearn.metrics import accuracy_score
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


# ## Homework 7 : Additional Assignment
# 
#  In Project 1, you used 6 different methods. Here's what you need to do:
# 
# Step 1: Pick the 3 methods which got the best accuracy for you in Project 1 and use them to vote. Does your accuracy improve?
# 
# Step 2: Now add the 4th best method. Does accuracy improve? What do you decide to do in the case of a tie?
# 
# Step 3: Now add the 5th best method. Does accuracy improve?

# using the best perceptron algorithm
perceptron_model = Perceptron(tol=1, random_state=0)
perceptron_model.fit(X_train, Y_train)
perceptron_predict = perceptron_model.predict(X_test)
perceptron_accuracy = np.round(perceptron_model.score(X_test, Y_test) * 100, 4)
print('Accuracy of Perceptron(%) =', perceptron_accuracy)

# using the best logistric regression model
logistic_model = LogisticRegression(solver='liblinear', random_state=0)
logistic_model.fit(X_train, Y_train)
logistic_predict = logistic_model.predict(X_test)
logistic_accuracy = np.round(logistic_model.score(X_test, Y_test) * 100, 4)
print('Accuracy of Logistic model(%) =', logistic_accuracy)

# using the best support vector machine model
svm_model = SVC(kernel='rbf')
svm_model.fit(X_train, Y_train)
svm_predict = svm_model.predict(X_test)
svm_accuracy = np.round(svm_model.score(X_test, Y_test) * 100, 4)
print('Accuracy of SVM model(%) =', svm_accuracy)

# using the best decision tree model
dtree_model = DecisionTreeClassifier()
dtree_model.fit(X_train, Y_train)
dtree_predict = dtree_model.predict(X_test)
dtree_accuracy = np.round(dtree_model.score(X_test, Y_test) * 100, 4)
print('Accuracy of Decision tree model(%) =', dtree_accuracy)

# using the random tree classifier
raf_model = RandomForestClassifier(max_depth=4, random_state=0)
raf_model.fit(X_train, Y_train)
raf_predict = raf_model.predict(X_test)
raf_accuracy = np.round(raf_model.score(X_test, Y_test) * 100, 4)
print('Accuracy of Random Forest model(%) =', raf_accuracy)

# using the KNN model
knn_model = KNeighborsClassifier(algorithm='auto', n_neighbors=13)
knn_model.fit(X_train, Y_train)
knn_predict = knn_model.predict(X_test)
knn_accuracy = np.round(knn_model.score(X_test, Y_test) * 100, 4)
print('Accuracy of KNN model(%) =', knn_accuracy)

# Ensemble predictions. Computes the true labels using a majority voting logic
# Prints the tie scenarios in cases as required based on flag.
#    Args:
#        algo_predictions (list): Predicated labels from algorithms in the ensemble.
#        true_labels (np.array): The true labels as obtained from the split.
#        is_print (bool, optional): Print the tie labels. Defaults to False.

#    Returns:
#        float: Return accuracy as a percentage
def ensemble_prediction(algo_predictions: list, true_labels: np.array, is_print: bool = False) -> float:
    print(f'Running ensemble of {len(algo_predictions)}..')
    ensemble = np.array(algo_predictions).T
    final_preds = np.zeros_like(true_labels)
    
    for idx, pred in enumerate(ensemble):
        if np.count_nonzero(pred == 1) > np.count_nonzero(pred == 2):
            final_preds[idx] = 1
        elif np.count_nonzero(pred == 2) > np.count_nonzero(pred == 1):
            final_preds[idx] = 2
        else:
            final_preds[idx] = np.random.choice([1, 2])
            if is_print:
                print(f'Found a tie : Selected choice : {int(final_preds[idx])} as target class')
    
    return np.round(accuracy_score(final_preds, true_labels) * 100, 4)




# Run ensemble of the 3 best methods
ensemble_3 = [raf_predict, knn_predict, svm_predict]
accuracy_3 = ensemble_prediction(ensemble_3, Y_test)
print(f'Accuracy of ensemble predictions using Random Forest, KNN and SVM = {accuracy_3}%')


# Run ensemble of the 4 best methods
ensemble_4 = [raf_predict, knn_predict, svm_predict, perceptron_predict]
accuracy_4 = ensemble_prediction(ensemble_4, Y_test, True)
print(f'Accuracy of ensemble predictions using Random Forest, KNN, perceptron and SVM = {accuracy_4}%')

if accuracy_4 > accuracy_3:
    print(f"Accuracy of ensemble of 4 methods improved over ensemble of 3.")
else:
    print(f"Accuracy of ensemble of 4 methods did not improve over ensemble of 3.")


# Run ensemble of the 5 best methods
ensemble_5 = [raf_predict, knn_predict, svm_predict, perceptron_predict, logistic_predict]
accuracy_5 = ensemble_prediction(ensemble_5, Y_test)
print(f'Accuracy of ensemble predictions using Random Forest, KNN, Logistic, Perceptron and SVM = {accuracy_5}%')

if accuracy_5 > accuracy_4:
    print(f"Accuracy of ensemble of 5 methods improved over ensemble of 4.")
else:
    print(f"Accuracy of ensemble of 5 methods did not improve over ensemble of 4.")