import numpy as np # needed for arrays
import pandas as pd # data frame
import matplotlib.pyplot as plt # modifying plot
from sklearn.model_selection import train_test_split # splitting data
from sklearn.preprocessing import StandardScaler # scaling data
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA # PCA package
from sklearn.metrics import accuracy_score # grading
from sklearn.metrics import confusion_matrix # generate the matrix
import warnings
warnings.filterwarnings('ignore')

# Function to print confusion
# matrix results.
def print_rocks_mine(y_test):
    rocks = 0                # initialize counters
    mines = 0
    for obj in y_test:    # for all of the objects in the test set
        if obj == 2:         # mines are class 2, rocks are class 1
            mines += 1     # increment the appropriate counter
        else:
            rocks += 1
    print("rocks", rocks," mines",mines)    # print the results

data = pd.read_csv('./sonar_all_data_2.csv', header=None)

# data[61].value_counts().plot(kind='bar')

X = data.drop(columns=[60, 61], axis=1)
y = data[60]


# Split the data into training and testing at 70:30 ratio.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)


# ## Perform Scaling

# 1. Perform Standard Scaling
stdscl = StandardScaler()
X_train_std = stdscl.fit_transform(X_train)
X_test_std = stdscl.fit_transform(X_test)


# ## Run PCA for 1 to 61 features over an MLP Classifier

# #### 2) For each number of components used, print the number of components and the accuracy achieved. (Use test accuracy.)
## Parameters obtained after running grid search
HIDDEN_LAYER = (100,)
ALPHA = 1e-5

# Perform PCA for n features
accuracy_comp = list()
model = MLPClassifier(hidden_layer_sizes=HIDDEN_LAYER, activation='logistic', max_iter=2000, alpha=ALPHA,
     solver='adam', tol=0.0001, random_state=0)
for idx in range(1, 61):
    pca = PCA(n_components=idx) # only keep idx "best" features!
    X_train_pca = pca.fit_transform(X_train_std) # apply to the train data
    X_test_pca = pca.transform(X_test_std) # do the same to the test data
    model.fit(X_train_pca, y_train)
    y_pred = model.predict(X_test_pca)
    accuracy = accuracy_score(y_test, y_pred) * 100.0
    
    accuracy_comp.append({
        'component_id': idx,
        'accuracy': accuracy,
        'predictions': y_pred
    }) # Object to store predictions, accuracy and component id for later

    print(f"Epoch : {idx} => Accuracy for MLP on PCA with component_size({idx}) is {accuracy}%")


# #### 3) At the end, print the maximum accuracy along with the number of components that achieved that accuracy.
print(accuracy_comp[:2])

# Get the max accuracy after processing all PCA components for MLP
max_accuracy_item = max(accuracy_comp, key=lambda x: x['accuracy'])

# Get the components which have the maximum accuracy.
components_with_max_accuracy = [item['component_id'] for item in accuracy_comp if item['accuracy'] == max_accuracy_item['accuracy']]
print(f"Max Accuracy: {max_accuracy_item['accuracy']}%")
print(f"Components the max accuracy : {components_with_max_accuracy}")


# #### 4) Plot accuracy versus the number of components.

plt.plot([x['component_id'] for x in accuracy_comp], [y['accuracy'] for y in accuracy_comp])
plt.title('Accuracy versus The number of components.')
plt.xlabel('Number of components')
plt.ylabel('Accuracy')
plt.show()


# #### 5) Print the confusion matrix which results from the analysis which resulted in the maximum accuracy.
# Print rocks and mines
print_rocks_mine(y_test=y_test)

conf_mat = confusion_matrix(
        y_test, max_accuracy_item['predictions'])
print("Confusion Matrix:", conf_mat)

