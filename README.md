# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
STEP 1: Start the program.

STEP 2: Load the Iris dataset and create a Pandas DataFrame with features and target.

STEP 3: Split the dataset into features (X) and target (y).

STEP 4: Split the data into training and testing sets using train_test_split.

STEP 5: Initialize the SGDClassifier with default parameters.

STEP 6: Train the classifier using the training data.

STEP 7: Predict the target values for the testing set.

STEP 8: Calculate and display the model's accuracy and confusion matrix.

STEP 9: End the program.
## Program:
```
# Program to implement the prediction of iris species using SGD Classifier.
# Developed by: JONATHAN SAMRAJ A
# Register Number: 212225040160

from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load Iris Dataset
iris = load_iris()

# Create DataFrame
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target

print(df.head())

# Split features and target
X = df.drop('target', axis=1)
y = df['target']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create SGD Classifier
sgd_clf = SGDClassifier(max_iter=1000, tol=1e-3)

# Train the model
sgd_clf.fit(X_train, y_train)

# Make predictions
y_pred = sgd_clf.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.3f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:")
print(cm)

# Plot Confusion Matrix
plt.figure(figsize=(6, 4))

sns.heatmap(
    cm,
    annot=True,
    cmap="Blues",
    fmt='d',
    xticklabels=iris.target_names,
    yticklabels=iris.target_names
)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")

plt.show()
```

## Output:
<img width="973" height="741" alt="image" src="https://github.com/user-attachments/assets/8e55bc9c-416b-4529-b44e-ae0683292ccb" />



## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
