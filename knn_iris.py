import numpy as np
import pandas as pd

dataset = pd.read_csv('iris.csv') #Reading dataset

feature_columns = ['SepalLength', 'SepalWidth', 'PetalLength','PetalWidth'] #feature selection
X = dataset[feature_columns].values
y = dataset['Class'].values

from sklearn.preprocessing import LabelEncoder #Providing numeric values to categorical output data
le = LabelEncoder()
y = le.fit_transform(y)

from sklearn.model_selection import train_test_split #splitting data into test and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score

# Instantiate learning model (k = 3)
classifier = KNeighborsClassifier(n_neighbors=3)

# Fitting the model
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
print(cm)

accuracy = accuracy_score(y_test, y_pred)*100
print('Accuracy of our model is equal ' + str(round(accuracy, 2)) + ' %.')
