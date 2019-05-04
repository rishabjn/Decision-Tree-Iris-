import numpy as np 
import pandas as pd 
from matplotlib import pyplot as plt 
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

iris_data = pd.read_csv("iris.csv") #Reading data using pandas

featues = iris_data[["SepalLength","SepalWidth","PetalLength","PetalWidth"]] #Selecting features for training

target_var = iris_data.Class   #selecting target class

ftrain, ftest, ttrain,ttest = train_test_split(featues,target_var,test_size=0.2)  #splitting dataset into train set and test set

model = DecisionTreeClassifier() # Running the Model 

fitted_model = model.fit(ftrain, ttrain) #fitting on the training dataset


prediction = fitted_model.predict(ftest) #prediction on test dataset

print(confusion_matrix(ttest,prediction))  #Printing confusion matrix and accuracy 
print(accuracy_score(ttest, prediction))
