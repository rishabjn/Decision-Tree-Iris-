import numpy as np 
import pandas as pd 
from matplotlib import pyplot as plt 
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

iris_data = pd.read_csv("iris.csv")

featues = iris_data[["SepalLength","SepalWidth","PetalLength","PetalWidth"]]

target_var = iris_data.Class

ftrain, ftest, ttrain,ttest = train_test_split(featues,target_var,test_size=0.2)

model = DecisionTreeClassifier()

fitted_model = model.fit(ftrain, ttrain)


prediction = fitted_model.predict(ftest)

print(confusion_matrix(ttest,prediction))
print(accuracy_score(ttest, prediction))
