import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import matplotlib.pyplot as plt


pd_df = pd.read_csv("play_tennis.csv")
# pd_df = pd.DataFrame(pd_rd)

y = pd_df["play"]
x = pd_df.drop(["play"],axis=1)

#Convert into categorical data
x_dummies = pd.get_dummies(x)


#Convert the whole dataframe in training and test data

x_train , x_test , y_train , y_test  = train_test_split(x_dummies , y , test_size=0.3)

#Use decision tree classifier with Information Gain based on Entropy

clf = tree.DecisionTreeClassifier(criterion="entropy")
clf_fit = clf.fit(x_train , y_train) #Fit the model on the training set for calculation


#Prediction
y_pred = clf.predict(x_test)


#Validate the model via accuracy
cm = confusion_matrix(y_test , y_pred)
print(clf.score(x_test , y_test)*100)

cols = list(x_dummies.columns.values)
cols = cols[14:]
print(cols)


from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus
dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('play.png')
Image(graph.create_png())