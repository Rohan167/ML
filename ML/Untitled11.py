#!/usr/bin/env python
# coding: utf-8

# In[10]:


import pandas as pd
from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import train_test_split 
from sklearn import metrics
from pandas import DataFrame 
pima = DataFrame.from_csv('/home/su/Desktop/diabetes.csv')
pima.head()
# Get Predictor Names (all but 'class')
X = list(pima.columns)
print("List of Attributes:", X) 
X.remove('Outcome') #Remove the class attribute 
print("Predicting Attributes:", X)
X=pima[X]
y=pima.Outcome
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) 
clf = DecisionTreeClassifier()
clf = clf.fit(X,y)
from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus
dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,filled=True, rounded=True,special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())


# In[7]:


conda install python-graphviz


# In[8]:


conda install -c anaconda graphviz

