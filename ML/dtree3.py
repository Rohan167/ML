#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[26]:



from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import accuracy_score 
from sklearn import tree 
from sklearn.preprocessing import LabelEncoder

import pandas as pd 
import numpy as np 


# In[27]:


df = pd.read_csv('D://ML//play_tennis.csv') 


# In[28]:


print(df)


# In[44]:


lb = LabelEncoder() 
df['outlook'] = lb.fit_transform(df['outlook']) 
df['temp'] = lb.fit_transform(df['temp'] ) 
df['humidity'] = lb.fit_transform(df['humidity'] ) 
df['wind'] = lb.fit_transform(df['wind'] )   
df['play'] = lb.fit_transform(df['play'] ) 
print(df)

X = df.iloc[:,1:5] 
print(X)

Y = df.iloc[:,5]
print(Y)


# In[45]:


clf_entropy = DecisionTreeClassifier(criterion='entropy')


# In[46]:


dtree=clf_entropy.fit(X, Y)


# In[47]:


from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus
dot_data = StringIO()
export_graphviz(dtree, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())


# In[ ]:




