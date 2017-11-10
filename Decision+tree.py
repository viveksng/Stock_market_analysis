
# coding: utf-8

# # load Dataset and preprocessing, import Depedencies

# In[144]:

from azureml import Workspace

ws = Workspace()
ds = ws.datasets['PM data.csv']
frame = ds.to_dataframe()


# In[145]:

frame


# In[146]:

import pandas as pd
import numpy as np
np.random.seed(10)


# In[147]:

td = frame
td.columns=['y','d','n','s']


# In[148]:

ti={'Sell':0,'Hold':1,'Buy':2}
ts={'P':1,'L':-1,'C':0}


# In[149]:

td['y']=td['y'].map(ti)


# In[150]:

td


# In[151]:

td['s']=td['s'].map(ts)


# In[152]:

td


# In[153]:


X = td[['d','n','s']]
Y = td['y']
X = np.array(X)
Y = np.array(Y)


# # Split the Dataset into train-test 

# In[154]:

from  sklearn.model_selection import train_test_split


# In[155]:

X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.40, random_state=42)


# In[156]:

from sklearn.model_selection import KFold
kf = KFold(n_splits=20, random_state=None, shuffle=True)
kf


# # DecisionTree

# In[157]:

from sklearn import tree 


# In[158]:

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train,y_train) # name of the classifire
clf


# In[159]:

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score


# In[160]:

y_pred = clf.predict(X_test)


# In[161]:

confusion_matrix(y_test, y_pred)


# In[162]:

accuracy_score(y_test, y_pred)


# In[163]:

f1_score(y_test, y_pred,average='macro')


# In[164]:

clf.predict([[.5,-.9,0]]) # {'sell':0,'hold':1,'buy':2}


# In[ ]:



