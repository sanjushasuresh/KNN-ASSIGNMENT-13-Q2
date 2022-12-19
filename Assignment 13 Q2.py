# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 18:25:58 2022

@author: SANJUSHA
"""

import numpy as np 
import pandas as pd

df=pd.read_csv("Zoo.csv")
df.shape
df.head()

# Dropping the first column 
df.drop(["animal name"],axis=1,inplace=True)
df.head()
df.info()
df.corr()
# Type and backbone has a high correlation of -0.82884, eggs and hair has high correlation of 
# -0.81738, milk and hair has the highest positive correlation of 0.87850 and eggs and milk has
# the highest negative correlation of -0.93884

# Splitting the variables 
X=df.iloc[:,0:16]
Y=df["type"]

# There is relation btw hair,eggs,milk
X1=df.drop(df.columns[[0,2,16]],axis=1)
X1.columns
X1.shape
X1.head()

# Standardization
from sklearn.preprocessing import MinMaxScaler
MM=MinMaxScaler()

X=MM.fit_transform(X)
X=pd.DataFrame(X)

X1=MM.fit_transform(X1)
X1=pd.DataFrame(X1)
 
# MODEL FITTING BY X 

from sklearn.neighbors import KNeighborsClassifier
KNN=KNeighborsClassifier(n_neighbors=5)
KNN.fit(X,Y)
Y_predict=KNN.predict(X)

from sklearn.metrics import accuracy_score
as1=accuracy_score(Y,Y_predict)
print(as1) # 0.95049
# Accuracy is 95%

from sklearn.model_selection import KFold, cross_val_score
k=10
k_fold=KFold(n_splits=k, random_state=None)
cv_scores=cross_val_score(KNN, X, Y, cv=k_fold)
mean_acc_score=sum(cv_scores)/len(cv_scores)
# Mean accuracy score = 93%

from sklearn.metrics import roc_curve, roc_auc_score
KNN.predict_proba(X)[:,1]
fpr, tpr, threshold  = roc_curve(Y,KNN.predict_proba(X)[:,1], pos_label=1)
import matplotlib.pyplot as plt
plt.plot(fpr,tpr)
plt.ylabel('tpr - True Positive Rate')
plt.xlabel('fpr - False Positive Rate')
plt.show()

aucvalue = roc_auc_score(Y,KNN.predict_proba(X), multi_class='ovo')
print("aucvalue", aucvalue.round(3))
# Aucvalue = 0.997


#=============================================================================================#


# MODEL FITTING BY X1

from sklearn.neighbors import KNeighborsClassifier
KNN=KNeighborsClassifier(n_neighbors=5)
KNN.fit(X1,Y)
Y_predict=KNN.predict(X1)

from sklearn.metrics import accuracy_score
as2=accuracy_score(Y,Y_predict)
print(as2) # 0.94059
# Accuracy is 94%

from sklearn.model_selection import KFold, cross_val_score
k=7
k_fold=KFold(n_splits=k, random_state=None)
cv_scores=cross_val_score(KNN, X1, Y, cv=k_fold)
mean_acc_score=sum(cv_scores)/len(cv_scores)
# Mean accuracy score = 92%

from sklearn.metrics import roc_curve, roc_auc_score
KNN.predict_proba(X1)[:,1]
fpr, tpr, threshold  = roc_curve(Y,KNN.predict_proba(X1)[:,1], pos_label=1)
import matplotlib.pyplot as plt
plt.plot(fpr,tpr)
plt.ylabel('tpr - True Positive Rate')
plt.xlabel('fpr - False Positive Rate')
plt.show()

aucvalue = roc_auc_score(Y,KNN.predict_proba(X1), multi_class='ovo')
print("aucvalue", aucvalue.round(3))
# Aucvalue = 0.992
