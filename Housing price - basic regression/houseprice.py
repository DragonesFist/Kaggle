# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 16:59:58 2020

@author: Jagadeesh
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder as ohe
from matplotlib import pyplot as plt
import seaborn as sns 
from sklearn.linear_model import LinearRegression as lr
from sklearn.linear_model import LogisticRegression as lg

store_train = {}
store_test = {}
input_X = pd.read_csv('train.csv', sep=',')
inde = input_X['Id']

input_X.describe()
corr_data=input_X
nu = input_X.isna().sum()
nu.to_numpy
input_X.head(5)
# there are no null values in the dataset but LotFrontage has 259 na values which is fine
input_price = input_X['SalePrice']
input_X.drop('SalePrice', inplace=True, axis=1)

a4_dims = (11.7, 8.27)
corr_mat =input_X.corr().round(2)
# fig, ax = pyplot.subplots(figsize=(10,10))
sns.heatmap(data=corr_mat, annot=True, linewidths=2.0)

# collecting all the categorical values into a dataframe
column = input_X.select_dtypes(['object']).columns
col = pd.DataFrame()
for i in range(len(column)):
    print('Scanning '+str(i)+' column')
    temp = pd.DataFrame()
    temp = input_X[column[i]].copy()
    print(temp)
    col = pd.concat([col,temp],axis=1,ignore_index=True)

col.columns=column
#find out the null values
count_null_cat = len(col)-col.count()

#plot a scatter plot for null values
plt.figure(1)
x_axis=count_null_cat.index
y_axis=count_null_cat.values
plt.scatter(x_axis,y_axis)
plt.show()

#plot for misc feature and sales price
plt.figure(num=2,figsize=(30,10), dpi=80)
x_axis=col['MiscFeature'].index
y_axis=input_price.values
plt.scatter(x_axis,y_axis)  
plt.show()

#genrating a heatmap to visualize the coor
corr_mat2=corr_data.corr().round(2)
plt.figure(num=3,figsize=(30,30),dpi=80)
sns.heatmap(corr_mat2)


 ##dataframe with cat variables as column names
col.columns=column 
col_enco = pd.get_dummies(col)

#col_enco=col_enco.T.reindex(col_enco_cp.columns).T.fillna(0) #this is used when we get different number of columns after using dummy encoding technique.
train=input_X
train.drop(column,inplace=True,axis=1)
train = pd.concat([train,col_enco],axis=1)
train = train.drop(['Id'],axis=1)
train = train.fillna(0)
train.isna()


#linear Regression
lr_res = lr().fit(train,input_price)
lr_res.score(train,input_price) #0.93
result = lr_res.predict(train)
fin = pd.DataFrame({'Id':inde,'SalePrice':result[:,]})
fin.to_csv('sample_submission.csv',header=True,index=False)
#logistic regression
lg_res = lg(random_state=0).fit(train,input_price)
lg_res.score(train,input_price) #0.85



    
