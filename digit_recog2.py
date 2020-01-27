# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 22:24:57 2020

@author: 240022854
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
import tensorflow.keras.layers as tk
import tensorflow.keras.utils as tku
#data 

input_X = pd.read_csv('train.csv')
input_X[0:]
label_X = input_X.iloc[:,0]
label_X = label_X.values
input_X.drop(input_X.columns[0],axis=1,inplace=True)
input_X.shape
# input_X = input_X.iloc[1:]
input_X= input_X.values
label_X.shape
input_X = input_X/255
input_X.shape
#conversion of categorical values to binary vector
label_X = tf.keras.utils.to_categorical(label_X, num_classes= 10, dtype='float32')
label_X.shape
# spilting the dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(input_X, label_X, test_size=0.30, random_state=42)
X_train.shape
y_train.shape
#neural network start
# input_dim = Input(shape=(784,))
model = tf.keras.Sequential()
model.add(tk.Dense(64,kernel_initializer='glorot_uniform',input_dim=784,activation='relu')) 
model.add(tk.Dropout(0.5))
model.add(tk.Dense(32,activation='relu'))
model.add(tk.Dropout(0.5))
model.add(tk.Dense(10,activation='softmax'))
model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])
model.fit(X_train,y_train,epochs=10,batch_size=1000)
# prediction = model.predict(input_X,batch_size=1000)
# prediction = np.argmax(prediction,axis=1)
# prediction= pd.DataFrame(prediction)
# prediction
# prediction.to_csv('sample_submission.csv',header=True)
# score = model.evaluate(X_test, y_test, batch_size=1000)



