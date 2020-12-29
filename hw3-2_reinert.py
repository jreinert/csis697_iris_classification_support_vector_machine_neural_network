# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 21:06:54 2020

@author: reine
"""
#import libraries
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
#from sklearn.metrics import r2_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import OneHotEncoder
import keras
from keras.models import Sequential
from keras.layers import Dense

# Load the Iris Dataset
iris = datasets.load_iris()

# Create the dataframe
iris_df = pd.DataFrame(iris.data, columns = iris.feature_names)
target_names = [iris.target_names[i] for i in iris.target]
iris_df['species'] = target_names

X = iris.data
#Y = iris_df.iloc[:, 4:]
Y = iris.target.reshape(-1,1)

encoder = OneHotEncoder(sparse=False)
Y = encoder.fit_transform(Y)


# Split the data into Train/Test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 42)

# Build the model
num_neurons = 4
num_layers = 4
num_epochs = 500
num_batch = 5
act_func = 'softplus'

model = Sequential()
model.add(Dense(units = num_neurons, kernel_initializer = 'uniform', activation = act_func, input_dim = 4)) # Input layer & 1 hidden layer

# Loop for added hidden layers
i = 0

while i < num_layers:
    model.add(Dense(units = num_neurons, kernel_initializer = 'uniform', activation = act_func))
    i += 1
    
model.add(Dense(units = 3, kernel_initializer = 'uniform', activation = 'softmax')) # Output layer with 3 outputs

# Compile the model
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
model.fit(X_train, Y_train, epochs=num_epochs, batch_size=num_batch)

# Make predictions
iris_pred = model.predict(X_test)
cm = confusion_matrix(Y_test.argmax(axis = 1), iris_pred.argmax(axis = 1))
print(f'\n{cm}')

total_right = cm[0][0] + cm[1][1] + cm[2][2]
total_wrong = cm[0][1] + cm[0][2] + cm[1][0] + cm[1][2] + cm[2][0] + cm[2][1]
total_acc = total_right / (total_right + total_wrong)
print(f'The total accuracy of the model is {total_acc * 100:.2f}%')

