import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plot
from sklearn.model_selection import train_test_split

# ".values" to change from panda's dataframe 
# data structure to numpy's array
data = pd.read_csv('data/banknotes.csv').values

# get 70% as training data, 30% as test data
# len(x_train) must match len(y_train)
# y_train are the "labels/classes" for x_train
x_train, x_test, y_train, y_test = train_test_split(
	data[:,1:-1], data[:,-1:], test_size=0.3)

# convert from python's list to numpy's array
# tensorflow library expecting numpy's array
x_train = np.asarray(x_train).astype('float32')
y_train = np.asarray(y_train).astype('float32')
x_test = np.asarray(x_test).astype('float32')
y_test = np.asarray(y_test).astype('float32')

# one-hot encode the single digit labels of
# y_train and y_test 
y_train = tf.keras.utils.to_categorical(y_train, 2)
y_test = tf.keras.utils.to_categorical(y_test, 2)

# create neural network 
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(units=80, 
	input_shape=(x_train.shape[1],), activation='sigmoid'))
model.add(tf.keras.layers.Dense(units=50, activation='sigmoid'))
model.add(tf.keras.layers.Dense(units=2, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy',
	metrics=['accuracy'])
model.fit(x_train, y_train, epochs=200)

# use our test data to evaluate the accuracy of our classifier
loss, accuracy = model.evaluate(x_test, y_test, verbose=1)
print(loss, accuracy)



