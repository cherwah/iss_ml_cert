import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

'''
Train a Neural Network to sum 3 numbers. Here, we are performing 
a form of data regression, where we train our model to learnquit
the correct sum of 3 numbers.
'''

'''
Generate data for training and testing
'''
def generate_data(low, high, n_rows, n_cols):
  # produces a n_rows x n_cols matrix of values between
  # 'low' and 'high'
  x_train = np.random.randint(low, high, size=(n_rows, n_cols))
  
  # sum up values for each row
  # axis=1 means collapse along columns
  # keepdims=True keeps the original dimension (which is
  # array within an array - the form that Keras want) 
  y_train = np.sum(x_train, axis=1, keepdims=True)

  return x_train, y_train


'''
Create our Neural Network model.
'''
def create_model(n_features):
  model = tf.keras.Sequential()

  # can add multiple hidden-layers to our model
  # our activation function is ReLU
  model.add(tf.keras.layers.Dense(50, 
    input_shape=(n_features,), activation='relu'))
  
  # output only has 1 neuron as that's the sum of our value for each row
  model.add(tf.keras.layers.Dense(1))

  # a Keras model must be compiled before it can be used
  # the optimizer is a gradient-descent algorithm
  # our loss function is mean-square error (loss='mse')
  model.compile(optimizer='adam', loss='mse')
  return model


'''
Train our Neural Network model.
'''
def train_model(model, x_train, y_train, epochs):
  # train our model by asking it to learn from the
  # training data; we want it to learn the "relation"
  # between the x_train and y_train data
  return model.fit(x_train, y_train, epochs=epochs)


'''
Test our Neural Network model.
'''
def test_model(model, x_test, y_test):
  # return the average loss between actual and predicted values
  # as computed by the loss-function
  return model.evaluate(x=x_test, y=y_test)
  

'''
Show ground-truths vs predictions.
'''
def show_diffs(model, x_test, y_test):
  # returns an array of predicted values
  predictions = model.predict(x=x_test)

  # display the actual and predicted values
  # for each row of test data
  for i in np.arange(len(predictions)):
    print('Data: {}, Actual: {}, Predicted: {}'.format(\
      x_test[i], y_test[i], predictions[i]))


'''
Plot the loss curve
'''
def plot_loss(history):
  _, ax = plt.subplots()

  plt.plot(history.history['loss'])
  ax.set_xlabel('Epochs')
  ax.set_ylabel('Error')
  ax.set_title('Loss curve')

  plt.show()
  

'''
Entry point of our program.
'''
def main():
  # for re-producibility in the debugging stage with this, 
  # our generated "random" numbers will always be the same 
  # (easy to debug during development)
  np.random.seed(123)
  
  # prepare training data
  x_train, y_train = generate_data(-100, 100, 300, 3)

  # prepare testing data
  x_test, y_test = generate_data(-100, 100, 20, 3)

  # create the NN model
  model = create_model(n_features=x_train.shape[1])
  
  # train it
  history = train_model(model, x_train, y_train, epochs=100)

  # plot training loss curve
  plot_loss(history)

  # test it
  loss = test_model(model, x_test, y_test)
  print('Loss = {}\n'.format(loss))

  # eyeball it
  show_diffs(model, x_test, y_test)


# running via "python nn_add.py"
if __name__ == "__main__":
  main()