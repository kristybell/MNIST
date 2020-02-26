#!/usr/bin/env python
# coding: utf-8

# ## Deep Neural Network for MNIST Classification

# We will apply all the knowledge from the lectures in this section to write a deep neural network. The problem we have chosen is referred to as the "Hello World" of deep learning because for most students it is the first deep learning algorithm they see.

# The dataset is called MNIST and refers to handwritten digit recognition. You can find more about it on Yann LeCun's website (Director of AI Research, Facebook). He is one of the pioneers of what we've been talking about and of more complex approaches that are widely used today, such as covolutional neural networks (CNNs).

# The dataset provides 70,000 images (28x28 pixels) of handwritten digits (1 digit per image).

# The goal is to write an algorithm that detects which digit is written. Since there are only 10 digits (0, 1,2,3,4,5,6,7,8,9), this is a classification problem with 10 classes.

# Our goal would be to build a neural network with 2 hidden layers.

# ### Import the Relevant Packages

# In[1]:


import numpy as np
import tensorflow as tf

import tensorflow_datasets as tfds


# ### Data

# In[2]:


# PREPROCESSING DATA
# 'tfds.load(name)' loads a dataset from TensorFlow datasets
# 'as_supervised' = True, loads the data in a 2-tuple structure [input,target]
# 'with_info' = True, provides a tuple containing info about version, features, # samples of the dataset
mnist_dataset, mnist_info = tfds.load(name='mnist', with_info=True, as_supervised=True)

# extract the train and test data
mnist_train, mnist_test = mnist_dataset['train'], mnist_dataset['test']

# mnsit dataset module does not have a validation dataset

# take 10% of the training dataset to serve as validation
num_validation_samples = 0.1 * mnist_info.splits['train'].num_examples
# not sure if the results will come out as an integer, thus must state this limit so it does
#'tf.cast(x,dtype)' converts a variable into a diven data tyoe
num_validation_samples = tf.cast(num_validation_samples, tf.int64)

# do the same for the test samples
num_test_samples = mnist_info.splits['test'].num_examples
num_test_samples = tf.cast(num_test_samples, tf.int64)

# normally, we'd like to scale our data in some way to make the results more numerically stable (e.g. inputs between 0 and 1)
def scale(image,label):
    image = tf.cast(image, tf.float32)  # to ensure it is a float
    image /= 255.   # the '.' at the end ensures it is a float
    return image, label

# 'dataset.map(*function*)' -> applies a custom transformation to a given dataset; it takes as input function which determines the transformation

scaled_train_and_validation_data = mnist_train.map(scale)

test_data = mnist_test.map(scale)

# Shuffling -> keeping the same information but in a different order
# - necessary so that the batching has a variety of random data, as the data typically is listed in ascending order of the targets

# in cases of dealing with enormous datasets, we can't shuffle all data at once
BUFFER_SIZE = 10000   # shuffle the dataset with 10,000 data at a time

# Note: if buffer_size = 1, no shuffling will actually happen
#       if buffer_size >= num_samples, shuffling will happen at once (uniformly)
#       if 1 < buffer_size < num_samples, we will be optimizing the computational power

shuffled_train_and_validation_data = scaled_train_and_validation_data.shuffle(BUFFER_SIZE)

# to create validation data extract the from the data the same amount as there are samples 
validation_data = shuffled_train_and_validation_data.take(num_validation_samples)

# to create train data by extracting all data except for the validation data
train_data = shuffled_train_and_validation_data.skip(num_validation_samples)

# set batch size; batch size=1 -> stochastic gradient descent (SGD); batchsize = # samples -> (single batch) GD
BATCH_SIZE = 100

# 'dataset.batch(batch_size)' -> a method that combines the consecutive elements of a dataset into batches
train_data = train_data.batch(BATCH_SIZE)

# since we are only forward propagating on the validation data and not backpropagating, it is not necessary to create a batchsize
# when batching we find the AVERAGE loss
# BUT, the model expects the validation and test data in batch form too
validation_data = validation_data.batch(num_validation_samples)
test_data = test_data.batch(num_test_samples)

# the MNIST data is iterable and in 2-tuple format (as_supervised = True)
# thus, must extract and convert the validation inputs and targets appropriately
# 'iter()' creates an object which can be iterated one element at a time (i.e. in a for loop or while loop)
# 'next()' loads the next element of an iterable object
validation_inputs, validation_targets = next(iter(validation_data))


# ## Model

# ### Outline the model

# In[3]:


# 784 inputs --> 784 input layers
# 10 digits --> 10 output layers
# 2 hidden layers

input_size = 784
output_size = 10
# the underlying assumption is that all hidden layers are of the same siz
hidden_layer_size = 150

# our data (from tfds) is such that each input is 28x28x1
# must flatten a tensor into a vector
# 'tf.keras.layers.Flatten(original shape)' transforms (flattens) a tensor into a vector
# 'tf.keras.layers.Dense(output size)' takes the inputs, provided to the model and calculates the dot product of the inputs and the weights and adds the bias; also where we can apply an activation function
# when creating a classifier, the activation function of the output layer ust transform the vlaues into probabilities
# this is done by using 'softmax'
model = tf.keras.Sequential([
                            tf.keras.layers.Flatten(input_shape=(28,28,1)),
                            tf.keras.layers.Dense(hidden_layer_size, activation='relu'),
                            tf.keras.layers.Dense(hidden_layer_size, activation='relu'),
                            tf.keras.layers.Dense(output_size, activation='softmax')
])


# ### Choose the optimizer and the loss function

# In[4]:


# one of the best choices we've got is the adaptive moment estimation (ADAM)
# these strings are NOT case sensitive
# TensorFlow employs (3) built-in variations of a cross entropy loss
#     1. binary_crossentropy
#     2. categorical_crossentropy -> expects that you've one-hot encoded the targets
#     3. sparse_categorical_crossentrophy -> applies one-hot encoding
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy']) 


# ### Training

# In[5]:


# fit the model we built and see it actually works

# how many epochs we wish to train for
NUM_EPOCHS = 5

# determine the number of validation steps (batch_size)
NUM_STEPS = num_validation_samples

# fit the model
model.fit(train_data, epochs=NUM_EPOCHS, validation_data=(validation_inputs, validation_targets), validation_steps=NUM_STEPS, verbose=2)

# WHAT HAPPENS INSIDE AN EPOCH
#   1. At the beginning, of each epoch, the training loss will be set to 0
#   2. The algorithm will iterate over a preset number of batches, all from train_data
#   3. The weights and biases will be updated as many times as there are batches
#   4. We will get a value for the loss function, indicating how the training is going
#   5. We will also see a training accuracy (thanks to 'verbose')
#   6. At the end of the epoch, the algorithm will forward proagate the whole validation set
#   *** When we reach the maximum number of epochs, the training will be over


# In[7]:


# Ouput when 'hidden_layer_size = 50':
# several lines of output are shown above
# loss decreases with each batch, but didn't change too much because after the first epoch, have already had 540 different weight and bias updates  
# the accuracy shows in what % of the cases our outputs were equal to the targets
# usually keep an eye on the validation loss (or set early stopping mechanisms) to determine whether the model is overfitting
# the 'val_accuracy' is the TRUE VALIDATION ACCURACY OF THE MODEL
# 97.3% accuracy is great, but let's see if we can do better

# Output when 'hidden_layer_size = 100':
# 97.6% accuracy; which is a 0.3% increase in accuracy from a 50 hidden layer size

# Can we do better than this?

# Output when 'hidden_layer_size = 150':
# 98.3% accuracy; which is a 0.3% increase from a hidden layer size of 50


# ### Test the Model

# In[8]:


# the final accuracy of the model comes from forward propagating the test dataset, NOT the validation
# the reason the accuracy of the validation data is not the final, is because we may have overfitted the model
# we train on the training data, and then validate on the validation data
# that's how we make sure our parameters of the weights and biases don't overfit
# once the first model is trained, we fiddle with the hyperparameters:
#  1. adjust width of hidden layers
#  2. adjust the depth of the learning rate
#  3. adjust the batch size
#  4. adjust the activation functions for each layer  etc.

# the test data set is our reality check that prevents us from overfitting the hyperparameters


# In[11]:


test_loss, test_accuracy = model.evaluate(test_data)


# In[13]:


print('Test Loss: {0:.2f}. Test Accuracy: {1:.2f}%'.format(test_loss, test_accuracy*100.))


# In[ ]:


# after we test the model. conceptually, we are no longer allowed to change it because
# the test data will no longer be a data set the model has never seen
# you would have feedback 
# main point of the test dataset is to simulate model deployment if we get 50 or 60% testing accuracy

