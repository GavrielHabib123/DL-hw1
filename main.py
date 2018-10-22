from utils import mnist_reader
from tensorflow.examples.tutorials.mnist import input_data
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Dropout
from keras.layers.normalization import BatchNormalization
from keras import regularizers
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.models import Sequential
from keras.optimizers import SGD
from keras.utils import np_utils
import numpy as np

dropout_flag = False
l2_flag = False
bn_flag = False
regularization_flag = False

Loss = "categorical_crossentropy"
if l2_flag:
    Loss = "mean_squared_error"

data = input_data.read_data_sets('data/fashion')

train_data_raw, train_labels = mnist_reader.load_mnist('data/fashion', kind='train')
test_data_raw, test_labels = mnist_reader.load_mnist('data/fashion', kind='t10k')

#################################

train_data_raw = train_data_raw.reshape((train_data_raw.shape[0],int(np.sqrt(train_data_raw.shape[1])),
                                 int(np.sqrt(train_data_raw.shape[1]))))
test_data_raw = test_data_raw.reshape((test_data_raw.shape[0],int(np.sqrt(test_data_raw.shape[1])),
                                 int(np.sqrt(test_data_raw.shape[1]))))
train_data_raw = train_data_raw[:,:,:,np.newaxis]
test_data_raw = test_data_raw[:,:,:,np.newaxis]

train_data = train_data_raw / 255.0
test_data = test_data_raw / 255.0

train_labels = np_utils.to_categorical(train_labels, 10)
test_labels = np_utils.to_categorical(test_labels, 10)

model = Sequential()

# Add the first convolution layer
model.add(Convolution2D(
    filters = 20,
    kernel_size = (5, 5),
    padding = "same",
    input_shape = (28, 28, 1)))

if dropout_flag:
    model.add(Dropout(rate=0.25))

if bn_flag:
    model.add(BatchNormalization())

# Add a ReLU activation function
model.add(Activation(
    activation = "relu"))

# Add a pooling layer
model.add(MaxPooling2D(
    pool_size = (2, 2),
    strides = (2, 2)))

# Add the second convolution layer
model.add(Convolution2D(
    filters = 50,
    kernel_size = (5, 5),
    padding = "same"))

if bn_flag:
    model.add(BatchNormalization())

# Add a ReLU activation function
model.add(Activation(
    activation = "relu"))

# Add a second pooling layer
model.add(MaxPooling2D(
    pool_size = (2, 2),
    strides = (2, 2)))

# Flatten the network
model.add(Flatten())

# Add a fully-connected hidden layer
if regularization_flag:
    model.add(Dense(500,kernel_regularizer=regularizers.l2(0.01)))
else:
    model.add(Dense(500))

if bn_flag:
    model.add(BatchNormalization())

# Add a ReLU activation function
model.add(Activation(
    activation = "relu"))

# Add a fully-connected output layer
model.add(Dense(10))

if bn_flag:
    model.add(BatchNormalization())

# Add a softmax activation function
model.add(Activation("softmax"))

# Compile the network
model.compile(
    loss = Loss,
    optimizer = SGD(lr = 0.01),
    metrics = ["accuracy"])

# Train the model
model.fit(
    train_data,
    train_labels,
    batch_size = 128,
    nb_epoch = 20,
	  verbose = 1) # model.history

# Evaluate the model
(loss, accuracy) = model.evaluate(
    test_data,
    test_labels,
    batch_size = 128,
    verbose = 1)

# Print the model's accuracy
print(accuracy)