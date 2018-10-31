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
from keras.utils import np_utils
import numpy as np
import matplotlib.pyplot as plt

class LenetOverFashionMnist:
    def __init__(self, dropout, l2, bn, regularization, name):
        self.dropout_flag = dropout
        self.l2_flag = l2
        self.bn_flag = bn
        self.regularization_flag = regularization
        self.name = name


    def load_data(self):
        data = input_data.read_data_sets('data/fashion')

        train_data_raw, self.train_labels = mnist_reader.load_mnist('data/fashion', kind='train')
        test_data_raw, self.test_labels = mnist_reader.load_mnist('data/fashion', kind='t10k')

        train_data_raw = train_data_raw.reshape((train_data_raw.shape[0], int(np.sqrt(train_data_raw.shape[1])),
                                                 int(np.sqrt(train_data_raw.shape[1]))))
        test_data_raw = test_data_raw.reshape((test_data_raw.shape[0], int(np.sqrt(test_data_raw.shape[1])),
                                               int(np.sqrt(test_data_raw.shape[1]))))
        train_data_raw = train_data_raw[:, :, :, np.newaxis]
        test_data_raw = test_data_raw[:, :, :, np.newaxis]

        self.train_data = train_data_raw / 255.0
        self.test_data = test_data_raw / 255.0

        self.train_labels = np_utils.to_categorical(self.train_labels, 10)
        self.test_labels = np_utils.to_categorical(self.test_labels, 10)

        # self.train_data = self.train_data[0:100,:,:,:]
        # self.train_labels = self.train_labels[0:100, :]
        # self.test_data = self.test_data[0:100, :, :, :]
        # self.test_labels = self.test_labels[0:100, :]
    def build_model(self):
        Loss = "categorical_crossentropy"
        if self.l2_flag:
            Loss = "mean_squared_error"

        self.model = Sequential()


        # Add the first convolution layer
        self.model.add(Convolution2D(
            filters=32,
            kernel_size=(2, 2),
            padding="same",
            input_shape=(28, 28, 1)))

        if self.bn_flag:
            self.model.add(BatchNormalization())

        if self.dropout_flag:
            self.model.add(Dropout(rate=0.25))

        # # Add a ReLU activation function
        self.model.add(Activation(
            activation="relu"))

        # Add a pooling layer
        self.model.add(MaxPooling2D(
            pool_size=(2, 2),
            strides=(2, 2)))

        # Add the second convolution layer
        self.model.add(Convolution2D(
            filters=32,
            kernel_size=(2, 2),
            padding="same"))

        if self.bn_flag:
            self.model.add(BatchNormalization())

        # Add a ReLU activation function
        self.model.add(Activation(
            activation="relu"))

        # Add a second pooling layer
        self.model.add(MaxPooling2D(
            pool_size=(2, 2),
            strides=(2, 2)))

        # Flatten the network
        self.model.add(Flatten())

        # Add a fully-connected hidden layer
        if self.regularization_flag:
            self.model.add(Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
        else:
            self.model.add(Dense(128, activation='relu'))

        if self.bn_flag:
            self.model.add(BatchNormalization())

        # Add a fully-connected output layer
        self.model.add(Dense(10, activation='softmax'))

        if self.bn_flag:
            self.model.add(BatchNormalization())

        # Compile the network
        self.model.compile(
            loss=Loss,
            optimizer='adam',
            metrics=["accuracy"])

    def print_model(self):
        print(self.model.summary())

    def train_model(self):
        self.model.fit(
            self.train_data,
            self.train_labels,
            validation_data=(self.test_data, self.test_labels),
            batch_size=32,
            nb_epoch=20,
            verbose=1)

    def evaluate_model(self):
        (loss, accuracy) = self.model.evaluate(
            self.test_data,
            self.test_labels,
            batch_size=128,
            verbose=1)

        # Print the model's accuracy
        print(accuracy)

    def plot_graphs(self, i):
        plt.figure(i)
        plt.plot(self.model.history.history['acc'])
        plt.plot(self.model.history.history['val_acc'])
        plt.title('Model Accuracy ' + self.name)
        plt.ylabel('Accuracy')
        plt.xlabel('Number of epoch')
        plt.legend(['train', 'test'])
        plt.show()

basic_lenet = LenetOverFashionMnist(dropout=False,l2=False,bn=False,
                                    regularization=False,name='basic Lenet')
dropout_lenet = LenetOverFashionMnist(dropout=True,l2=False,bn=False,
                                    regularization=False,name='With Dropout')
l2_lenet = LenetOverFashionMnist(dropout=False,l2=True,bn=False,
                                    regularization=False,name='With L2 Loss')
bn_lenet = LenetOverFashionMnist(dropout=False,l2=False,bn=True,
                                    regularization=False,name='With Batch norm')
regularization_lenet = LenetOverFashionMnist(dropout=False,l2=False,bn=False,
                                    regularization=True,name='With Regularization')

lenet_models =[bn_lenet]# [basic_lenet,dropout_lenet,l2_lenet,
               # regularization_lenet,bn_lenet]

for model in lenet_models:
    model.load_data()
    model.build_model()
    model.print_model()
    model.train_model()
    model.evaluate_model()

i = 0
for model in lenet_models:
    i = i+1
    model.plot_graphs(i)
