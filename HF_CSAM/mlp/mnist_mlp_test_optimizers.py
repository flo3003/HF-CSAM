'''Trains a simple deep NN on the MNIST dataset.

Gets to 98.40% test accuracy after 20 epochs
(there is *a lot* of margin for parameter tuning).
2 seconds per epoch on a K520 GPU.
'''
#from __future__ import print_function
import sys
s = int(sys.argv[4])
from numpy.random import seed
seed(s)
from tensorflow import set_random_seed
set_random_seed(s+1)

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, Callback, CSVLogger, TerminateOnNaN
from keras.optimizers import RMSprop, Adam, SGD
from hfcsam import HFCSAM

import numpy as np
import time
import pandas as pd

#For a quick run go to ../HF_CSAM directory and run the following commands:
# python mlp/mnist_mlp_test_optimizers.py HFCSAM 0.07 0.99 1
# python mlp/mnist_mlp_test_optimizers.py SGD 0.01 0.7 1
# python mlp/mnist_mlp_test_optimizers.py Adam 0.001 0 1
# python mlp/mnist_mlp_test_optimizers.py RMSprop 0.001 0 1

class TimeHistory(Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)

batch_size = 128
num_classes = 10
epochs = 1000

optimizer = sys.argv[1]
lr = float(sys.argv[2])
mom = float(sys.argv[3])

if optimizer=="SGD":
    optimizer = optimizer+'('+str(lr)+', momentum='+str(mom)+')'
elif optimizer=="HFCSAM":
    optimizer = optimizer+'(dP='+str(lr)+', xi='+str(mom)+')'
else:
    optimizer = optimizer+'('+str(lr)+')'

print "-------------------------------------------------------"
print "-------------------------------------------------------"

print "Running..."
print "Architecture : mlp"
print "Dataset : mnist"
print "Optimizer : ", optimizer
print "Learing Rate : ", lr
print "Momentum : ", mom
print "Batch Size : ", batch_size
print "Number of Epochs : ", epochs

print "-------------------------------------------------------"
print "-------------------------------------------------------"

def save_history(history, result_file):
    loss = history.history['loss']
    acc = history.history['acc']
    val_loss = history.history['val_loss']
    val_acc = history.history['val_acc']
    epochs = len(acc)

    with open(result_file, "w") as fp:
        fp.write("epoch,loss,acc,val_loss,val_acc\n")
        for i in range(epochs):
            fp.write("%d,%f,%f,%f,%f\n" %
                     (i, loss[i], acc[i], val_loss[i], val_acc[i]))

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print x_train.shape[0], 'train samples'
print x_test.shape[0], 'test samples'

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Dense(512, bias_initializer='zeros', activation='relu', input_shape=(784,)))
model.add(Dropout(0.2))
model.add(Dense(512, bias_initializer='zeros', activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(num_classes, bias_initializer='zeros', activation='softmax'))

model.summary()

opt = eval(optimizer)

model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

call_nan = TerminateOnNaN()
early_stopping = EarlyStopping(monitor='val_acc', patience=10)
filename = 'results/mlp_mnist_'+str(optimizer)+'.csv'
csv_logger = CSVLogger(filename)
time_callback = TimeHistory()

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test),
		    callbacks=[call_nan,early_stopping, time_callback, csv_logger])

times = time_callback.times
av_time = np.mean(times, dtype = np.float64)

score = model.evaluate(x_test, y_test, verbose=0)
print 'Test loss:', score[0]
print 'Test accuracy:', score[1]

#save_history(history, filename)

df = pd.read_csv(filename)

max_val_acc = df['val_acc'].max(axis=0)
max_val_acc_index = df['val_acc'].idxmax(axis=0) + 1

loss = df['loss'][max_val_acc_index-1]
acc = df['acc'][max_val_acc_index-1]
val_loss = df['val_loss'][max_val_acc_index-1]

print "mlp", "mnist", sys.argv[1], lr, mom, loss, acc, val_loss, max_val_acc, max_val_acc_index, av_time
