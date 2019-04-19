'''Train a simple deep CNN on the CIFAR10 small images dataset.

It gets to 75% validation accuracy in 25 epochs, and 79% after 50 epochs.
(it's still underfitting at that point, though).
'''

#from __future__ import print_function
import sys

s = int(sys.argv[5])
from numpy.random import seed
seed(s)
from tensorflow import set_random_seed
set_random_seed(s+1)

import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, Callback, CSVLogger, TerminateOnNaN
from keras.optimizers import RMSprop, Adam, SGD
from hfcsam import HFCSAM

import numpy as np
import time
import pandas as pd
import os

#For a quick run go to ../HF_CSAM directory and run the following commands:
# python cnn/cifar10_cnn_test_optimizers.py HFCSAM 0.07 0.99 f 1
# python cnn/cifar10_cnn_test_optimizers.py SGD 0.01 0.7 f 1
# python cnn/cifar10_cnn_test_optimizers.py Adam 0.001 0 f 1
# python cnn/cifar10_cnn_test_optimizers.py RMSprop 0.001 0 f 1

class TimeHistory(Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)

batch_size = 128
num_classes = 10
epochs = 100
num_predictions = 20

optimizer = sys.argv[1]
lr = float(sys.argv[2])
mom = float(sys.argv[3])
aug = str(sys.argv[4])

if optimizer=="SGD":
    optimizer = optimizer+'('+str(lr)+', momentum='+str(mom)+')'
elif optimizer=="HFCSAM":
    optimizer = optimizer+'(dP='+str(lr)+', xi='+str(mom)+')'
else:
    optimizer = optimizer+'('+str(lr)+')'

if aug=="t":
    data_augmentation = True
elif aug=="f":
    data_augmentation = False

print "-------------------------------------------------------"
print "-------------------------------------------------------"

print "Running..."
print "Architecture : cnn"
print "Dataset : cifar"
print "Optimizer : ", optimizer
print "Learing Rate : ", lr
print "Momentum : ", mom
print "Data Augmentation : ", data_augmentation
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

#save_dir = os.path.join(os.getcwd(), 'saved_models')
#model_name = 'keras_cifar10_trained_model.h5'

# The data, split between train and test sets:
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print 'x_train shape:', x_train.shape
print x_train.shape[0], 'train samples'
print x_test.shape[0], 'test samples'

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=x_train.shape[1:],
		 bias_initializer='zeros'))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3), bias_initializer='zeros'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same', bias_initializer='zeros'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3), bias_initializer='zeros'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512, bias_initializer='zeros'))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, bias_initializer='zeros'))
model.add(Activation('softmax'))

opt = eval(optimizer)

# Let's train the model
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

call_nan = TerminateOnNaN()
early_stopping = EarlyStopping(monitor='val_acc', patience=10)
filename = 'results/cnn_cifar_'+str(optimizer)+'.csv'
csv_logger = CSVLogger(filename)
time_callback = TimeHistory()

if not data_augmentation:
    print 'Not using data augmentation.'
    history = model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test),
              shuffle=True,
              callbacks=[call_nan,early_stopping,csv_logger,time_callback])
else:
    print 'Using real-time data augmentation.'
    # This will do preprocessing and realtime data augmentation:
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        zca_epsilon=1e-06,  # epsilon for ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        shear_range=0.,  # set range for random shear
        zoom_range=0.,  # set range for random zoom
        channel_shift_range=0.,  # set range for random channel shifts
        fill_mode='nearest',  # set mode for filling points outside the input boundaries
        cval=0.,  # value used for fill_mode = "constant"
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False,  # randomly flip images
        rescale=None,  # set rescaling factor (applied before any other transformation)
        preprocessing_function=None,  # set function that will be applied on each input
        data_format=None,  # image data format, either "channels_first" or "channels_last"
        validation_split=0.0)  # fraction of images reserved for validation (strictly between 0 and 1)

    # Compute quantities required for feature-wise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(x_train)
    print 'x_train shape:', x_train.shape
    print x_train.shape[0], 'new train samples'
    print x_test.shape[0], 'test samples'

    # Fit the model on the batches generated by datagen.flow().
    history = model.fit_generator(datagen.flow(x_train, y_train,
                                     batch_size=batch_size),
                        epochs=epochs,
                        validation_data=(x_test, y_test),
                        workers=4,
			callbacks=[call_nan,early_stopping,csv_logger,time_callback])

# Save model and weights
#if not os.path.isdir(save_dir):
#    os.makedirs(save_dir)
#model_path = os.path.join(save_dir, model_name)
#model.save(model_path)
#print('Saved trained model at %s ' % model_path)

times = time_callback.times
av_time = np.mean(times, dtype = np.float64)

# Score trained model.
scores = model.evaluate(x_test, y_test, verbose=1)
print 'Test loss:', scores[0]
print 'Test accuracy:', scores[1]

#save_history(history, filename)

df = pd.read_csv(filename)

max_val_acc = df['val_acc'].max(axis=0)
max_val_acc_index = df['val_acc'].idxmax(axis=0) + 1

loss = df['loss'][max_val_acc_index-1]
acc = df['acc'][max_val_acc_index-1]
val_loss = df['val_loss'][max_val_acc_index-1]

if not data_augmentation:
    arc="cifar_no_aug"
else:
    arc="cifar_with_aug"

print "cnn", arc, sys.argv[1], lr, mom, loss, acc, val_loss, max_val_acc, max_val_acc_index, av_time
