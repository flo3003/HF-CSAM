import sys

s = int(sys.argv[5]) 
from numpy.random import seed
seed(s)
from tensorflow import set_random_seed
set_random_seed(s+1)
########
#initialize biases zeros in dense

from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, Callback, CSVLogger, TerminateOnNaN
from keras.utils import np_utils
from keras.optimizers import RMSprop, Adam, SGD
from hfcsam import HFCSAM

import numpy as np
import time
import pandas as pd
import sys

class TimeHistory(Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)

#For a quick run go to ../HF_CSAM directory and run the following commands:
# python mlp/cifar10_mlp_test_optimizers.py HFCSAM 0.07 0.99 f 1
# python mlp/cifar10_mlp_test_optimizers.py SGD 0.01 0.7 f 1
# python mlp/cifar10_mlp_test_optimizers.py Adam 0.001 0 f 1
# python mlp/cifar10_mlp_test_optimizers.py RMSprop 0.001 0 f 1

optimizer = sys.argv[1]
lr = float(sys.argv[2])
mom = float(sys.argv[3])
aug = str(sys.argv[4])

epochs = 200
batch_size = 128
nb_classes = 10

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
print "Architecture : mlp"
print "Dataset : cifar"
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


if __name__ == '__main__':

    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255.0
    X_test /= 255.0

    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)

    model = Sequential()
    model.add(Flatten(input_shape=(32, 32, 3)))
    model.add(Dense(1024, bias_initializer='zeros'))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(512, bias_initializer='zeros'))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(512, bias_initializer='zeros'))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(10, bias_initializer='zeros'))
    model.add(Activation('softmax'))

    opt = eval(optimizer)

    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
    model.summary()

    call_nan = TerminateOnNaN()
    early_stopping = EarlyStopping(monitor='val_acc', patience=10)
    filename = 'results/mlp_cifar10_'+str(optimizer)+'.csv'
    csv_logger = CSVLogger(filename)
    time_callback = TimeHistory()


    # training
    if not data_augmentation:

	print 'Not using data augmentation.'
        history = model.fit(X_train, Y_train,
                            batch_size=batch_size,
                            epochs=epochs,
                            verbose=1,
                            validation_data=(X_test, Y_test),
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
        datagen.fit(X_train)

        # Fit the model on the batches generated by datagen.flow().
        history = model.fit_generator(datagen.flow(X_train, Y_train,
                            batch_size=batch_size),
                            epochs=epochs,
                            validation_data=(X_test, Y_test),
                            workers=4,
    			    callbacks=[call_nan,early_stopping,csv_logger,time_callback])

    #save_history(history, filename)
    times = time_callback.times
    av_time = np.mean(times, dtype = np.float64)

    loss, acc = model.evaluate(X_test, Y_test, verbose=0)

    df = pd.read_csv(filename)

    max_val_acc = df['val_acc'].max(axis=0)
    max_val_acc_index = df['val_acc'].idxmax(axis=0) + 1

    loss = df['loss'][max_val_acc_index-1]
    acc = df['acc'][max_val_acc_index-1]
    val_loss = df['val_loss'][max_val_acc_index-1]

    print('Test loss:', loss)
    print('Test acc:', acc)

    if not data_augmentation:
        arc="cifar_no_aug"
    else:
        arc="cifar_with_aug"

    print "mlp", arc, sys.argv[1], lr, mom, loss, acc, val_loss, max_val_acc, max_val_acc_index, av_time
