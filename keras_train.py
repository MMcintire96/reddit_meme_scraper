from __future__ import print_function

import keras
import numpy as np
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential

import data_prepare

img_row = 299
img_col = 299
img_channels = 3
classes = 2
batch_size = 2
epochs = 1

(train_data, train_labels), (test_data, test_labels) = data_prepare.return_data(new_data=False)

train_data = train_data[:10]
train_labels = train_labels[:10]
test_data = test_data[:10]
test_labels = test_labels[:10]

#set shape
train_data = train_data.reshape(train_data.shape[0], img_row, img_col, img_channels)
test_data = test_data.reshape(test_data.shape[0], img_row, img_col, img_channels)
input_shape = (img_row, img_col, img_channels)

#normalize data
train_data, test_data = train_data/255, test_data/255
train_labels = train_labels.astype('float32')
test_labels = test_labels.astype('float32')
#model
model = Sequential()
model.add(Conv2D(32, (3,3),
        input_shape=input_shape,
        activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Flatten())

model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.2))

#softmax
model.add(Dense(classes, activation='softmax', name='predictions'))

#compile model
model.compile(loss=keras.losses.categorical_crossentropy,
        optimizer=keras.optimizers.Adam(),
        metrics=['accuracy'])

#callbacks
callbacks = keras.callbacks.TensorBoard(log_dir='model/keras_model/logs',
        histogram_freq=2, write_graph=True, write_images=False)

#start train
model.fit(train_data, train_labels,
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        callbacks=[callbacks],
        validation_data=(test_data, test_labels))
score = model.evaluate(test_data, test_labels, verbose=0)
model.summary()
keras.models.save_model(model, 'model/keras_model/saves/cnn_1.h5')

print('Test loss: ', score[0])
print('Test Accuracy: ', score[1])
