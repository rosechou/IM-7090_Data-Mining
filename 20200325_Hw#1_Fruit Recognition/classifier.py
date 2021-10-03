from __future__ import absolute_import, division, print_function, unicode_literals

import math

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rotation_range=20, # rotate the image 20 degrees
                                width_shift_range=0.10, # Shift the pic width by a max of 5%
                                height_shift_range=0.10, # Shift the pic height by a max of 5%
                                rescale=1/255, # Rescale the image by normalzing it.
                                shear_range=0.1, # Shear means cutting away part of the image (max 10%)
                                zoom_range=0.1, # Zoom in by 10% max
                                horizontal_flip=True, # Allo horizontal flipping
                                fill_mode='nearest' # Fill in missing pixels with the nearest filled value
)
test_datagen = ImageDataGenerator(rotation_range=20, # rotate the image 20 degrees
                                width_shift_range=0.10, # Shift the pic width by a max of 5%
                                height_shift_range=0.10, # Shift the pic height by a max of 5%
                                rescale=1/255, # Rescale the image by normalzing it.
                                shear_range=0.1, # Shear means cutting away part of the image (max 10%)
                                zoom_range=0.1, # Zoom in by 10% max
                                horizontal_flip=True, # Allo horizontal flipping
                                fill_mode='nearest' # Fill in missing pixels with the nearest filled value
                                )

# load and iterate training dataset
train_it = train_datagen.flow_from_directory('datasets/Original_gray/train', class_mode='categorical', batch_size=64)

# load and iterate test dataset
test_it = test_datagen.flow_from_directory('datasets/Original_gray/test', class_mode='categorical', batch_size=64)

# number_of_examples = len(test_it.filenames)
# number_of_generator_calls = math.ceil(number_of_examples / (1.0 * 64)) 
# 1.0 above is to skip integer division

# y_labels = []

# for i in range(0,int(number_of_generator_calls)):
#     y_labels.extend(np.array(test_it[i][1]))
#     # print(y_labels[i])

# y_true = np.argmax(y_labels, axis=1)
# print(len(y_true))
# print(y_true[:50])

model = keras.Sequential([
    # keras.layers.Flatten(),
    # keras.layers.Dense(32, activation='relu'),
    # keras.layers.Dense(32),
    # keras.layers.Dense(2, activation='softmax')
    keras.layers.Conv2D(16, 3, padding='same', activation='relu', input_shape=(256, 256 ,3)),
    keras.layers.MaxPooling2D(),
    keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
    keras.layers.MaxPooling2D(),
    keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
    keras.layers.MaxPooling2D(),
    keras.layers.Flatten(),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dense(3, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# model.fit(train_images, train_labels, epochs=10)

# fit model
model.fit_generator(train_it, steps_per_epoch=40, epochs=10)

model.save('fruit_orig_gray_TRUE.h5')

# evaluate model
loss, acc = model.evaluate_generator(test_it, verbose=1)
print('loss: '+str(loss)+' acc: '+str(acc))

# make a prediction
# yhat = model.predict_generator(test_it, steps=None)
# # print('predictions_b: ', yhat[0])
# # print('predictions_p:', yhat[-1])

# from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, confusion_matrix
# y_pred = np.argmax(yhat, axis=1)
# print(y_pred[:50])
# # y_true = np.argmax(y_labels, axis=1)
# print(classification_report(y_true, y_pred))

# test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

# print('\nTest accuracy:', test_acc)

# # model.save('my_model.h5')

# probability_model = tf.keras.Sequential([model, 
#                                          tf.keras.layers.Softmax()])

# predictions = probability_model.predict(test_images)

# print(predictions[0])
# print(test_labels[0])
