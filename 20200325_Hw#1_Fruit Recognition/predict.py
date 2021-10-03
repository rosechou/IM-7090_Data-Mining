# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.preprocessing.image import ImageDataGenerator

import numpy as np
import math

# from keras.models import load_model
model = keras.models.load_model('fruit_dwt1.h5')


test_datagen = ImageDataGenerator(rotation_range=20, # rotate the image 20 degrees
                                width_shift_range=0.10, # Shift the pic width by a max of 5%
                                height_shift_range=0.10, # Shift the pic height by a max of 5%
                                rescale=1/255, # Rescale the image by normalzing it.
                                shear_range=0.1, # Shear means cutting away part of the image (max 10%)
                                zoom_range=0.1, # Zoom in by 10% max
                                horizontal_flip=True, # Allo horizontal flipping
                                fill_mode='nearest' # Fill in missing pixels with the nearest filled value
                                )

# load and iterate test dataset
test_it = test_datagen.flow_from_directory('datasets/DWT_1/test', class_mode='categorical', batch_size=64)

number_of_examples = len(test_it.filenames)
number_of_generator_calls = math.ceil(number_of_examples / (1.0 * 64)) 
# 1.0 above is to skip integer division

y_labels = []

print('y_labels')
for i in range(0,int(number_of_generator_calls)):
    y_labels.extend(np.array(test_it[i][1]))
    # print(y_labels[i])
print(y_labels)

print('argmax y_true')
y_true = np.argmax(y_labels, axis=1)
print(len(y_true))
print(y_true[:50])

# make a prediction
print('predicting')
yhat = model.predict_generator(test_it, steps=None)
# print('predictions_b: ', yhat[0])
# print('predictions_p:', yhat[-1])
print(yhat)

from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, confusion_matrix
y_pred = np.argmax(yhat, axis=1)
print(y_pred)
# print(precision_score(test_it, y_pred_bool , average="macro"))
# print(recall_score(test_it, y_pred_bool , average="macro"))
# print(f1_score(test_it, y_pred_bool , average="macro"))
print(classification_report(y_true, y_pred))
