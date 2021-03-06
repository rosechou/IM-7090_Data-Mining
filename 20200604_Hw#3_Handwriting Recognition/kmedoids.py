# -*- coding: utf-8 -*-
"""KMedoids.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1QqG-cOIHMzpjoTBzsyzOKXbygydIpGSc
"""

pip install scikit-learn-extra

pip install pyclustering

import pandas as pd
import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from sklearn_extra.cluster import KMedoids
from sklearn.metrics.pairwise import pairwise_distances
from pyclustering.cluster.kmedoids import kmedoids

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

n_components = 20



print(x_train)

print(x_train[2][200])

data = np.asarray(x_train)

from sklearn.decomposition import NMF
from sklearn import decomposition

estimator = decomposition.NMF(n_components = n_components, init = 'random', tol=5e-3)    
W = estimator.fit_transform(data[0:10000])
H = estimator.components_

data = np.dot(W,H)

print(data)

#KMed=KMedoids(n_clusters=2,init='random',max_iter=1)

KMed=KMedoids(n_clusters=10,init='random',random_state=5)

KMed.fit(data[0:10000])

KMed.labels_

def infer_cluster_labels(kmedoids, actual_labels):
    """
    Associates most probable label with each cluster in KMeans model
    returns: dictionary of clusters assigned to each label
    """

    inferred_labels = {}

    for i in range(KMed.n_clusters):

        # find index of points in cluster
        labels = []
        index = np.where(KMed.labels_ == i)

        # append actual labels for each point in cluster
        labels.append(actual_labels[index])

        # determine most common label
        if len(labels[0]) == 1:
            counts = np.bincount(labels[0])
        else:
            counts = np.bincount(np.squeeze(labels))

        # assign the cluster to a value in the inferred_labels dictionary
        if np.argmax(counts) in inferred_labels:
            # append the new number to the existing array at this slot
            inferred_labels[np.argmax(counts)].append(i)
        else:
            # create a new array in this slot
            inferred_labels[np.argmax(counts)] = [i]

        #print(labels)
        #print('Cluster: {}, label: {}'.format(i, np.argmax(counts)))
        
    return inferred_labels

def infer_data_labels(X_labels, cluster_labels):
    """
    Determines label for each array, depending on the cluster it has been assigned to.
    returns: predicted labels for each array
    """
    
    # empty array of len(X)
    predicted_labels = np.zeros(len(X_labels)).astype(np.uint8)
    
    for i, cluster in enumerate(X_labels):
        for key, value in cluster_labels.items():
            if cluster in value:
                predicted_labels[i] = key
                
    return predicted_labels

cluster_labels = infer_cluster_labels(KMed, y_train)

test_clusters = KMed.predict(x_test)

print(cluster_labels)

print(test_clusters)

predicted_labels = infer_data_labels(test_clusters, cluster_labels)

from sklearn import metrics

print('Accuracy: {}\n'.format(metrics.accuracy_score(y_test, predicted_labels)))

print('Homogeneity: {}\n'.format(metrics.homogeneity_score(y_test, predicted_labels)))