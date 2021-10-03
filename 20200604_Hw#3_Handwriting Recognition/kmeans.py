#!/usr/bin/env python
# coding: utf-8

# In[1]:


import keras
from keras.datasets import mnist


# In[2]:


from sklearn import metrics


# In[237]:


from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import KMeans


# In[4]:


import numpy as np
import tensorflow as tf
from sklearn.decomposition import PCA


# In[5]:


from sklearn.decomposition import NMF


# In[47]:


from sklearn.metrics.cluster import homogeneity_score


# In[264]:


mnist = tf.keras.datasets.mnist


# In[265]:


# read MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()


# In[266]:


x = np.concatenate((x_train,x_test))
y = np.concatenate((y_train,y_test))


# In[270]:


n_components = int(round(28*28*0.9))


# In[271]:


# preprocessing the images
# convert each image to 1 dimensional array
X = x.reshape(len(x),-1)
Y = y
# normalize the data to 0 - 1
X = X.astype(float) / 255.


# In[ ]:


# dimension reduction by PCA
pca = PCA(n_components) # n_components 為要降到的維度
pca.fit(X) 
X = pca.transform(X) # 使用 transform() 即可取得降維後的陣列


# In[ ]:


# #initialize and fit KMeans algorithm 
kmeans = KMeans(n_clusters=10, random_state=0)
kmeans.fit(X)


# In[ ]:


# calculate and print accuracy
print('Homogeneity Score: {}\n'.format(homogeneity_score(y, kmeans.labels_)))


# In[ ]:




