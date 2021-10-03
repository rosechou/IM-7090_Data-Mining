from keras.datasets import mnist
import numpy as np
from sklearn import cluster, metrics

# 讀入資料
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32')
x_train /= 255
X = x_train.reshape(len(x_train), -1)
X = X[:10000]

# Hierarchical Clustering 演算法
y_hc = cluster.AgglomerativeClustering(linkage = 'ward', affinity = 'euclidean', n_clusters = 12)

print("start training")
# 印出分群結果
y_hc.fit_predict(X)
cluster_labels = y_hc.labels_
print(cluster_labels.shape)
print(cluster_labels)
print("---")

# 印出label看看
print(y_train.shape)
y_train = np.transpose(y_train)
print(y_train)

from sklearn import metrics
print("homogeneity_score")
print(metrics.homogeneity_score(y_train[:10000], cluster_labels))
print("completeness_score")
print(metrics.completeness_score(y_train[:10000], cluster_labels))
print("v_measure_score")
print(metrics.v_measure_score(y_train[:10000], cluster_labels))
