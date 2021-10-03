#!/usr/bin/env python
# coding: utf-8

# In[1]:


from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt


# In[21]:


def rebuild_img(u, sigma, v, p):
    print(p)
    m = len(u)
    n = len(v)
    a = np.zeros((m, n))
    
    count = (int)(sum(sigma))
    curSum = 0
    k = 0
    while curSum <= count * p:
        uk = u[:, k].reshape(m, 1)
        vk = v[k].reshape(1, n)
        a += sigma[k] * np.dot(uk, vk)
        curSum += sigma[k]
        k += 1
    print('k:',k)
    a[a < 0] = 0
    a[a > 255] = 255
    #按照最近距離取整数
    return np.rint(a).astype("uint8")


# In[22]:


p = 0.7
folder = 'fruit70'


# In[23]:


count = 0

for filename in os.listdir("/Users/yishan/Downloads/DM/fruit-recognition/Apple"):
    if filename != ".DS_Store":
        count += 1
        print(count)

        img = Image.open("/Users/yishan/Downloads/DM/fruit-recognition/Apple/" + filename, 'r')
        a = np.array(img)

        u, sigma, v = np.linalg.svd(a[:, :, 0])
        R = rebuild_img(u, sigma, v, p)

        u, sigma, v = np.linalg.svd(a[:, :, 1])
        G = rebuild_img(u, sigma, v, p)

        u, sigma, v = np.linalg.svd(a[:, :, 2])
        B = rebuild_img(u, sigma, v, p)

        I = np.stack((R, G, B), 2)
#         轉換為灰階
#         grayscale_image = np.dot(I[...,:3], [0.2989, 0.5870, 0.1140])
#         plt.imshow(grayscale_image, cmap=plt.get_cmap('gray'))
#         plt.show()
        new_p = Image.fromarray(I)
        new_p = new_p.convert('RGB')
        if count < 487:
            new_p.save("/Users/yishan/Downloads/DM/" + folder + "/validation/Apple/Apple" + str(count) + ".jpg")
        else:
            new_p.save("/Users/yishan/Downloads/DM/" + folder + "/train/Apple/Apple" + str(count) + ".jpg")


# K 是奇異值個數
# k=64 | 0.9
# k=31 | 0.8
# k=17,18 | 0.7
# k=10 | 0.6

# In[24]:


count = 0

for filename in os.listdir("/Users/yishan/Downloads/DM/fruit-recognition/Banana"):
    if filename != ".DS_Store":
        count += 1
        print(count)

        img = Image.open("/Users/yishan/Downloads/DM/fruit-recognition/Banana/" + filename, 'r')
        a = np.array(img)

        u, sigma, v = np.linalg.svd(a[:, :, 0])
        R = rebuild_img(u, sigma, v, p)

        u, sigma, v = np.linalg.svd(a[:, :, 1])
        G = rebuild_img(u, sigma, v, p)

        u, sigma, v = np.linalg.svd(a[:, :, 2])
        B = rebuild_img(u, sigma, v, p)

        I = np.stack((R, G, B), 2)
#         轉換為灰階
#         grayscale_image = np.dot(I[...,:3], [0.2989, 0.5870, 0.1140])
#         new_p = Image.fromarray(grayscale_image)
#         new_p = Image.fromarray(I)
        new_p = new_p.convert('RGB')
        if count < 606:
            new_p.save("/Users/yishan/Downloads/DM/" + folder + "/validation/Banana/Banana" + str(count) + ".jpg")
        else:
            new_p.save("/Users/yishan/Downloads/DM/" + folder + "/train/Banana/Banana" + str(count) + ".jpg")


# In[25]:


count = 0

for filename in os.listdir("/Users/yishan/Downloads/DM/fruit-recognition/Pitaya"):
    if filename != ".DS_Store":
        count += 1
        print(count)

        img = Image.open("/Users/yishan/Downloads/DM/fruit-recognition/Pitaya/" + filename, 'r')
        a = np.array(img)

        u, sigma, v = np.linalg.svd(a[:, :, 0])
        R = rebuild_img(u, sigma, v, p)

        u, sigma, v = np.linalg.svd(a[:, :, 1])
        G = rebuild_img(u, sigma, v, p)

        u, sigma, v = np.linalg.svd(a[:, :, 2])
        B = rebuild_img(u, sigma, v, p)

        I = np.stack((R, G, B), 2)
#         轉換為灰階
#         grayscale_image = np.dot(I[...,:3], [0.2989, 0.5870, 0.1140])
#         new_p = Image.fromarray(grayscale_image)
#         new_p = Image.fromarray(I)
        new_p = new_p.convert('RGB')
        if count < 501:
            new_p.save("/Users/yishan/Downloads/DM/" + folder + "/validation/Pitaya/Pitaya" + str(count) + ".jpg")
        else:
            new_p.save("/Users/yishan/Downloads/DM/" + folder + "/train/Pitaya/Pitaya" + str(count) + ".jpg")

