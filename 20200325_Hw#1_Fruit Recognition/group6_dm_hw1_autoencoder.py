#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import math
import os


# In[2]:


import tensorflow.keras.layers
import tensorflow.keras.models
import tensorflow.keras.optimizers  


# In[3]:


from keras.layers import Dense, Flatten, Reshape, Input, InputLayer
from keras.models import Sequential, Model


# In[4]:


x_train_orig = []
y_train = []
x_test_orig = []
y_test = []

x_apple = []
x_banana = []
x_pitaya = []
x = []
y = []
i = 0
for subdir, dirs, files in os.walk("fruit-recognition"):
    length = len(files)
    for file in files:
        #print os.path.join(subdir, file)
        filepath = subdir + os.sep + file
        if filepath.endswith(".png"):
            # 讀取圖檔
            img = cv2.imread(filepath)
            img2 = cv2.resize(img, (150, 150), interpolation=cv2.INTER_CUBIC)
            
            if "banana" in file or "Banana" in file:
                x_banana.append(img2)
                y.append("banana")
            elif "Pitaya" in file or "pitaya" in file:
                x_pitaya.append(img2)
                y.append("pitaya")
            else:
                x_apple.append(img2)
                y.append("apple")
            i += 1
            if i % 500 == 0:
                print("img: "+str(i)+"/"+str(length))


# In[5]:


x = x_apple + x_banana + x_pitaya
newX = np.array(x)
prop_apple = round(len(x_apple)*0.8)
prop_banana = round(len(x_banana)*0.8)
prop_pitaya = round(len(x_pitaya)*0.8)


# In[6]:


x_train_orig = np.array(x_apple[:prop_apple] + x_banana[:prop_banana] + x_pitaya[:prop_pitaya])
x_test_orig = np.array(x_apple[prop_apple:] + x_banana[prop_banana:] + x_pitaya[prop_pitaya:])


# In[7]:


print(len(x_train_orig))
print(len(x_test_orig))


# In[8]:


x_train_orig = x_train_orig.astype("float32") / 255.0
x_test_orig = x_test_orig.astype("float32") / 255.0


# In[9]:


#x_train = np.reshape(x_train_orig, newshape=(x_train_orig.shape[0], np.prod(x_train_orig.shape[1:])))
#x_test = np.reshape(x_test_orig, newshape=(x_test_orig.shape[0], np.prod(x_test_orig.shape[1:])))


# In[10]:


x_train = np.array(x_train_orig)
x_test = np.array(x_test_orig)


# In[11]:


def build_autoencoder(img_shape, code_size):
    # The encoder
    encoder = Sequential()
    encoder.add(InputLayer(img_shape))
    encoder.add(Flatten())
    encoder.add(Dense(code_size))

    # The decoder
    decoder = Sequential()
    decoder.add(InputLayer((code_size,)))
    decoder.add(Dense(np.prod(img_shape))) # np.prod(img_shape) is the same as 32*32*3, it's more generic than saying 3072
    decoder.add(Reshape(img_shape))

    return encoder, decoder


# In[12]:


# Same as (32,32,3), we neglect the number of instances from shape
IMG_SHAPE = newX.shape[1:]
print(IMG_SHAPE)
encoder, decoder = build_autoencoder(IMG_SHAPE, 32)

inp = Input(IMG_SHAPE)
code = encoder(inp)
reconstruction = decoder(code)

autoencoder = Model(inp,reconstruction)
autoencoder.compile(optimizer='adamax', loss='mse',metrics=['accuracy'])

print(autoencoder.summary())


# In[20]:


history = autoencoder.fit(x=x_train, y=x_train, epochs=1000,
                validation_data=[x_test, x_test])


# In[21]:


score = autoencoder.evaluate(x_test, x_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# In[22]:


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[23]:


def visualize(img,encoder,decoder):
    """Draws original, encoded and decoded images"""
    # img[None] will have shape of (1, 32, 32, 3) which is the same as the model input
    code = encoder.predict(img[None])[0]
    
    reco = decoder.predict(code[None])[0]

    plt.subplot(1,3,1)
    plt.title("Original")
    show_image(img)

    plt.subplot(1,3,2)
    plt.title("Code")
    plt.imshow(code.reshape([code.shape[-1]//2,-1]))

    plt.subplot(1,3,3)
    plt.title("Reconstructed")
    show_image(reco)
    plt.show()


# In[24]:


def show_image(x):
    plt.imshow(cv2.cvtColor(np.clip(x, 0, 1), cv2.COLOR_BGR2RGB))


# In[34]:


for i in range(10):
    img = x_test[i*90+2]
    visualize(img,encoder,decoder)


# In[26]:


deLen = len(x_test)
for i in range(deLen):
    img = x_test[i]
    code = encoder.predict(img[None])[0]
    reco = decoder.predict(code[None])[0]
    
    decoded_arr = (reco.astype("float32")*255).astype("int")
    cv2.imwrite("v13/"+str(i)+"_decoded.png", decoded_arr)
    
    ori_arr = (img.astype("float32")*255).astype("int")
    cv2.imwrite("v13/"+str(i)+"_original.png",ori_arr)
    
    if i % 100 == 0 and i > 0:
        print("saved: "+str(i)+"/"+str(deLen))


# In[ ]:




