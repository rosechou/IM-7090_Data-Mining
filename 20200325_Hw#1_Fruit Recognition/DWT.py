import numpy as np
import matplotlib.pyplot as plt

import cv2

import pywt
import pywt.data

import os

my_data_dir = 'datasets/Original/test/Apple/'

for f in os.listdir(my_data_dir):
    img = cv2.imread(my_data_dir+f)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # img = np.float32(img)
    # img /= 255

    # coeffs2 = pywt.dwt2(img, 'bior1.3')
    # LL, (LH, HL, HH) = coeffs2
    # coeffs2_2 = pywt.dwt2(LL, 'bior1.3')
    # LL_2, (LH_2, HL_2, HH_2) = coeffs2_2
    # LL_2 *= 255
    # # LL *= 255
    cv2.imwrite('datasets/Original_gray/test/Apple/'+f, img)
    # cv2.imshow('yee',LL_2)
    # cv2.waitKey(0)
    print(f)

print('finish')



'''
image = cv2.imread('datasets/Banana/Banana01.png')
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Convert to float for more resolution for use with pywt
image = np.float32(image)
image /= 255

# Load image
# original = pywt.data.camera()
# Wavelet transform of image, and plot approximation and details
titles = ['Approximation', ' Horizontal detail',
          'Vertical detail', 'Diagonal detail']
coeffs2 = pywt.dwt2(image, 'bior1.3')
LL, (LH, HL, HH) = coeffs2
fig = plt.figure(figsize=(12, 20))

cv2.imshow('orig', image)
cv2.imshow('output',LL)
cv2.waitKey(0)
# for i, a in enumerate([LL, LH, HL, HH]):
#     ax = fig.add_subplot(1, 4, i + 1)
#     ax.imshow(a, interpolation="nearest", cmap=plt.cm.gray)
#     ax.set_title(titles[i], fontsize=10)
#     ax.set_xticks([])
#     ax.set_yticks([])

fig.tight_layout()
# plt.show()
'''

####### convert all image data ########
