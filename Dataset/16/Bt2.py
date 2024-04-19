import numpy as np
import cv2
from matplotlib import pyplot as plt
image = cv2.imread('final2.jpg')
image1 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

img = cv2.blur(image1,(3,3))
kernel = np.ones((9,9),np.float32)/60
img = cv2.filter2D(image1,-1,kernel)
plt.imshow(img, cmap='gray')
plt.title('filter')


lap = cv2.Laplacian(img, cv2.CV_64F, ksize=3)
lap = np.uint8(np.absolute(lap))
sobelX = cv2.Sobel(img, cv2.CV_64F, 1, 0)
sobelY = cv2.Sobel(img, cv2.CV_64F, 0, 1)
edges = cv2.Canny(img, 100, 200)

sobelX = np.uint8(np.absolute(sobelX))
sobelY = np.uint8(np.absolute(sobelY))

sobelCombined = cv2.bitwise_or(sobelX, sobelY)


plt.figure(figsize = (12, 12))
title = ['image', 'Laplacian', 'sobelX', 'sobelY', 'sobelCombined', 'Canny']
images = [image, lap, sobelX, sobelY, sobelCombined, edges]
for i in range(6):
    plt.subplot(2, 3, i+1), plt.imshow(images[i], 'gray')
    plt.title(title[i])
    plt.xticks([]), plt.yticks([])
    
plt.show()


