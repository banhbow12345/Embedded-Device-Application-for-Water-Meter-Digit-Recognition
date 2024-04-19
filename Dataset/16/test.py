import cv2 as cv
import numpy as np
import imutils
from matplotlib import pyplot as plt
from flatten_json import flatten

img_path = 'C:/Users/ADMIN/Downloads/16/picture48.jpg'
img = cv.imread(img_path)
img_gray = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
new_width = 800
new_height = 600
img_resized = cv.resize(src=img_gray, dsize=(new_width, new_height))
cropped_image = img_resized[167:220, 200:453]
brightness = 10 
contrast = 2.3 
image2 = cv.addWeighted(cropped_image, contrast, np.zeros(cropped_image.shape, img_resized.dtype), 0, brightness)
cv.imshow('image_his3', image2)

kernel2 = np.array([[0, -1, 0],
                   [-1, 5,-1],
                   [0, -1, 0]])
image_sharp = cv.filter2D(src=image2, ddepth=-1, kernel=kernel2)
cv.imshow('image_his5', image_sharp)
cropped_image2 = image_sharp[1:51, 215:250]
# test = 'test.jpg' 
# cv.imwrite(test, cropped_image2)
cv.imshow('cropped_image2', cropped_image2)


cv.waitKey(0)
cv.destroyAllWindows()