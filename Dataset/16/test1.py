import cv2 as cv 
image = cv.imread('test.jpg')
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
blurred = cv.GaussianBlur(gray, (3, 3), 0)
sharp = cv.subtract(gray, blurred)
cv.imshow('lmao', sharp)
cv.waitKey(0)
cv.destroyAllWindows