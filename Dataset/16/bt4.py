import cv2
import numpy as np
import requests
from matplotlib import pyplot as plt

# Đọc hình ảnh
# url = 'https://imgur.com/7J223M7.png'
# url = 'http://final4.jpg'
# resp = requests.get(url)
resp = cv2.imread('final4.jpg')

# Đọc ảnh lấy từ internet
img = np.asarray(bytearray(resp), dtype="uint8") # chuyển các byte ảnh yêu cầu từ internet sẽ được chuyển thành mảng array
img = cv2.imdecode(img, cv2.IMREAD_COLOR)

# Lọc ảnh nhị phân bằng thuật toán canny
imgCanny = cv2.Canny(resp, 100, 200)
contours, hierarchy = cv2.findContours(imgCanny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Vì drawContours sẽ thay đổi ảnh gốc nên cần lưu ảnh sang một biến mới.
imgOrigin = resp.copy()
# img1 = resp.copy()
# img2 = resp.copy()

# Erosion
img1 = cv2.medianBlur(resp, 3) 
# cv2.imwrite(path, img) #lưu ảnh vào đường dẫn
# Dilation
img2 = cv2.convertScaleAbs(resp, 1.1, 5)

# Vẽ toàn bộ contours trên hình ảnh gốc
image1 = cv2.drawContours(img1, contours, -1, (0, 255, 0), 3)
# Vẽ chỉ contour thứ 4 trên hình ảnh gốc
image2 = cv2.drawContours(img2, contours, 100, (0, 255, 0), 3)

plt.figure(figsize = (12, 3))
plt.subplot(141),plt.imshow(imgOrigin),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(142),plt.imshow(imgCanny),plt.title('Canny Binary Image')
plt.xticks([]), plt.yticks([])
plt.subplot(143),plt.imshow(image1),plt.title('All Contours')
plt.xticks([]), plt.yticks([])
plt.subplot(144),plt.imshow(image2),plt.title('Contour 4')
plt.xticks([]), plt.yticks([])

plt.show()