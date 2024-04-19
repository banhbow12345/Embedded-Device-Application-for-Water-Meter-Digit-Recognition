import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

def apply_filter(data, filter_size):
    a = []
    indexer = filter_size // 2
    data_final = []
    data_final = np.zeros((len(data),len(data[0])))
    for i in range(len(data)):

        for j in range(len(data[0])):

            for z in range(filter_size):
                if i + z - indexer < 0 or i + z - indexer > len(data) - 1:
                    for c in range(filter_size):
                        a.append(0)
                else:
                    if j + z - indexer < 0 or j + indexer > len(data[0]) - 1:
                        a.append(0)
                    else:
                        for k in range(filter_size):
                            a.append(data[i + z - indexer][j + k - indexer])

            a.sort()
            data_final[i][j] = a[len(a) // 2]
            a = []
    return data_final

def compute_hist(img):
    hist = np.zeros((256,), np.uint8)
    h, w = img.shape[:2]
    for i in range(h):
        for j in range(w):
            hist[int(img[i][j])] += 1
    return hist

def equal_hist(hist):
    cumulator = np.zeros_like(hist, np.float64)
    for i in range(len(cumulator)):
        cumulator[i] = hist[:i].sum()
    print(cumulator)
    new_hist = (cumulator - cumulator.min())/(cumulator.max() - cumulator.min()) * 255
    new_hist = np.uint8(new_hist)
    return new_hist

def main():
    img = cv2.imread('test.jpg', cv2.IMREAD_GRAYSCALE)
    plt.figure(figsize = (12, 12))
    plt.subplot(221)
    plt.imshow(img, cmap='gray')
    plt.title('Original')
    
    
    array = np.array(img)
    removed_noise = apply_filter(array, 3) 
    # out_img = Image.fromarray(removed_noise)
    plt.subplot(222)
    plt.imshow(removed_noise, cmap='gray')
    plt.title('Median')
    
    
    hist = compute_hist(removed_noise).ravel()
    new_hist = equal_hist(hist)
    plt.subplot(224)
    plt.plot(new_hist)

    h, w = removed_noise.shape[:2]
    for i in range(h):
        for j in range(w):
            removed_noise[i,j] = new_hist[int(removed_noise[i,j])]
    plt.subplot(223)
    plt.imshow(removed_noise, cmap='gray')
    plt.title('Histogram')
    
    # plt.xticks([]), plt.yticks([])
    plt.show()
    cv2.waitKey(0)
    cv2.destroyAllWindows()
main()

