import cv2
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

img = cv2.imread('images/dog.jpg', 1) # read the image in the rgb format
# img = cv2.imread('images/dog.jpg', 0) # read the image in the gray format
# img = cv2.imread('images/dog.jpg', -1) # read the original image
edges = cv2.Canny(img, 100, 200)

plt.subplot(121)
plt.imshow(img)
plt.title('original image')
plt.xticks([]) # do not show xticks
plt.yticks([]) # do not show yticks

plt.subplot(122)
plt.imshow(edges, cmap='gray')
plt.title('edge image')
plt.xticks([])
plt.yticks([])

plt.show()