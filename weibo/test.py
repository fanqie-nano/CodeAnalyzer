import cv2
import numpy as np
import matplotlib.pyplot as plt
import pylab
from PIL import Image


img = Image.open('test1.png')
new_img = img.crop((402, 39, 411, 47))
new_img.save('1111.png')
print np.array(new_img)

# def inverse_color(image):
#     height,width = image.shape
#     img2 = image.copy()

#     for i in range(height):
#         for j in range(width):
#             img2[i,j] = (255-image[i,j]) 
#     return img2

# data = Image.open('1.png')
# data = data.convert('L')
# data = data.resize((data.size[0] * 10, data.size[1] * 10))
# imgTmp = np.array(data)
# imgTmp = inverse_color(imgTmp)
# img = cv2.threshold(imgTmp, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
# # img = cv2.imread('man.png',0) #直接读为灰度图像

# kernel = np.ones((10,10),np.uint8)
# erosion = cv2.erode(img,kernel,1)
# erosion = cv2.dilate(img,kernel,1)
# erosion = cv2.dilate(img,kernel,1)
# plt.subplot(1,3,1),plt.imshow(imgTmp,'gray')#默认彩色，另一种彩色bgr
# plt.subplot(1,3,2),plt.imshow(img,'gray')
# plt.subplot(1,3,3),plt.imshow(erosion,'gray')
# pylab.show()