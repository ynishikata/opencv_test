import numpy as np
import cv2
import math


from matplotlib import pyplot as plt 


# https://qiita.com/hmuronaka/items/008de58a8c21e8dd98e7

def contrast(image, a):
  lut = [ np.uint8(255.0 / (1 + math.exp(-a * (i - 128.) / 255.))) for i in range(256)]
  result_image = np.array( [ lut[value] for value in image.flat], dtype=np.uint8 )
  result_image = result_image.reshape(image.shape)
  return result_image

plt.rcParams['figure.figsize'] = (10, 10)

img=cv2.imread('./IMG_0150.JPG', cv2.IMREAD_GRAYSCALE)

img=contrast(img, 1)

neibourhood4=np.array([[0,1,0],[1,1,1],
        [0,1,0]], np.uint8)
neibourhood24=np.ones((5,5), dtype=np.uint8)
dilated=cv2.dilate(img, neibourhood24, iterations=1)

diff=cv2.absdiff(dilated, img)

contour = 255 - diff


plt.imshow(contour, cmap='gray')


plt.show()



