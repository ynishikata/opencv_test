import numpy as np
import cv2
import matplotlib.pyplot as plt

def makecontour(path):
    # カーネルを定義
    kernel = np.ones((5,5), np.uint8)
    kernel[0,0] = kernel[0,4] = kernel[4,0] = kernel[4,4] = 0

    # グレースケールで画像を読み込む.
    #gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    # いらすとやの画像はアルファチャンネルがあるのでこれをまず白にする
    # ImageMagickの convert -flatten x.png y.png に対応
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
#    if img.shape[2] == 4:
#        mask = img[:,:,3] == 0
#        img[mask] = [255] * 4
#        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ノイズ除去が必要なら
    # gray = cv2.fastNlMeansDenoising(gray, h=20)
    # 白い部分を膨張させる
    dilated = cv2.dilate(gray, kernel, iterations=1)
    # 差をとる
    diff = cv2.absdiff(dilated, gray) # 白黒反転して2値化
    _, contour = cv2.threshold(255 - diff, 240, 255, cv2.THRESH_BINARY)
    # あるいは
    #contour = cv2.adaptiveThreshold(255 - diff, 255,
    #        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    #        cv2.THRESH_BINARY, 7, 8)
    return contour

# plt.close("all")
# plt.figure(figsize=[8, 8])
plt.set_cmap("gray")
# plt.clf()
contour = makecontour("IMG_3147.PNG")
plt.imshow(contour)
cv2.imwrite("190831a.png", contour)
