import cv2
import matplotlib.pyplot as plt

img = cv2.imread('image.jpg')

if img is None:
    print("Image not found or unable to read")
else:
    # 裁剪图像
    crop_img = img[500:1500, 500:1500]

    # 显示原始图像
    plt.subplot(2, 2, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")

    # 显示裁剪后的图像
    plt.subplot(2, 2, 2)
    plt.imshow(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))
    plt.title("Cropped Image")

    # 转换成灰度图像
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    plt.subplot(2, 2, 3)
    plt.imshow(gray_img, cmap='gray')
    plt.title("Gray Image")
    
    #降低像素值
    lower_reso = cv2.resize(img,(0,0), fx=0.1, fy=0.1)
    plt.subplot(2, 2, 4)
    plt.imshow(cv2.cvtColor(lower_reso, cv2.COLOR_BGR2RGB))
    plt.title("Lower Resolution Image")
        
    # 显示所有图像
    plt.show()
