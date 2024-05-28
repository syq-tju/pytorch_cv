#魔方红色识别
import cv2
import numpy as np
from tkinter import Tk, messagebox
import datetime

def save_image_if_color_dominant(cap, lower_color, upper_color, threshold=0.2):
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        return
    
    # 转换到HSV颜色空间
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # 创建掩码以检测特定颜色范围
    mask = cv2.inRange(hsv, lower_color, upper_color)
    
    # 计算颜色占比
    color_ratio = cv2.countNonZero(mask) / (frame.shape[0] * frame.shape[1])
    
    if color_ratio > threshold:
        # 颜色占比超过阈值
        root = Tk()
        root.withdraw()  # 隐藏主窗口
        if messagebox.askyesno("Save Image", "Color ratio exceeds 20%. Save the image?"):
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            cv2.imwrite(f"{timestamp}.png", frame)
            print("Image saved.")
        root.destroy()
    
    cv2.imshow('Frame', frame)
    cv2.imshow('Mask', mask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        return False
    return True

# 主程序
cap = cv2.VideoCapture(0)

# 定义想要检测的颜色范围，例如红色
lower_red = np.array([160, 100, 100])
upper_red = np.array([180, 255, 255])

try:
    while True:
        if not save_image_if_color_dominant(cap, lower_red, upper_red):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
