import cv2
import numpy as np
from tkinter import Tk, messagebox
import datetime

def save_image_if_color_dominant(cap, color_ranges, threshold=0.2):
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        return
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    total_pixels = frame.shape[0] * frame.shape[1]
    
    for color, (lower, upper) in color_ranges.items():
        mask = cv2.inRange(hsv, lower, upper)
        color_ratio = cv2.countNonZero(mask) / total_pixels
        
        if color_ratio > threshold:
            root = Tk()
            root.withdraw()  # 隐藏主窗口
            if messagebox.askyesno("Save Image", f"{color.capitalize()} color ratio exceeds 20%. Save the image?"):
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                cv2.imwrite(f"{timestamp}.png", frame)
                print(f"Image saved for {color}.")
            root.destroy()
            break  # 保存图片后退出循环
    
    cv2.imshow('Frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        return False
    return True

# 主程序
cap = cv2.VideoCapture(0)

# 定义多种颜色的HSV范围
color_ranges = {
    "white": (np.array([0, 0, 168]), np.array([172, 111, 255])),
    "red": (np.array([160, 100, 100]), np.array([180, 255, 255])),
    "blue": (np.array([94, 80, 2]), np.array([126, 255, 255])),
    "green": (np.array([35, 52, 72]), np.array([85, 255, 255])),
    "purple": (np.array([129, 50, 70]), np.array([158, 255, 255])),
    "yellow": (np.array([22, 93, 0]), np.array([45, 255, 255]))
}

try:
    while True:
        if not save_image_if_color_dominant(cap, color_ranges):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
