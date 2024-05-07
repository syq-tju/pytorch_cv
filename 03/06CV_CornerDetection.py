import cv2
import numpy as np

# Load the image
img = cv2.imread("0606.jpeg")
img = cv2.resize(img, (0, 0), fx=2, fy=2)  # Resize the image

# Crop the image 
img = img[150:img.shape[0] - 150, 150:img.shape[1] - 150]  # Crop the image to remove the white border

color_img = img.copy()  # Make a copy of the original image to keep it in color
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale for corner detection

# Detect corners
corners = cv2.goodFeaturesToTrack(gray_img, 100, 0.01, 58)   # Detect 100 corners with a minimum distance of 10 pixels and quality level of 0.01
corners = np.int0(corners)                                  # Convert the corners to integers

# Draw circles at each corner on the color image
for corner in corners:
    x, y = corner.ravel()
    cv2.circle(color_img, (x, y), 15, (255, 0, 0), -1)  # Draw blue circles for visibility

# Draw lines between each corner on the color image
for i in range(len(corners)):
    for j in range(i + 1, len(corners)):
        corner1 = tuple(corners[i][0])
        corner2 = tuple(corners[j][0])
        color = tuple(map(int, np.random.randint(0, 255, size=3)))  # Generate random colors
        cv2.line(color_img, corner1, corner2, color, 1)

# Display the original color image with colored lines and circles
cv2.imshow("Colorful Lines Image", color_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
