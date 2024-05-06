import cv2
img = cv2.imread('image.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)

img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

cv2.imwrite('image_rotated.png', img)
cv2.imshow('Rotated Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()