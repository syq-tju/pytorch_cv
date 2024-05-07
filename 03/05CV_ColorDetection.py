import cv2
import numpy as np

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_RBUTTONDOWN:  # Right-click event
        screenshot_filename = "screenshot_05.png"  # Define the filename
        cv2.imwrite(screenshot_filename, frame)  # Save the current frame to file
        print(f"Screenshot saved as {screenshot_filename}")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open video capture.")
    exit()

cv2.namedWindow("Video with Color Detection")
cv2.setMouseCallback("Video with Color Detection", mouse_callback)  # Set mouse callback for the window

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Convert frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define range for yellow color and create mask
    lower_yellow = np.array([20, 100, 100])
    higher_yellow = np.array([30, 255, 255])
    mask = cv2.inRange(hsv, lower_yellow, higher_yellow)

    # Obtain the final result by applying the mask
    result = cv2.bitwise_and(frame, frame, mask=mask)

    # Display the result in the window
    cv2.imshow('Video with Color Detection', result)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Corrected for bitwise AND with 0xFF
        break

cap.release()
cv2.destroyAllWindows()
