import cv2
import numpy as np

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_RBUTTONDOWN:  # Right-click event
        screenshot_filename = "screenshot_04.png"  # Define the filename
        cv2.imwrite(screenshot_filename, frame)  # Save the current frame to file
        print(f"Screenshot saved as {screenshot_filename}")

cap = cv2.VideoCapture(0)  # Start video capture

cv2.namedWindow("Video with Drawings")
cv2.setMouseCallback("Video with Drawings", mouse_callback)  # Set mouse callback for the window

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
    
    # Calculate the center of the frame
    height, width = frame.shape[:2]
    center_x, center_y = width // 2, height // 2

    # Draw a rectangle
    cv2.rectangle(frame, (center_x - 150, center_y - 100), (center_x + 150, center_y + 100), (255, 0, 0), 3)

    # Draw a circle inside the rectangle
    cv2.circle(frame, (center_x, center_y), 50, (0, 255, 0), -1)

    # Draw a line crossing the circle
    cv2.line(frame, (center_x - 75, center_y), (center_x + 75, center_y), (0, 0, 255), 3)

    # Draw an ellipse
    cv2.ellipse(frame, (center_x, center_y), (100, 50), 0, 0, 180, (255, 255, 0), 2)

    cv2.imshow('Video with Drawings', frame)  # Display the frame with the drawings

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Break the loop if 'q' is pressed
        break

cap.release()  # Release the video capture object
cv2.destroyAllWindows()  # Close all OpenCV windows
