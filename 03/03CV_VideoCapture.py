import cv2
import numpy as np

cap = cv2.VideoCapture(0)  # Start video capture

ret, frame = cap.read()
if not ret:
    print("Failed to grab frame")
    cap.release()
    cv2.destroyAllWindows()
    exit()  # Exit if no frame can be captured

# Determine the dimensions of the camera frame
height, width = frame.shape[:2]

# Create a new frame with the same dimensions to hold all transformations
output_frame = np.zeros((height, width, 3), dtype=np.uint8)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
    
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert color space from BGR to RGB
    frame_small = cv2.resize(frame, (width // 2, height // 2))  # Resize frame to half of the original dimensions
    frame_r90 = cv2.rotate(frame_small, cv2.ROTATE_90_CLOCKWISE)  # Rotate frame 90 degrees clockwise
    frame_r180 = cv2.rotate(frame_small, cv2.ROTATE_180)  # Rotate frame 180 degrees
    frame_r270 = cv2.rotate(frame_small, cv2.ROTATE_90_COUNTERCLOCKWISE)  # Rotate frame 270 degrees counterclockwise

    # Resize rotated frames to correctly fit into the quadrants
    # Note the swapped dimensions for rotated frames
    frame_r90_resized = cv2.resize(frame_r90, (width // 2, height // 2))  # Swap dimensions for rotated frame
    frame_r270_resized = cv2.resize(frame_r270, (width // 2, height // 2))  # Swap dimensions for rotated frame

    # Place each small frame into the respective quadrant
    output_frame[:height//2, :width//2] = frame_small
    output_frame[height//2:, :width//2] = frame_r90_resized
    output_frame[:height//2, width//2:] = frame_r180
    output_frame[height//2:, width//2:] = frame_r270_resized
    
    cv2.imshow('Video', output_frame)  # Display the composite frame

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Break the loop if 'q' is pressed
        break

cap.release()  # Release the video capture object
cv2.destroyAllWindows()  # Close all OpenCV windows
