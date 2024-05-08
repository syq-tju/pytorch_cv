import cv2
import numpy as np

img = cv2.imread("07.png",0)
template = cv2.imread("template.png",0)

h, w = template.shape  # Get the height and width of the template

methods = [cv2.TM_CCOEFF, cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR,
           cv2.TM_CCORR_NORMED, cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]

for method in methods:
    img2 = img.copy()   # Make a copy of the original image
   
    result = cv2.matchTemplate(img2, template, method)  # Perform template matching
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)  # Get the min and max value locations
    print(min_val, max_val, min_loc, max_loc)
   
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        location = min_loc
    else:
        location = max_loc
       
    bottom_right = (location[0] + w, location[1] + h)  # Calculate the bottom right corner of the rectangle
    cv2.rectangle(img2, location, bottom_right, 255, 2)  # Draw the rectangle on the image 

    cv2.imshow("match", img2)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
