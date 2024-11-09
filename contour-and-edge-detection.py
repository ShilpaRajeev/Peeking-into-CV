""" Contour detection is another important technique used in computer 
vision that can identify and extract the boundaries of objects in an image. 
Contours are curves joining all continuous points along the boundary 
having the same color or intensity."""

import cv2
import numpy as np
from google.colab.patches import cv2_imshow

try:
  # Load image
  image = cv2.imread('shapes.jpg')

  # Check if the image was successfully loaded
  if image is None:
    print("Error: Unable to load the image.")
    exit(1)
  
  # Convert ti grayscale
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

  # Find Canny Edges
  edged = cv2.Canny(gray, 30, 200)

  # Finding Contours
  contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

  # Display the Canny edges image
  cv2_imshow(edged)
  cv2.waitKey(0)

  print("Number of Contours found = " + str(len(contours)))

  # Draw all contours
  cv2.drawContours(image, contours, -1, (0,255, 0), 3)

  # Display the image with contours
  cv2_imshow(image)
  cv2.waitKey(0)

  # Close all OpenCV windows
  cv2.destroyAllWindows()

except Exception as e:
  print("An error occured:", e)
