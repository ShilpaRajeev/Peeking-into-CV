"""
 The Canny edge detector is an edge detection operator that uses a multi-stage algorithm to detect a wide range of edges in images.
 It was developed by John F. Canny in 1986. Canny also produced a computational theory of edge detection explaining why the technique works.

 https://medium.com/@rohit-krishna/coding-canny-edge-detection-algorithm-from-scratch-in-python-232e1fdceac7
 https://medium.com/@cloudoers/a-computer-vision-project-1-using-canny-edge-detection-algorithm-to-showcase-the-contours-of-41c60ae3ffee

 """

# edge detection is a technique used to identify the edges found in an image. We can define an edge to be a change in pixel or image intensity.

"""
One of the most widely used ones is known as Canny edge detection, which is a multi-step edge detection technique that can detect edges by following these steps:

1. Gaussian smoothing
2. Calculating the gradient intensity
3. Non-maximum suppression
4. Double thresholding
5. Edge tracking

"""
# Note: Other edge detection algorithms include Sobel, Fuzzy logic, etc.

import cv2
import matplotlib.pyplot as plt

img = cv2.imread('image.jpg')

edges = cv2.Canny(img, 100, 200, 3, L2gradient = True)

plt.figure()
plt.title('Image')
plt.imsave('image_canny.png', edges, cmap='gray', format='png')
plt.imshow(edges, cmap='gray')
plt.show()