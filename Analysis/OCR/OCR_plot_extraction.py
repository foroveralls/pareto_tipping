# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 22:38:52 2023

@author: everall
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pdf2image import convert_from_path

# Convert the PDF page to an image
pdf_path = "../Data/Lit/Xie et al_2011_Social consensus through the influence of committed minorities.pdf"
page_images = convert_from_path(pdf_path, dpi=300, first_page=3, last_page=3)
page_image = np.array(page_images[0])

#%%
# Convert to grayscale (if the image is in RGB format)
if len(page_image.shape) == 3:
    page_image = cv2.cvtColor(page_image, cv2.IMREAD_GRAYSCALE)

# Crop the region corresponding to Fig 1 (a)
fig_1a_coords = (200, 250, 1100, 1400)
fig_1a_image = page_image[fig_1a_coords[1]:fig_1a_coords[3], fig_1a_coords[0]:fig_1a_coords[2]]



plt.figure(figsize=(10, 10))
plt.imshow(fig_1a_image)
plt.axis('on')
plt.show()
#%%
# Apply Gaussian blur to the cropped image
fig_1a_blur = cv2.GaussianBlur(fig_1a_image, (5, 5), 0)

# Convert to grayscale and apply binary thresholding
_, binary_image = cv2.threshold(fig_1a_blur, 200, 255, cv2.THRESH_BINARY_INV)

# Find contours in the binarized image
contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Find the largest rectangle (by area)
largest_rectangle = None
largest_area = 0

for contour in contours:
    # Approximate the contour to a polygon
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    
    # Check if the polygon has four vertices (indicating a rectangle)
    if len(approx) == 4:
        area = cv2.contourArea(contour)
        if area > largest_area:
            largest_area = area
            largest_rectangle = approx

# Convert the binarized image to RGB for visualization
binary_rgb_previous = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2RGB)

# Draw the largest rectangle (from our previous detection) on the image
if largest_rectangle is not None:
    cv2.drawContours(binary_rgb_previous, [largest_rectangle], 0, (0, 255, 0), 2)
    
plt.figure(figsize=(10, 10))
plt.imshow(binary_rgb_previous)
plt.axis('on')
plt.show()
