import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import blob_log
from skimage.feature import blob_dog
from math import sqrt
    

def detect_red_laser_points(source, mask):
    # Apply the mask
    img = np.zeros_like(source)
    img[mask] = source[mask]

    # Assuming 'img' is your input image
    hsv_frame = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) #convert from BGR to HSV

    # Define two ranges for red (due to the circular nature of the hue scale in HSV)
    low_red1 = np.array([0, 120, 70])
    high_red1 = np.array([15, 255, 255])
    low_red2 = np.array([165, 120, 70])
    high_red2 = np.array([180, 255, 255])

    # Create masks for the red ranges
    red_mask1 = cv2.inRange(hsv_frame, low_red1, high_red1)
    red_mask2 = cv2.inRange(hsv_frame, low_red2, high_red2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)

    # Apply mask to the image
    red = cv2.bitwise_and(img, img, mask=red_mask)

    # Extract red channel for processing (assuming red dots are the brightest in the red channel)
    r = red[:,:,2]

    # Apply Gaussian blur for smoothing
    blur = cv2.GaussianBlur(r, (25, 25), 0)

    # Apply thresholding
    _, thresh = cv2.threshold(blur, 210, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Dilation to enlarge dots (tune parameters as needed)
    thresh = cv2.dilate(thresh, None, iterations=2)


    # Blob detection
    blobs = blob_dog(thresh, max_sigma=20, threshold=0.30)
    blobs = sorted(blobs, key=lambda b: b[1])

    min_distance_px = 30  # minimum number of pixels apart
    max_distance_px = 600  # maximum number of pixels apart

    points = []
    for blob in blobs:
        y, x, r = blob
        if r > 1:  # Check for area of blob
            if not points:
                # If no points have been added, add the first valid blob as a point.
                points.append((int(x), int(y)))
            else:
                # Calculate squared distance from the last added point
                squared_distance = (x - points[-1][0])**2 + (y - points[-1][1])**2
                # Check if the distance is within the allowed range
                if min_distance_px**2 <= squared_distance <= max_distance_px**2:
                    points.append((int(x), int(y)))

    return points[:2]  # Return only the first two points


def detect_green_laser_points(source, mask):
    # Apply the mask
    img = np.zeros_like(source)
    img[mask] = source[mask]

    # Convert from BGR to HSV
    hsv_frame = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define range for green
    low_green = np.array([32, 80, 20])
    high_green = np.array([75, 255, 255])

    # Create mask for the green range
    green_mask = cv2.inRange(hsv_frame, low_green, high_green)

    # Apply mask to the image
    green = cv2.bitwise_and(img, img, mask=green_mask)

    # Extract green channel for processing (assuming green dots are the brightest in the green channel)
    g = green[:,:,1]

    # Apply Gaussian blur for smoothing
    blur = cv2.GaussianBlur(g, (15, 15), 0)

    # Apply thresholding
    _, thresh = cv2.threshold(blur, 210, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Dilation to enlarge dots (tune parameters as needed)
    thresh = cv2.dilate(thresh, None, iterations=1)

    # Blob detection
    blobs = blob_log(thresh, max_sigma=20, num_sigma=5, threshold=.05)
    blobs = sorted(blobs, key=lambda b: b[1])

    min_distance_px = 40  # minimum number of pixels apart
    points = []
    for blob in blobs:
        y, x, r = blob
        if r > 1:  # Check for area of blob
            # Check if this blob is far enough from the last one
            if not points or (x - points[-1][0])**2 + (y - points[-1][1])**2 >= min_distance_px**2:
                points.append((int(x), int(y)))

    return points[:2]  # Return only the first two points