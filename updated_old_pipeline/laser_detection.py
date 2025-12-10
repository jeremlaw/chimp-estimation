#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Feb 20 2024

@author: jbrandinger

NAME: laser_detection.py

PURPOSE: 
    - detect red laser points on image

HOW TO USE:
    - provide 'detect_laser_points' with image containing red lasers

NOTES:
    - coordinates of laser points are returned as a list of tuples
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import blob_log
from skimage.feature import blob_dog
from math import sqrt
    

def apply_mask(source, mask):
    """
    Safely apply a boolean mask to a 3-channel color image.
    source: (H, W, 3)
    mask: (H, W) or (H, W, 1)
    Returns a masked image of same shape.
    """
    if mask.ndim == 3:  # if mask has a singleton channel dimension
        mask = mask[:, :, 0]
    masked_img = np.zeros_like(source)
    masked_img[mask] = source[mask]
    return masked_img


def detect_red_laser_points(source, mask):
    img = apply_mask(source, mask)

    hsv_frame = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    low_red1 = np.array([0, 120, 70])
    high_red1 = np.array([15, 255, 255])
    low_red2 = np.array([165, 120, 70])
    high_red2 = np.array([180, 255, 255])

    red_mask1 = cv2.inRange(hsv_frame, low_red1, high_red1)
    red_mask2 = cv2.inRange(hsv_frame, low_red2, high_red2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)

    red = cv2.bitwise_and(img, img, mask=red_mask)
    r = red[:, :, 2]

    blur = cv2.GaussianBlur(r, (25, 25), 0)
    _, thresh = cv2.threshold(blur, 210, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh = cv2.dilate(thresh, None, iterations=2)

    blobs = blob_dog(thresh, max_sigma=20, threshold=0.30)
    blobs = sorted(blobs, key=lambda b: b[1])

    min_distance_px = 30
    max_distance_px = 600

    points = []
    for blob in blobs:
        y, x, r = blob
        if r > 1:
            if not points:
                points.append((int(x), int(y)))
            else:
                squared_distance = (x - points[-1][0])**2 + (y - points[-1][1])**2
                if min_distance_px**2 <= squared_distance <= max_distance_px**2:
                    points.append((int(x), int(y)))

    return points[:2]


def detect_green_laser_points(source, mask):
    img = apply_mask(source, mask)

    hsv_frame = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    low_green = np.array([45, 120, 70])
    high_green = np.array([75, 255, 255])

    green_mask = cv2.inRange(hsv_frame, low_green, high_green)
    green = cv2.bitwise_and(img, img, mask=green_mask)
    g = green[:, :, 1]

    blur = cv2.GaussianBlur(g, (15, 15), 0)
    _, thresh = cv2.threshold(blur, 210, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh = cv2.dilate(thresh, None, iterations=1)

    blobs = blob_log(thresh, max_sigma=20, num_sigma=5, threshold=0.05)
    blobs = sorted(blobs, key=lambda b: b[1])

    min_distance_px = 40
    points = []
    for blob in blobs:
        y, x, r = blob
        if r > 1:
            if not points or (x - points[-1][0])**2 + (y - points[-1][1])**2 >= min_distance_px**2:
                points.append((int(x), int(y)))

    return points[:2]
