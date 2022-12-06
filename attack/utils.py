import numpy as np
import cv2
import math


def create_diamond_mask(mask_shape, y, x, s):
    mask = np.zeros(mask_shape)
    d = int(s*math.sqrt(2)/2)
    points = [np.array([[x-d, y], [x, y-d], [x+d, y], [x, y+d]])]
    cv2.fillPoly(mask, points, color=(1, 1, 1))
    return mask


def create_triangle_mask(mask_shape, y, x, l):
    mask = np.zeros(mask_shape)
    d = l//2
    h = int(d*math.sqrt(3))
    points = [np.array([[x-d, y], [x+d, y], [x, y-h]])]
    cv2.fillPoly(mask, points, color=(1, 1, 1))
    return mask
