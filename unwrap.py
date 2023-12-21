#!pip install unwrap_labels
import cv2
import numpy as np
import matplotlib.pyplot as plt

from unwrap_labels import LabelUnwrapper


def detect_and_unwrap(imgcenter, points, plot_points=False):
    if plot_points:
        plt.imshow(imgcenter)#results[0].masks.data[0])
        for i, dot in enumerate(points):
            plt.scatter(*dot, label = i)

        plt.legend()

    h, w = imgcenter.shape[:2]

    points_unwrap = [[x/w, y/h] for x, y in points]
    
    unwrapper = LabelUnwrapper(src_image=imgcenter, percent_points=points_unwrap)
    dst_image = unwrapper.unwrap()
    
    return dst_image
