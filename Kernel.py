import cv2
import numpy as np


def getKernel(KERNEL_TYPE):
    if KERNEL_TYPE == "dilation":
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    if KERNEL_TYPE == "opening":
        kernel = np.ones((3, 3), np.uint8)
    if KERNEL_TYPE == "closing":
        kernel = np.ones((11, 11), np.uint8)

    return kernel
#