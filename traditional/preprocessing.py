import cv2
import numpy as np


def load_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return img


import cv2
import numpy as np

def roi(image):
    h, w = image.shape
    return image[int(h*0.2):int(h*0.8), int(w*0.2):int(w*0.8)]


def clahe(image, clip=9.0, grid=(18,18)):
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=grid)
    return clahe.apply(image)


def gaussian_blur(image, kernel=(5,5)):
    return cv2.GaussianBlur(image, kernel, 0)