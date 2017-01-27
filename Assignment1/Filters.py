import numpy as np
import cv2 as cv2

from lib.pictureLogger import imageLogger
il = imageLogger()

def main():
    image = [
        [ 0, 135, 0, 135, 0],
        [0, 0, 0, 0, 0],
        [180, 180, 180, 0, 135],
        [180, 180, 180, 0, 0],
        [180, 180, 180, 0, 135]
     ]

    image = np.array(image,np.uint8)
    il.log(image,"Unfiltered")

    imav = averagingFilter(image)
    immed = medianFilter(image)

def medianFilter(original):
    im = original
    filtered = cv2.medianBlur(im,3)
    il.log(filtered,"median filter")
    return filtered

def averagingFilter(original):
    im = original
    kernel = [
        [1,1,1],
        [1,1,1],
        [1,1,1]
    ]

    kernel = np.array(kernel,np.float32)
    kernel=kernel/10
    filtered = cv2.filter2D(im,-1,kernel)
    il.log(filtered,"averaging Filter")

    return filtered


if __name__ == '__main__':
    main()