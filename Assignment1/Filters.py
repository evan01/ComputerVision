import cv2 as cv2
import numpy as np

import Assignment1.Question4 as p
from Assignment1.pictureLogger import imageLogger

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
    print str(imav)
    immed = medianFilter(image)
    print str(immed)

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
    # filtered = cv2.filter2D(im,-1,kernel)
    filtered = p.convolutionFunction(original,kernel)
    il.log(filtered,"averaging Filter")

    return filtered


if __name__ == '__main__':
    main()