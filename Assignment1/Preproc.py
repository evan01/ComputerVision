import numpy as np
import cv2 as cv2
from lib.pictureLogger import imageLogger

il = imageLogger()
from tqdm import tqdm as tqdm


def main():
    image = cv2.imread("./images/IMG_6967.jpg", 0)
    il.log(image, "Original")

    # Resize the image
    image = cv2.resize(image, (480, 640))
    # image = cv2.resize(image, (120, 160))
    il.log(image, "resized")

    # Then do the convolution function
    kernel = getKernel()
    newIm = convolutionFunction(image, kernel)
    il.log(newIm, "Convoluded/Filtered image")

    # Or do the sobel edge detection algorithm
    sobelEdgeDetection(image)

def sobelEdgeDetection(image):
    '''
    This function will apply the sobel edge detection algorithm to an image
    :param image:
    :return:
    '''
    im = image.copy()
    print "Running sobel edge detection"

    sx = [
        [-1,0,1],
        [-2,0,2],
        [-1,0,1]
    ]

    sy = [
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1]
    ]
    sx,sy = np.array(sx,np.uint8),np.array(sy,np.uint8)

    #Get the results of the filter output of every pixel for each mask on image
    im_x = convolutionFunction(im,sx)
    il.log(im_x,"X Convolution")

    im_y = convolutionFunction(im,sy)
    il.log(im_y,"Y Convolution")

    #Then combine each value to get the gradient and direction
    print "Combine values"

    #Then visualize this information somehow

def getMagAndDir(sx,sy):
    '''
    this function will take in 2 sobel x,y filtered images and calculate the gradient information
    :param sx:
    :param sy:
    :return:
    '''


def convolutionFunction(image, kernel):
    print "Convolving the image with specified kernel..."
    im = image.copy()
    newIm = image.copy()
    # First zero pad the image
    im = np.pad(im, 1, "constant")
    il.log(im, "padded")

    # Then convolude the image
    shape = im.shape
    xlen = shape[1]
    ylen = shape[0]

    # Loop through the whole image
    for i in tqdm(range(1, ylen - 1)):
        for j in range(1, xlen - 1):
            # Get the points immediately surrounding the point of interest in original image
            points = getPointsWith3Kernel(i, j, im)
            newIm[i - 1][j - 1] = convolvePointsWith3Kernel(points, kernel)

    print "Done the convolution function"
    return newIm


def convolvePointsWith3Kernel(points, kernel):
    '''
    This function will convolve an area of 9 pixels with a 3x3 kernel, and return a value of convolution
    :type kernel: np.array
    :param points: [a,b,]
    :param kernel:
    :return:
    '''

    newValue = 0
    sz = kernel.shape
    for i in range(sz[0]):
        for j in range(sz[1]):
            fctor = points[(3 * i) + j] * kernel[i][j]
            newValue = newValue + fctor
    return newValue


def getPointsWith3Kernel(x, y, imP):
    '''
    A function that given an x,y coordinate of an image, will return the 9points around that point (point inclusive).
    We can assume a zero padded image.
    :return: The 9 points around the points of an image
    [a,b,c]
    [d,e,f]
    [g,h,i] ----> [a,b,c,d,e,f,g,h,i]

    '''

    points = []
    for i in range(x - 1, x + 2):
        for j in range(y - 1, y + 2):
            points.append(imP[i][j])

    return points


def getKernel():
    kernel = [
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]
    ]

    kernel = np.array(kernel, np.float32)
    kernel = kernel / 10
    return kernel


if __name__ == '__main__':
    main()
