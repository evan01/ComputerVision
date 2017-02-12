'''
This file exists because iPython and jupyter is the actual worst
'''
import cv2 as cv2
import matplotlib.pyplot as plt
from pictureLogger import imageLogger
import numpy as np

il = imageLogger()
from tqdm import tqdm as tqdm

DEBUG = True  # Smaller image and more logging


def resizeImage(image, dbug=DEBUG):
    if dbug:
        # image = cv2.resize(image, (480, 640))
        image = cv2.resize(image, (120, 160))
    else:
        image = cv2.resize(image, (1920, 2560))
    return image


def convolvePointsWith3Kernel(points, kernel):
    '''
    This function will convolve an area of 9 pixels with a 3x3 kernel, and return a value of convolution
    :type kernel: np.array
    :param points: [a,b,]
    :param kernel:
    :return:
    '''

    # FIRST FLIP THE KERNEL???
    kernel = np.fliplr(kernel)

    newValue = 0
    sz = kernel.shape
    for i in range(sz[0]):
        for j in range(sz[1]):
            fctor = points[i][j] * kernel[i][j]
            newValue += fctor
    return int(newValue)


def getPointsWith3Kernel(x, y, imP):
    '''
    A function that given an x,y coordinate of an image, will return the 9points around that point (point inclusive).
    We can assume a zero padded image.
    :return: The 9 points around the point e of an image, including the point passed in
    [a,b,c]
    [d,e,f]
    [g,h,i] ----> [a,b,c,d,e,f,g,h,i]

    '''

    points = []
    for i in range(x - 1, x + 2):
        row = []
        for j in range(y - 1, y + 2):
            row.append(imP[i][j])
        points.append(row)
    return np.array(points, dtype=np.float32)


def convolutionFunction(image, kernel):
    print "Convolving the image with specified kernel...\n" + str(kernel)
    im = image.copy()
    newIm = image.copy()
    # First zero pad the image
    im = np.pad(im, 1, "constant")

    # Then convolude the image
    shape = im.shape
    xlen = shape[1]
    ylen = shape[0]

    # Loop through the whole image
    for i in tqdm(range(1, ylen - 1), desc="Convolution function"):
        for j in range(1, xlen - 1):
            # Get the points immediately surrounding the point of interest in original image
            points = getPointsWith3Kernel(i, j, im)
            newIm[i - 1][j - 1] = convolvePointsWith3Kernel(points, kernel)

    return newIm


def edgeMaximaSuppression(corn, windowSize):
    '''
    This function takes an image, and only returns the local maximums of each pixel.
    Basically removes all of the 'not so hot corners'
    Taken from this algorithm below.
    http://stackoverflow.com/questions/29057159/non-local-maxima-suppression-in-python
    :param corn: Image with bad corners
    :return: Image without all the noise
    '''
    dx, dy = windowSize, windowSize
    length, width = corn.shape[0], corn.shape[1]
    for x in range(length - dx + 1):
        for y in range(0, width - dy + 1):
            '''
            The idea is to create a window and move it along the entire image
            Then you can sum all the values inside of the window, apply operations to them...
            In this case, we want to find the local maxima of the window, and then make all the other pixel vals 0
            '''
            wind = corn[x:x + dx, y:y + dy]  # Create the window
            # If the sum of values in the window is 0, then the local max is a zero
            if np.sum(wind) == 0:
                lmax = 0
            else:
                lmax = np.amax(wind)  # Gets the maximum value along the window
            maxPosition = np.argmax(wind)  # The the x,y value that has this maximum value


def cornerness(Sx2, Sy2, Sxy):
    '''
    The idea is that at each pixel of the image we define a value based off of the derivatives
    that have already been filtered by the Gaussian kernel

    For every single value. We need to find the HARRIS OPERATOR at that pixels value. This number
    tells us how likely that pixel is to being a corner somehow
    :param filteredImage:
    :return:
    '''
    A = 1
    sz = Sx2.shape
    corn = np.zeros(sz, dtype=np.uint8)
    # First iterate through every pixel in the different images
    for i in tqdm(range(sz[0]), desc="Cornerness Function"):
        for j in range(sz[1]):
            x = Sx2[i][j]
            y = Sy2[i][j]
            xy = Sxy[i][j]

            # Apply harris cornerness function as per slide 46 of Lecture 6, Computer Vision, McGill
            har = x * y - (xy ** 2) - (A * (x + y))
            corn[i][j] = har

    return corn


def gaussian_filter(ix2, iy2, ixy):
    '''
    The 3x3 kernel was taken from http://dev.theomader.com/gaussian-kernel-calculator/
    :param ix2:chro
    :param iy2:
    :param ixy:
    :param image:
    :return:
    '''
    kernel = [
        [1, 2, 1],
        [2, 4, 2],
        [1, 2, 1]
    ]
    kernel = np.array(kernel, dtype=np.float32)
    kernel / 16

    ix2 = convolutionFunction(ix2, kernel)
    iy2 = convolutionFunction(iy2, kernel)
    ixy = convolutionFunction(ixy, kernel)

    return ix2, iy2, ixy


def squareDervivatives(dx, dy):
    '''
    This function will square all of the derivatives that we are interested in and
    generate the x squared, y squared and xy squared values nescessary for finding the harris corners
    :param dx:
    :param dy:
    :return:
    '''
    shp = dx.shape
    ix2, iy2, ixy = np.zeros(shp), np.zeros(shp), np.zeros(shp)
    length, width = shp[0], shp[1]
    for i in tqdm(range(length), "Squaring derivatives"):
        for j in zip(range(width)):
            # Get the value at the x,y coordinate of each image
            x = dx[i][j]
            y = dy[i][j]

            ix2[i][j] = x ** 2
            iy2[i][j] = y ** 2
            ixy[i][j] = x * y
    return ix2, iy2, ixy


def computeImageDerivatives(im):
    # todo optionally blue the image...
    sx = [
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ]

    sy = [
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]
    ]
    sx, sy = np.array(sx, np.uint8), np.array(sy, np.uint8)

    sx, sy = sx, sy

    # Get the results of the filter output of every pixel for each mask on image
    im_x = convolutionFunction(im, sx)
    im_y = convolutionFunction(im, sy)

    return im_x, im_y


def computeImageDerivatives2(im):
    '''
    Seems annoying to manually implement the np.gradient function but here goes
    Gradient function taken from the site below
    http://math.stackexchange.com/questions/1394455/how-do-i-compute-the-gradient-vector-of-pixels-in-an-image
    :param im: Unpadded raw image
    :return:
    '''

    # Create the derivative image copies
    D_X, D_Y = np.zeros(im.shape), np.zeros(im.shape)
    shp = im.shape

    # Zero pad the image
    im = np.pad(im, 1, "constant")

    # Then starting at 1,1 the (original) beginning of the image, start computing the derivatives
    dx, dy = 0, 0
    for i in tqdm(range(1, shp[0]), desc="Computing Image Derivatives"):
        for j in range(1, shp[1]):
            # Then compute the image derivatives
            dx = .5 * (im[i + 1][j] - im[i - 1][j])
            dy = .5 * (im[i][j + 1] - im[i][j - 1])
            D_X[i - 1][j - 1] = dx
            D_Y[i - 1][j - 1] = dy

    # testx,texy = np.gradient(im) #So much easier than this :/
    return D_X, D_Y


def harrisCorners(im):
    # First log the original image
    il.log(im, "greyscale before harris corners")

    # Zero pad the image
    im = np.pad(im, 1, "constant")
    il.log(im, "padded")

    # First compute the image derivatives
    dx, dy = computeImageDerivatives2(im)
    il.log(dx, "x derivative")
    il.log(dy, "y derivatives")

    # Then square the derivatives
    ix2, iy2, ixy = squareDervivatives(dx, dy)
    il.log(ix2, "x squared")
    il.log(iy2, "y squared")
    il.log(ixy, "xy")

    # Then apply the gaussian filter
    Gx2, Gy2, Gxy = gaussian_filter(ix2, iy2, ixy)
    il.log(ix2, "x squared Gaussian")
    il.log(iy2, "y squared Gaussian")
    il.log(ixy, "xy Gaussian")

    # Then the cornerness function
    corn = cornerness(Gx2, Gy2, Gxy)

    # Then non-maxima suppresion
    edgeMaximaSuppression(corn)  # Perform the canny edge detector


def main():
    # First thing to do is import the image
    image = cv2.imread(".//images/Rebecca1.jpg", 0)

    # then resize the image
    image = resizeImage(image)

    # Then apply the harris corners algorithm to the image
    harrisCorners(image)


if __name__ == '__main__':
    main()
