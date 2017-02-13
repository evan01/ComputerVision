'''
This file exists because iPython and jupyter is the actual worst
'''
import cv2 as cv2
import matplotlib.pyplot as plt
from pictureLogger import imageLogger
import numpy as np
import warnings

warnings.filterwarnings("ignore")

il = imageLogger()
from tqdm import tqdm as tqdm

DEBUG = True  # Smaller image and more logging
WINDOW_SIZE = 10


def resizeImage(image, dbug=DEBUG):
    if dbug:
        image = cv2.resize(image, (480, 640))
        # image = cv2.resize(image, (120, 160))
        # image = cv2.resize(image, (1920, 2560))
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


def edgeMaximaSuppression(corn, windowSize):
    """
    This function takes an image, and only returns the local maximums of each pixel.
    Basically removes all of the 'not so hot corners'
    Taken from this algorithm below.
    http://stackoverflow.com/questions/29057159/non-local-maxima-suppression-in-python
    :param corn: Image with bad corners
    :return: Image without all the noise
    """
    dx, dy = windowSize, windowSize
    length, width = corn.shape[0], corn.shape[1]
    for x in tqdm(range(length - dx + 1), desc="Edge Maxima Suppression"):
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
            wind[:] = 0  # MAke all values in the window 0
            wind.flat[maxPosition] = lmax  # Places the maximum value in the correct place
    return corn


def cornerness(Sx2, Sy2, Sxy):
    '''
    The idea is that at each pixel of the image we define a value based off of the derivatives
    that have already been filtered by the Gaussian kernel

    For every single value. We need to find the HARRIS OPERATOR at that pixels value. This number
    tells us how likely that pixel is to being a corner somehow
    :param filteredImage:
    :return:
    '''
    A = .05
    sz = Sx2.shape
    corn = np.zeros(sz, dtype=np.float32)
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


def gaussian_filter2(ix2, iy2, ixy):
    """
    We need to apply a Gaussian filter to smooth out the original image somewhat
    The old implementation was too slow
    :param ix2: Padded X_squared derivative image
    :param iy2:'' y_ysqured'' ...
    :param ixy:
    :return:
    """
    kernel = [
        [1, 2, 1],
        [2, 4, 2],
        [1, 2, 1]
    ]
    kernel = np.array(kernel, dtype=np.float32)
    kernel / 16
    ix2 = convolutionFunction2(ix2, kernel)
    iy2 = convolutionFunction2(iy2, kernel)
    ixy = convolutionFunction2(ixy, kernel)

    return ix2, iy2, ixy


def convolutionFunction2(image, kernel):
    '''
    This function takes as input an image and then convolves it with the spec. kernel
    :param image: The original black and white image
    :param kernel: A 3x3 kernel
    :return: The filtered image
    '''
    image = np.array(image, dtype=np.float32)

    # Zero pad the image
    image = np.pad(image, 1, "constant")

    # FLIP THE KERNEL?? https://en.wikipedia.org/wiki/Kernel_(image_processing)
    kernel = np.fliplr(kernel)

    # Convert kernel to a 1x9 matrix
    kernel = np.reshape(kernel, (1, 9))

    # Then start convolving
    sz = image.shape
    newImg = np.zeros(sz, dtype=np.float32)
    # print ("Convoluding image with specified kernel" + str(kernel))
    for i in tqdm(range(1, sz[0] - 1), desc="Convolution function"):
        for j in range(1, sz[1] - 1):
            window = image[i - 1:i + 2, j - 1:j + 2]  # First get the points that make up our window
            # Then convolve the kernel with these points
            m1 = np.reshape(window, (1, 9))
            newImg[i][j] = np.inner(kernel, m1)

    return newImg


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
            try:
                D_X[i - 1][j - 1] = .5 * (im[i + 1][j] - im[i - 1][j])
                D_Y[i - 1][j - 1] = .5 * (im[i][j + 1] - im[i][j - 1])
            except Exception:
                D_X[i - 1][j - 1] = 0
                D_Y[i - 1][j - 1] = 0
    # testx,texy = np.gradient(im) #So much easier than this :/
    return D_X, D_Y


def displayHarrisCorners(nonMCorn, image):
    '''
    This function takes as input an image and an 'image of corners' which we will color
    and then overlay over the original image
    :param corn: A 2D array, the same size of the image, where the values represent corners
    :param image: A 2D array, that represents a standard greyscale image
    :return:
    '''

    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    thresh = 200
    sz = image.shape
    l, w = sz[0], sz[1]
    for i in range(l):
        for j in range(w):
            if nonMCorn[i][j] > thresh:
                cv2.circle(image, (j, i), 3, (0, 0, 255), 3)
    return image


def harrisCorners(im):
    # Zero pad the image
    im = np.pad(im, 1, "constant")

    # First compute the image derivatives
    dx, dy = computeImageDerivatives2(im)

    # Then square the derivatives
    ix2, iy2, ixy = squareDervivatives(dx, dy)

    # Then apply the gaussian filter
    Gx2, Gy2, Gxy = gaussian_filter2(ix2, iy2, ixy)

    # Then the cornerness function
    corn = cornerness(Gx2, Gy2, Gxy)

    # Then non-maxima suppresion
    nonMaxCorn = edgeMaximaSuppression(corn, WINDOW_SIZE)  # Perform the canny edge detector

    # Then display the harris corners
    finalImage = displayHarrisCorners(nonMaxCorn, im)

    if DEBUG:
        # First log the original image
        il.log(im, "padded")
        il.log(im, "greyscale before harris corners")
        il.log(dx, "x derivative")
        il.log(dy, "y derivatives")
        il.log(ix2, "x squared")
        il.log(iy2, "y squared")
        il.log(ixy, "xy")
        il.log(ix2, "x squared Gaussian")
        il.log(iy2, "y squared Gaussian")
        il.log(ixy, "xy Gaussian")
        il.log(corn, "Corners Without supression")
        il.log(nonMaxCorn, "Corners with supression")
        il.log(finalImage, "Harris corners")
    return finalImage


def harrisCorns():
    # First thing to do is import the image
    image = cv2.imread("./images/Rebecca1.jpg", 0)
    np.array(image, dtype=np.uint8)

    # then resize the image
    image = resizeImage(image)

    # Then apply the harris corners algorithm to the image
    harrisCorners(image)


if __name__ == '__main__':
    harrisCorns()
