import math

import cv2 as cv2
import matplotlib.pyplot as plt
import numpy as np

from Assignment1.pictureLogger import imageLogger

il = imageLogger()
from tqdm import tqdm as tqdm

'''
    This file has all of the code required for the
'''



def doBinaryTransformation(img, thresh):
    newIm = img.copy()
    shp = newIm.shape
    for i in tqdm(range(shp[0]), desc="binTransformation"):
        for j in range(shp[1]):
            if img[i][j] > thresh:
                newIm[i][j] = 255
            else:
                newIm[i][j] = 0
    il.log(newIm, "Binary image")
    return newIm


def getBinaryThreshold(hist,total):
    '''
    This will given a histogram, return a propper threshold to do the binary translation
    :param hist:
    :return:
    '''
    # Our histogram will have 1 initial peak and then 1 main trough, find these
    #We can use OTSU's method to do this https://en.wikipedia.org/wiki/Otsu%27s_method#JavaScript_implementation
    sum = 0
    for elem,i in zip(hist,range(len(hist))):
        sum += i * elem
    sumB = 0
    wb = 0
    wF = 0
    mB = 0
    mF = 0
    max = 0
    between = 0
    t1 = 0
    t2 = 0
    for i,count in zip(hist,range(len(hist))):
        wb += hist[count]
        if (wb == 0):
            continue
        wF = total - wb
        if (wF == 0):
            break
        sumB += count * hist[count]
        mB = sumB / wb
        mF = (sum - sumB) /wF
        between = wb*wF*(mB-mF)*(mB-mF)
        if (between >= max):
            t1 = count
            if between > max:
                t2 = count
            max = between

    return (t1+t2)/2

def histogram_SLOW_Greyscale(image):
    intensity = [0]*256
    for i in tqdm(image, desc="Greyscale histogram"):
        for j in i:
            intensity[int(j)] += 1
    # When you want to plot the image
    plt.plot(intensity)
    plt.savefig("./images/histogram.jpeg")
    il.log(cv2.imread("./images/histogram.jpeg"),"Histogram")

    return intensity


def histogram_SLOW_Color(image):
    R = np.ndarray((256, 1), dtype=np.uint8)
    G = np.ndarray((256, 1), dtype=np.uint8)
    B = np.ndarray((256, 1), dtype=np.uint8)

    for i in tqdm(image, desc="Calculating Histogram"):
        for j in image:
            for k in j:
                # Append BGR vals
                R[k[2]] += 1
                B[k[0]] += 1
                G[k[1]] += 1

    return R, G, B
    print "done"


def histogram_FAST_Color(img):
    color = ('b', 'g', 'r')
    for i, col in enumerate(color):
        histr = cv2.calcHist([img], [i], None, [256], [0, 256])
        plt.plot(histr, color=col)
        plt.xlim([0, 256])
    print "plotting the histogram"
    plt.show()


def plotHistogram(R, G, B):
    # We want to plot 3 line graphs where the dependent variable is the 0,255 intensity range.
    a = plt.plot(R)
    plt.show()


def sobelEdgeDetection(image):
    '''
    This function will apply the sobel edge detection algorithm to an image
    :param image:
    :return:
    '''
    im = image.copy()
    print "Running sobel edge detection"

    sx = [
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ]

    sy = [
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1]
    ]
    sx, sy = np.array(sx, np.uint8), np.array(sy, np.uint8)

    # Get the results of the filter output of every pixel for each mask on image
    im_x = convolutionFunction(im, sx)
    il.log(im_x, "X Convolution")

    im_y = convolutionFunction(im, sy)
    il.log(im_y, "Y Convolution")

    # Then combine each value to get the gradient and direction
    print "Combine values"
    sobelVals = getMagAndDir(im_x, im_y)

    # Then visualize this information somehow
    sobel = visualizeSobel(sobelVals)


def visualizeSobel(sobelInfo):
    # First thing is to create the final image we want to display
    img = []

    print "Visualizing the sobel information"
    for i in range(len(sobelInfo)):
        row = []
        for j in range(len(sobelInfo[i])):
            row.append(convertSobelValsToRGB2(sobelInfo[i][j]))
        img.append(row)

    image = np.array(img, dtype=np.uint8)
    il.log(image, "SOBEL IMAGE")


def convertSobelValsToRGB(val):
    '''
    This function will take as input a magnitude and a direction and then return a color
    :param val: a python tuple (magnitude, direction)
    :return: a [1X3] array
    '''
    rgb = [0, 0, 0]
    mag, theta = val[0], val[1]

    # First thing is to choose a value between 0 and 255
    # Do this by using the magnitude of the value
    colorIntensity = (mag) % 255

    # Next is to choose a color
    if theta > 0 and theta < 120:
        rgb = [colorIntensity, 0, 0]
    elif theta > 120 and theta < 240:
        rgb = [0, colorIntensity, 0]
    else:
        rgb = [0, 0, colorIntensity]

    # Return the rgb values
    return rgb


def convertSobelValsToRGB2(val):
    if val[0] > int(2*255/3):
        return [0,0,255]
    elif val[0]> int(255/3):
        return [0,0,0]
    else:
        return [0,0,255]

def getMagAndDir(sx, sy):
    '''
    this function will take in 2 sobel x,y filtered images and calculate the gradient information
    :param sx:
    :param sy:
    :return: (magnitude, direction) for every pixel value
    '''
    # first create an array that's the size of the initial image
    sobelArray = []

    for i in tqdm(range(sx.shape[0]),desc="Combining Sobel vals"):
        row = []
        for j in range(sx.shape[1]):
            # Get the value from each sobel response array
            jx = sx[i][j]
            jy = sy[i][j]

            mag = (jx ** 2 + jy ** 2) ** (.5)
            try:
                theta = math.degrees(math.atan(float(jy) / float(jx)))
            except:
                theta = math.degrees(math.atan(float(jy) / float(0.00001)))
            row.append((mag, theta))
        sobelArray.append(row)

    return sobelArray


def convolutionFunction(image, kernel):
    print "Convolving the image with specified kernel...\n" + str(kernel)
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
    for i in tqdm(range(1, ylen - 1), desc="Convolution function"):
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

    #FIRST FLIP THE KERNEL???
    kernel = np.fliplr(kernel)

    newValue = 0
    sz = kernel.shape
    for i in range(sz[0]):
        for j in range(sz[1]):
            fctor = points[i][j]* kernel[i][j]
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
    return np.array(points,dtype=np.float32)


def getKernel():
    kernel = [
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]
    ]

    # kernel = [
    #     [-1, -1, -1],
    #     [-1, 8, -1],
    #     [-1, -1, -1]
    # ]

    kernel = np.array(kernel, np.float32)
    kernel = kernel / 10
    return kernel

def main():
    # First log the original image in greyscale
    image = cv2.imread("./images/Rebecca1.jpg", 0)
    il.log(image, "Original")

    # Resize the image, the larger the image the longer the next steps take
    image = cv2.resize(image,(1920,2560))
    # image = cv2.resize(image, (480, 640))
    # image = cv2.resize(image, (120, 160))
    # il.log(image, "resized")
    shp = image.shape
    sz = int(shp[0]*shp[1])
    # Then do the convolution function
    kernel = getKernel()
    newIm = convolutionFunction(image, kernel)
    il.log(newIm, "Convoluded/Filtered image")
    #
    # Then the sobel edge detection algorithm
    sobelEdgeDetection(image)
    #
    # Or finally do the histogram of the image, and get the binary threshhold
    hist = histogram_SLOW_Greyscale(image)
    thresh = getBinaryThreshold(hist,sz)

    binTrans = doBinaryTransformation(image, thresh)


if __name__ == '__main__':
    main()
