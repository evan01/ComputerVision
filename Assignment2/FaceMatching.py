import cv2 as cv2
import matplotlib.pyplot as plt
from pictureLogger import imageLogger
import numpy as np
import warnings
import os

warnings.filterwarnings("ignore")

il = imageLogger()
from tqdm import tqdm as tqdm

DEBUG = True


def importCelebImages():
    """
    This function will import all the images, and compute their descriptors
    :return: an array of image descriptors
    """
    os


def resizeImage(image, dbug=DEBUG):
    if dbug:
        image = cv2.resize(image, (128, 128))
        # image = cv2.resize(image, (120, 160))
        # image = cv2.resize(image, (1920, 2560))
    else:
        image = cv2.resize(image, (1920, 2560))
    return image


def computeLBPDescriptor(im):
    # Zero pad the image
    im = np.pad(im, 1, "constant")

    sz = im.shape
    histogram = np.zeros((256, 1), dtype=np.uint8)

    '''
    First thing to do is to get the binary label for each pixel value,
    and while doing that, build our histogram
    '''
    for i in range(1, sz[0] - 1):
        for j in range(1, sz[1] - 1):
            window = im[i - 1:i + 2, j - 1:j + 2]
            thresh = im[i][j]
            # Gets you the indexes that fall below/above the values
            zeroIndexes = window < thresh
            oneIndexes = window >= thresh
            wincpy = window.copy()
            wincpy[zeroIndexes] = 0
            wincpy[oneIndexes] = 1

            # Then get the unique binary value
            binary = wincpy.flatten()
            binary = np.delete(binary, 4)  # The 4th element is the middle index
            bin_val = binary.dot(2 ** np.arange(binary.size)[::-1])
            histogram[bin_val] += 1

    return histogram

def breakImageApart(im):
    """
    this function will break up the original image apart into roughly equal 7x7 sections
    :param im:
    :return:
    """
    segments = []
    # First get all of the vertical slices, should be around 18 pixels tall each
    horizontalSlices = np.array_split(im, 7)

    # Then for each vertical slice, horizontally split the arrays into segments...
    for i in horizontalSlices:
        verticalSlices = np.array_split(i, 7, axis=1)
        for slice in verticalSlices:
            segments.append(slice)

    return segments

def main():
    """
    This function will be a celebrity face matching exercise with my face
    :return: The celebrity I look the most like
    """

    # First thing is to import all the images, resize them, and compute their descriptors
    images = importCelebImages()

    # Then get our image, resize it and compute it's descriptor
    me_im = cv2.imread("./images/Rebecca1.jpg", 0)
    me_im = resizeImage(me_im)
    imSegments = breakImageApart(me_im)
    me_features = [computeLBPDescriptor(i) for i in tqdm(imSegments, desc="Finding features")]

    # Then compare our fDescriptor with all the others! Will return index of image we want
    im = compareFeatures(me_features, all_features)
    print "done"
    # lbp = computeLBPDescriptor(me_im)


    # Compute the lb feature descriptor


if __name__ == '__main__':
    main()
