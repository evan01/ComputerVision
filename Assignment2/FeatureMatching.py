import cv2 as cv2
import numpy as np
import warnings

from Assignment3.pictureLogger import imageLogger

warnings.filterwarnings("ignore")

il = imageLogger()

DEBUG = True


def resizeImage(image, dbug=DEBUG):
    if dbug:
        image = cv2.resize(image, (480, 640))
        # image = cv2.resize(image, (120, 160))
        # image = cv2.resize(image, (1920, 2560))
    else:
        image = cv2.resize(image, (1920, 2560))
    return image


def match(im1, im2):
    """
    Will apply the sift feature matching algorithm to 2 images
    The majority of the algorithm / code has been taken from this source below
    CITATION: http://docs.opencv.org/trunk/dc/dc3/tutorial_py_matcher.html
    :param im1:
    :param im2:
    :return:
    """
    '''
        First thing to do is to get the sift features
    '''
    print ("Computing the sift feature descriptors...")
    sift_1, sift_2 = cv2.xfeatures2d.SIFT_create(), cv2.xfeatures2d.SIFT_create()
    kp1, des1 = sift_1.detectAndCompute(im1, None)
    kp2, des2 = sift_2.detectAndCompute(im2, None)

    print ("Use the BF matcher to find the good matches between images")
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    print ("Pick the top 50 matches")
    bestMatches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            bestMatches.append((m, n))

    # Sort the matches by distance
    bMatches = sorted(bestMatches, key=lambda x: x[1].distance, reverse=False)[:50]

    print "Draw the matches"
    newIm = cv2.drawMatchesKnn(im1, kp1, im2, kp2, bMatches, None, flags=2)
    il.log(newIm, "matches")


    print ("done")


def featureMatch():
    # First thing to do is import the image
    im1 = cv2.imread("./images/Rebecca1.jpg", 0)
    np.array(im1, dtype=np.uint8)

    im2 = cv2.imread("./images/Rebecca2.jpg", 0)
    np.array(im2, dtype=np.uint8)

    # then resize the images
    im1 = resizeImage(im1)
    im2 = resizeImage(im2)

    # Then do the feature matching
    match(im1, im2)


if __name__ == '__main__':
    featureMatch()
