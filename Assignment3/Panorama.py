import cv2
import numpy as np


class Stitcher:
    # the __init__ function is automatically called upon instantiation
    # is't a good practice to initialize/reset all class members here
    def __init__(self):
        # create a SIFT object
        self.sift = cv2.xfeatures2d.SIFT_create()
        # create a Brute-Force Matcher
        self.bf = cv2.BFMatcher(cv2.NORM_L2)

    def featureExtraction(self):
        self.kp1, self.desc1 = self.sift.detectAndCompute(self.img1, None)
        self.kp2, self.desc2 = self.sift.detectAndCompute(self.img2, None)

    def matchDescriptors(self):
        self.matches = self.bf.match(self.desc1, self.desc2)

    def displayMatches(self, N):
        # Sort them in the order of their distance
        matches = sorted(self.matches, key=lambda x: x.distance)
        # Draw the best N matches
        img = cv2.drawMatches(self.img1, self.kp1, self.img2, self.kp2, matches[:N], None, flags=2)
        cv2.imshow("Matched points", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def homography(self):
        # We need to convert the matched points (cv.DMatch) into Nx2 Numpy arrays
        # A cv2.DMatch object contains distance, queryIdx and trainIdx
        # distance: The Euclidean distance between the the descriptors from the two sets
        # queryIdx: The index of the matched descriptor in the query set (1st set)
        # trainIdx:: The index of the matched descriptor in the train set (2nd set)

        # local placeholders for the matched points coordinates
        srcPts = np.empty((len(self.matches), 2))
        dstPts = np.empty((len(self.matches), 2))
        for i in range(len(self.matches)):
            a = self.matches[i].queryIdx
            srcPts[i, :] = np.float32(self.kp1[self.matches[i].queryIdx].pt)
            dstPts[i, :] = np.float32(self.kp2[self.matches[i].trainIdx].pt)

        # compute the 3x3 transformation matrix to map dstPts into srcPts (note the ordering)
        self.M, mask = cv2.findHomography(dstPts, srcPts, cv2.RANSAC, 5.0)

    def findOutputLimits(self):
        # we need to find the projected image corners to determine the size of the panorama image
        # the projected corners might map to negative pixel coordinates -> translate the image

        # the four corners are [0,0], [0,height-1], [width-1,0] and [width-1,height-1]
        # [x_proj y_proj 1] = M * [x_corner y_corner 1]

        # top left
        tl = np.dot(self.M, np.array([0, 0, 1]))
        tl = tl / tl[-1]
        # top right
        tr = np.dot(self.M, np.array([self.img2.shape[1] - 1, 0, 1]))
        tr = tr / tr[-1]
        # bottom left
        bl = np.dot(self.M, np.array([0, self.img2.shape[0] - 1, 1]))
        bl = bl / bl[-1]
        # bottom right
        br = np.dot(self.M, np.array([self.img2.shape[1] - 1, self.img2.shape[0] - 1, 1]))
        br = br / br[-1]

        # find the xMin and yMin
        self.xMin = min(tl[0], tr[0], bl[0], br[0], 0)
        self.xMax = max(tl[0], tr[0], bl[0], br[0], self.img1.shape[1])
        self.yMin = min(tl[1], tr[1], bl[1], br[1], 0)
        self.yMax = max(tl[1], tr[1], bl[1], br[1], self.img1.shape[0])

        # create a 3x3 translation matrix
        # [ [1 0 -xMin], [0 1 -yMin], [0 0 1] ]
        self.T = np.array([[1, 0, -self.xMin], [0, 1, -self.yMin], [0, 0, 1]])

        # apply the translation matrix to the transformation matrix
        #  M <- T * M
        self.M = np.dot(self.T, self.M)

        # compute the panorama size
        self.panoSize = (int(self.xMax - self.xMin + 1), int(self.yMax - self.yMin + 1))

    def perspectiveWarp(self):
        # warp the second image into the panorama
        self.pano = cv2.warpPerspective(self.img2, self.M, self.panoSize)
        # no blending, just copy the left-side image into the panorama
        self.pano[self.yMin:self.img1.shape[0] + self.yMin, self.xMin:self.img1.shape[1] + self.xMin] = self.img1

    def doStitch(self, image1, image2):
        # make a deep copy of the input images
        print("Making copies")
        self.img1 = image1.copy()
        self.img2 = image2.copy()

        # compute keypoints/descriptors
        print("Computing descriptors")
        self.featureExtraction()

        # match descriptors
        print("Computing descriptors")
        self.matchDescriptors()

        # compute the transformation matrix
        print("Computing descriptors")
        self.homography()

        # translate the images and compute the panorama size
        print("Computing descriptors")
        self.findOutputLimits()

        # apply a perspective warp to stitch the images\n,
        print("Computing descriptors")
        self.perspectiveWarp()


im1 = cv2.imread("./images/landscapeL.jpg")
im2 = cv2.imread("./images/landscapeR.jpg")

im1 = cv2.resize(im1, (120, 160))
im2 = cv2.resize(im2, (120, 160))
s = Stitcher()
s.doStitch(im1, im2)
