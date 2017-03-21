import cv2
from matplotlib import pyplot as plt
import os
from collections import defaultdict
import numpy as np
import itertools
from scipy.cluster.vq import vq, kmeans2
from random import shuffle
from tqdm import tqdm


class Vector:
    def __init__(self, x, y):
        self.u = x
        self.v = y


class opticalFlow:
    DEBUG = True
    SKIPIMPORT = False  # Means to only import a minimal ammount of images
    IMAGES = defaultdict()
    outImages = []
    FONT_SIZE = 20

    def resizeImage(self, image, dbug=DEBUG):
        if dbug:
            image = cv2.resize(image, (120, 160))
        else:
            image = cv2.resize(image, (2400, 1200))
        return image

    def displayImage(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.figure()
        plt.title(img[1], fontsize=self.FONT_SIZE, fontweight='bold', color="green")
        plt.imshow(img[0])

        plt.imshow(img)

    def displayImages(self, images):
        print("Displaying outs")
        for im in images:
            ima = im[0]
            if len(list(ima.shape)) < 3:
                ima = cv2.cvtColor(ima, cv2.COLOR_GRAY2RGB)
            else:
                ima = cv2.cvtColor(ima, cv2.COLOR_BGR2RGB)
            plt.figure()
            plt.title(im[1], fontsize=self.FONT_SIZE, fontweight='bold', color="green")
            plt.imshow(ima)

    def opticalFlow(self, im1_x, im1_y, im_dt):
        # First create the matrix of vectors that we want to compute
        shp = im1_x.shape
        FLOW = np.zeros((shp[0], shp[1], 2))

        ##Then for every pixel in the image, compute the matrix A * transA, ASSUME WIND of width and height = 3

        for i in tqdm(range(1, shp[0] - 1)):
            for j in range(1, shp[1] - 1):

                # First, it's not worth computing anything unless there is a dif in pixel values in the window
                # Could have used the whole window, but maybe this is a good speedup
                if im_dt[i][j] == 0:
                    continue

                # First get the dx and dy pixel values in the window
                Ix = im1_x[i - 1:i + 2, j - 1:j + 2].flatten()
                Iy = im1_y[i - 1:i + 2, j - 1:j + 2].flatten()
                It = im_dt[i - 1:i + 2, j - 1:j + 2].flatten()

                # Then get the sum of the products of these values as per the equation
                # todo numpy has got to have a better way of doing this...
                Ixy, Ixx, Iyy = np.array([0] * 9), np.array([0] * 9), np.array([0] * 9)
                Ixt, Iyt = np.array([0] * 9), np.array([0] * 9)
                for i in range(9):
                    Ixy[i] = Ix[i] * Iy[i]
                    Ixx[i] = Ix[i] * Ix[i]
                    Iyy[i] = Iy[i] * Iy[i]
                    Ixt[i] = Ix[i] * It[i]
                    Iyt[i] = Iy[i] * It[i]

                # then compute the sums
                Ixy = np.sum(Ixy)
                Ixx = np.sum(Ixx)
                Iyy = np.sum(Iyy)
                Ixt = np.sum(Ixt)
                Iyt = np.sum(Iyt)

                # Then create the matrix ATA, and the
                A = np.array([[Ixx, Ixy], [Ixy, Iyy]])
                B = np.array([Ixt, Iyt])
                B = -B

                # Check to see if the matrix is invertible,and the eigenvals aren't too small
                eigenvals = np.linalg.eigvals(A)
                if np.all(eigenvals > 1) and (eigenvals[0] / eigenvals[1]) / 1:
                    # Then we can invert the matrix, solve the system
                    A_Inv = np.linalg.inv(A)

                    # Then multiply the inverse to the left hand side of least squares equation to get flow vector
                    Flow = [A_Inv[0][0] * B[0] + A_Inv[1][0] * B[0], A_Inv[0][1] * B[1] + A_Inv[1][1] * B[1]]
                    FLOW[i][j] = Flow

        return FLOW

    def placeVectorsOnImage(self, flow, image):
        # First zero pad the image, given the flow matrix was computed with zeros padded around it
        image = np.pad(image, 1, "constant")

        # Then draw an arrow over the image for every... ten pixels?

    def main(self):
        # First upload both images, in grey and colour
        im1, im2 = cv2.imread("./images/flow_images/frame1.jpg", 0), cv2.imread("./images/flow_images/frame2.jpg", 0)
        im1_clr, im2_clr = cv2.imread("./images/flow_images/frame1.jpg"), cv2.imread("./images/flow_images/frame2.jpg")
        self.outImages.append((im1, "Frame 1"))
        self.outImages.append((im2, "frame2"))

        # Then zero pad the images, given we are using windows
        im1 = np.pad(im1, 1, "constant")
        im2 = np.pad(im2, 1, "constant")

        # Then find the X and Y derivatives of the image
        im1_x, im1_y = cv2.Sobel(im1, cv2.CV_8U, 1, 0, ksize=5), cv2.Sobel(im1, cv2.CV_8U, 0, 1, ksize=5)
        im2_x, im2_y = cv2.Sobel(im2, cv2.CV_8U, 1, 0, ksize=5), cv2.Sobel(im1, cv2.CV_8U, 0, 1, ksize=5)
        self.outImages.append((im1_x, "Sobel X of frame 1"))
        self.outImages.append((im1_y, "Sobel Y of frame 1"))
        self.outImages.append((im2_x, "Sobel X of frame 2"))
        self.outImages.append((im2_y, "Sobel Y of frame 2"))

        # Also find the difference between the two frames It
        I_dt = np.subtract(im2, im1)

        # Then compute the optical flow
        flow = self.opticalFlow(im1_x, im1_y, I_dt)

        # Then overlay flow vectors on the original image
        vectoredImage = self.placeVectorsOnImage(flow, im1_clr)

        self.displayImages(self.outImages)


# %pylab
# %matplotlib inline
plt.rcParams['figure.figsize'] = (20, 20.0)
b = opticalFlow()
b.main()
