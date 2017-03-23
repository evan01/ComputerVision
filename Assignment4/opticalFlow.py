import cv2
from matplotlib import pyplot as plt
from collections import defaultdict
import numpy as np


class opticalFlow:
    DEBUG = True
    SKIPIMPORT = False  # Means to only import a minimal ammount of images
    IMAGES = defaultdict()
    outImages = []
    FONT_SIZE = 20
    WINDOW_SIZE = 7

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
            ima = np.array(im[0], dtype=np.uint8)
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
        print "Computing Flow: ",
        for i in range(1, shp[0] - 1):
            # if (i % 50 == 0):
            #     print ('.'),
            for j in range(1, shp[1] - 1):

                # First, it's not worth computing anything unless there is a dif in pixel values in the window
                # Could have used the whole window, but maybe this is a good speedup
                It = im_dt[i - 1:i + 2, j - 1:j + 2].flatten()
                if It.max() == 0:
                    continue

                # First get the dx and dy pixel values in the window
                Ix = im1_x[i - 1:i + 2, j - 1:j + 2].flatten()
                Iy = im1_y[i - 1:i + 2, j - 1:j + 2].flatten()

                # Another speedup here
                mX, mY = Ix.max(), Iy.max()
                if mX == 0 and mY == 0:
                    continue

                # Then get the sum of the products of these values as per the equation
                # Build the Atranspose A matrix
                Ixx = [0] * 9
                Ixy = [0] * 9
                Iyy = [0] * 9
                Ixt = [0] * 9
                Iyt = [0] * 9
                for idx in range(9):
                    Ixy[idx] = Ix[idx] * Iy[idx]
                    Ixx[idx] = Ix[idx] * Ix[idx]
                    Iyy[idx] = Iy[idx] * Iy[idx]
                    Ixt[idx] = Ix[idx] * It[idx]
                    Iyt[idx] = Iy[idx] * It[idx]

                # then compute the sums
                Ixy = sum(Ixy)
                Ixx = sum(Ixx)
                Iyy = sum(Iyy)
                Ixt = sum(Ixt)
                Iyt = sum(Iyt)

                # Then create the matrix ATA, and the
                A = np.array([[Ixx, Ixy], [Ixy, Iyy]])
                B = np.array([Ixt, Iyt])
                B = -B

                # Check to see if the matrix is invertible,and the eigenvals aren't too small
                eigenvals = np.array(np.linalg.eigvals(A), dtype=np.uint8)
                invertible = not np.any(eigenvals == 0)
                if invertible and abs(eigenvals[0]) > 1 and abs(eigenvals[1]) > 1 and eigenvals[0] > eigenvals[1]:
                    # Then we can invert the matrix, solve the system
                    A_Inv = np.linalg.inv(A)

                    # Then multiply the inverse to the left hand side of least squares equation to get flow vector
                    Flow = [int(A_Inv[0][0] * B[0] + A_Inv[1][0] * B[0]), int(A_Inv[0][1] * B[1] + A_Inv[1][1] * B[1])]

                    # if self.DEBUG:
                    #     print "Flow: " + str(Flow) + " Eigs: "+ str(eigenvals )+ " (" +str(i)+ ","+str(j)+")"
                    FLOW[i][j] = Flow

        return FLOW

    def opticalFlow2(self, im1_x, im1_y, im_dt):
        """
        This is the second implementation of optical flow with variable window sizes
        :param im1_x:
        :param im1_y:
        :param im_dt:
        :return:
        """
        d = int(self.WINDOW_SIZE / 2)  # So we can dynamically have varying sizes of windows
        shp = im1_x.shape
        prog = shp[0] / 10  # So we can have a 10% progress bar as we compute the values
        print ("Optical Flow: "),
        for i in range(d, shp[0] - d):
            if i % prog == 0:
                print ".",
            for j in range(d, shp[0] - d):
                # First get all the values within the windows to build column vectors
                # If there is no change between two windows then carry on
                dt = im_dt[i - d:i + d + 1, j - d:j + d + 1].flatten()
                if (dt.sum() == 0):
                    continue
                dx = im1_x[i - d:i + d + 1, j - d:j + d + 1].flatten()
                dy = im1_y[i - d:i + d + 1, j - d:j + d + 1].flatten()

                if (dx.sum() == 0 and dy.sum() == 0):
                    continue

                # then construct the matrices of interest A and A.T
                A_trans = np.array([dx, dy])
                A = A_trans.T
                dt = np.array(dt).T

                # Find Atrans*A and Atrans*T
                M = np.matmul(A_trans, A)
                T = np.matmul(A_trans, dt)

                # Then find the eigenvals of M
                eigenvals = np.array(np.linalg.eigvals(M))
                print "k"

    def placeVectorsOnImage(self, flow, image):

        # Then draw an arrow over the image for every... ten pixels?
        shp = flow.shape
        for i in range(1, shp[0], 10):
            for j in range(1, shp[1], 10):
                # Get the arrow vector
                u, v = int(flow[i][j][0]), int(flow[i][j][1])

                # Draw the arrow vector
                pt1 = (i, j)
                pt2 = (i + v, j + u)

                if self.DEBUG:
                    print "u: " + str(u) + " v: " + str(v) + " (" + str(i) + "," + str(j) + ")"

                image = cv2.arrowedLine(image, pt1, pt2, (0, 0, 255), 2)

        return image

    def main(self):
        # First upload both images, in grey and colour
        im1, im2 = cv2.imread("./images/flow_images/frame1.jpg", 0), cv2.imread("./images/flow_images/frame2.jpg", 0)
        im1_clr = cv2.imread("./images/flow_images/frame1.jpg")
        self.outImages.append((im1, "Frame 1"))
        self.outImages.append((im2, "Frame2"))
        self.outImages.append((im1_clr, "Frame1 Color"))

        # Then zero pad the images, given we are using windows
        padding = int(self.WINDOW_SIZE / 2)
        im1 = np.pad(im1, padding, "constant")
        im2 = np.pad(im2, padding, "constant")

        # Then find the X and Y derivatives of the image
        im1_x = cv2.Sobel(im1, cv2.CV_8U, 1, 0, ksize=5)
        im1_y = cv2.Sobel(im1, cv2.CV_8U, 0, 1, ksize=5)

        self.outImages.append((im1_x, "Sobel X of frame 1"))
        self.outImages.append((im1_y, "Sobel Y of frame 1"))

        # Also find the difference between the two frames It
        I_dt = np.subtract(im2, im1)

        # Then compute the optical flow
        # flow = self.opticalFlow(im1_x, im1_y, I_dt)
        flow = self.opticalFlow2(im1_x, im1_y,I_dt)

        # Then overlay flow vectors on the original image
        vectoredImage = self.placeVectorsOnImage(flow, im1_clr.copy())
        self.outImages.append((vectoredImage, "Final Image"))

        self.displayImages(self.outImages)


# %pylab
# %matplotlib inline
plt.rcParams['figure.figsize'] = (20, 20.0)
b = opticalFlow()
b.main()
