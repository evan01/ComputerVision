'''
This file has the code for QUESTION 2 of the assignment
'''
import cv2
from matplotlib import pyplot as plt
import numpy as np


class MS_Window:
    def __init__(self, lower, upper):
        self.lower = lower
        self.upper = upper
        self.sum = 0
        self.mean = 0
        self.oldMean = 0
        self.elements = 0
        self.locked = False


class Segment:
    DEBUG = True
    outputImages = []
    FONT_SIZE = 50
    MS_WINDOW_SIZE = 50

    def resizeImage(self, image, dbug=DEBUG):
        if dbug:
            image = cv2.resize(image, (400, 600))
        else:
            image = cv2.resize(image, (400, 600))
        return image

    def displayImage(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img)

    def displayImages(self, images):
        print("Displaying outs")
        for im in images:
            ima = im[0]
            if len(ima.shape) > 2:
                ima = cv2.cvtColor(ima, cv2.COLOR_BGR2RGB)
            else:
                ima = cv2.cvtColor(ima, cv2.COLOR_GRAY2RGB)
            plt.figure(figsize=(80, 20))
            plt.title(im[1], fontsize=self.FONT_SIZE, fontweight='bold', color="green")
            plt.imshow(ima)

    def getImages(self, path):

        im_clr = self.resizeImage(cv2.imread(path))
        self.outputImages.append((im_clr, "Original Color image"))
        im_gray = self.resizeImage(cv2.imread(path, 0))
        self.outputImages.append((im_gray, "Original Gray image"))
        return im_gray, im_clr

    def kMeans(self, image, features, k):
        # First create a cluster of size k
        groups = [(i, j) for i, j in zip(range(k), range(k))]

        # then initialize the centroids with k random intensities
        print "init centroids"

    def gausMix(self, image, features, k):
        pass

    def meanShift_Intensity(self, features, winSize=50, maxIter=3, windSimilarity=3):
        '''
        This function will return a segmented image based on intensity features
        :param features: A set of ((x,y),Intensity) features that describe each pixel of image
        :return:
        '''
        # First create the proper number of windows, index in dictionary
        numWindows = int(255 / winSize)
        winHalf = int(winSize / 2)
        windows = dict.fromkeys(range(numWindows), None)

        # Then instantiate each window
        i = 0
        for i in range(numWindows):
            windows[i] = MS_Window(i * winSize, i * winSize + winSize)
            i += winSize

        iterations = 0
        lockedWindows = 0
        while (iterations < maxIter):
            print ("Iteration " + str(iterations))

            # First assign every feature a corresponding window
            for i in features:
                # For each feature, get the intensity, and map that to a window
                for win in range(numWindows):
                    if not windows[win].locked and i[1] >= windows[win].lower and i[1] < windows[win].upper:
                        # then assign the feature to the window
                        windows[win].sum += i[1]
                        windows[win].elements += 1

                        # also assign each feature to the proper window number
                        i[2] = win
                        break

            # Then recompute the means of windows and 'shift' the window
            for i in range(numWindows):
                if (windows[i].locked):
                    continue
                newMean = int(windows[i].sum / windows[i].elements)
                if abs(windows[i].mean - newMean) < windSimilarity:
                    # then we have a convergence
                    windows[i].locked = True
                    lockedWindows += 1
                else:
                    windows[i].mean = newMean
                windows[i].lower = windows[i].mean - winHalf
                windows[i].upper = windows[i].mean + winHalf
                if windows[i].lower < 0: windows[i].lower = 0
                if windows[i].upper > 255: windows[i].lower = 255
                windows[i].sum = 0
                windows[i].elements = 0

            # Convergence when all windows are no longer moving a lot
            if (lockedWindows == numWindows):
                break
            else:
                # Otherwise let's keep on going
                iterations += 1

        print "done"

    def getIntensityFeatures(self, im):
        '''
        This function will loop through each pixel in the image and then get the key features
        :return:
        '''
        # bins = np.histogram(im,range(256))
        intensityFeatures = [((i, j), im[i][j], 0) for i in range(len(im)) for j in range(len(im[i]))]
        if self.DEBUG:
            plt.hist(im.ravel(), 256, [0, 256]);
            plt.autoscale()
            plt.savefig("./images/histogram.png")
            hist = cv2.imread("./images/histogram.png")
            self.outputImages.append((hist, "Intensity Histogram"))

        return intensityFeatures

    def intensitySegmentation(self, features):
        # Do Mean Shift
        print "Mean shift with intensity features"
        mS_intensity = self.meanShift_Intensity(features, self.MS_WINDOW_SIZE)
        # Do kmeans

        # Do gaus

    def rgbSegmentation(self):
        pass

    def main(self):
        '''
        This is the main function for the segmentation class
        :return:i
        '''
        # First get the images
        print ("Loading the images")
        path = "./images/sarina.jpg"
        im_gray, im_color = self.getImages(path)

        # Then get the feature spaces for Intensity and RGB
        print ("Getting the intensity features")
        intensity_features = self.getIntensityFeatures(im_gray)

        # Then do the segmentations for the intensity features
        print "Run intensity segmentation algorithms"
        self.intensitySegmentation(intensity_features)

        # Then do segmentations for the color features

        if self.DEBUG:
            self.displayImages(self.outputImages)


# %pylab inline
# %matplotlib inline
plt.rcParams['figure.figsize'] = (60, 30)
c = Segment()
c.main()
