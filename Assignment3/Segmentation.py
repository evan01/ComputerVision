'''
This file has the code for QUESTION 2 of the assignment
'''
import cv2
from matplotlib import pyplot as plt
import numpy as np
import random as r
import itertools
import math
from scipy.spatial import distance


class MS_Window:
    def __init__(self, lower, upper, bin):
        self.lower = lower
        self.upper = upper
        self.sum = 0
        self.mean = 0
        self.oldMean = 0
        self.elements = 0
        self.locked = False
        self.color = [r.randint(0, 255), r.randint(0, 255), r.randint(0, 255)]
        self.bin = bin


class MS_WINDOW_COLOR:
    def __init__(self, color, bin):
        self.sum = [0, 0, 0]
        self.oldColor = Color([0, 0, 0])
        self.elements = 0
        self.locked = False
        self.color = color
        self.bin = bin


class Color:
    def __init__(self, color):
        self.R = color[0]
        self.G = color[1]
        self.B = color[2]

    def getColor(self):
        return [self.B, self.G, self.R]


class ColorFeature:
    def __init__(self, color, x, y, bin):
        self.clr = color
        self.x = x
        self.y = y
        self.bin = bin


class Segment:
    DEBUG = False
    outputImages = []
    FONT_SIZE = 50
    MS_WINDOW_SIZE = 50
    GRAY_SHAPE = None
    CLR_SHAPE = None
    MS_COLOR_WINDOWS = None

    def resizeImage(self, image, dbug=DEBUG):
        print ("Resizing images")
        if dbug:
            image = cv2.resize(image, (25, 50))
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
        self.CLR_SHAPE = im_clr.shape
        self.outputImages.append((im_clr, "Original Color image"))
        im_gray = self.resizeImage(cv2.imread(path, 0))
        self.GRAY_SHAPE = im_gray.shape
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
            windows[i] = MS_Window(i * winSize, i * winSize + winSize, i)
            i += winSize

        iterations = 0
        lockedWindows = 0
        while (iterations < maxIter):
            print ("\tIteration " + str(iterations))

            # First assign every feature a corresponding window
            for idx, i in enumerate(features):
                # For each feature, get the intensity, and map that to a window
                for win in range(numWindows):
                    if not windows[win].locked and i[1] >= windows[win].lower and i[1] < windows[win].upper:
                        # then assign the feature to the window
                        windows[win].sum += i[1]
                        windows[win].elements += 1

                        # also assign each feature to the proper window number
                        features[idx] = (features[idx][0], features[idx][1], win)
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

        # todo concatenate the windows

        '''
            Now that we have our windows, and MS is over,
            we just need to recombine everything based on euclidean distance
        '''
        newIm = np.zeros(self.CLR_SHAPE, dtype=np.uint8)  # Color the output??
        for i in features:
            x, y = i[0][0], i[0][1]
            color = windows[i[2]].color
            newIm[x][y] = color
        return newIm

    def euclideanDistance(self, a, b):
        '''
        Does what it says
        :param a: first color
        :param b: second color
        :return: euc. distance between 2 colors
        '''
        return distance.euclidean(a.getColor(), b.getColor())

    def meanShift_ClosestWindow(self, feature, e):
        """
        This is used in the vectorized function for the features
        Given a color feature, it will return the window who's mean has the closest euclidean distance
        to the given feature
        :return: the index of the window the feature belongs to
        """

        colors = np.array([i.color for j, i in self.MS_COLOR_WINDOWS.iteritems()], dtype=object)
        indexes = np.array([i.bin for j, i in self.MS_COLOR_WINDOWS.iteritems()], dtype=object)
        dists = e(a=colors, b=feature.clr)
        idx = np.argmin(dists)
        return self.MS_COLOR_WINDOWS[indexes[idx]].bin  # Return the bin that the feature is closest to.

    def meanShift_Color(self, features, winSize=50, maxIter=3, windSimilarity=3):
        '''
        This is very much like the mean shift with intensity except you are shifting towards a color
        and not an intensity. Sadly color is a 3d space, so you are moving somewhere in space... somewhere
        :param features:
        :param winSize:
        :param maxIter:
        :param windSimilarity:
        :return:
        '''
        # First instantiate each window, which is a 3D space, needs a (r,g,b) corner coord to define its loc
        combos = int(255 / winSize)  # The size of 1 side in the feature space
        combinations = []
        winHalf = int(winSize / 2)
        for i in range(combos):
            combinations.append(i * winSize)
            i += winSize
        permutations = list(itertools.product(combinations, repeat=3))

        # Create the proper number of 3d windows
        numWindows = len(permutations)
        windows = dict((i, None) for i in (range(numWindows)))
        for idx, i in enumerate(permutations):
            av = list(map(lambda x: x + winHalf, i))
            window = MS_WINDOW_COLOR(Color(av), idx)
            windows[idx] = window

        # Then start the mean shift algorithm
        iterations = 0
        lockedWindows = 0
        self.MS_COLOR_WINDOWS = windows

        # Vectorize the closestWindow function and eucDist functions
        euc = np.vectorize(self.euclideanDistance, excluded='b', cache=True)
        closestC = np.vectorize(self.meanShift_ClosestWindow, excluded='euc', cache=True)

        while (iterations < maxIter):
            print ("\tIteration " + str(iterations))
            # Trim the old windows to only have the new ones
            # windows = [self.MS_COLOR_WINDOWS[i] for i in range(len(self.MS_COLOR_WINDOWS)) if self.MS_COLOR_WINDOWS[i] != None]
            windows = dict((k, v) for k, v in windows.iteritems() if v is not None)
            self.MS_COLOR_WINDOWS = windows

            # First assign every feature a corresponding window it's closest to using vec.functions
            closestWindows = closestC(features, e=euc)

            # Then start to update each window with the colors that belong to the window
            for bin, index, feat in zip(closestWindows, range(len(features)), features):
                window = windows[bin]
                try:
                    window.sum = np.add(window.sum, feat.clr.getColor())
                except Exception:
                    print "e"
                window.elements += 1
                # update the features list
                features[index] = ColorFeature(feat.clr, feat.x, feat.y, window.bin)

            # Then recompute the means of windows and 'shift' the window
            for idx, wind in windows.iteritems():
                if wind.elements == 0 or wind == None:
                    windows[idx] = None  # Kind of like deleting the window
                    self.MS_COLOR_WINDOWS = windows
                    continue
                elif (wind.locked):
                    continue
                else:
                    w = MS_WINDOW_COLOR(wind.color, wind.bin)
                    # Compute the mean
                    mean = [int(i / wind.elements) for i in windows[idx].sum]

                    # Get the euclidean distance from the old mean to the new mean
                    dist = distance.euclidean(wind.color.getColor(), mean)

                    # If the window hasn't moved a whole lot, then lock it in
                    if dist < windSimilarity:
                        # then we have a convergence, lock the window
                        wind.locked = True
                        lockedWindows += 1
                    else:
                        w.color = Color(mean)
                        w.oldColor = wind.color
                        w.bin = wind.bin
                        windows[idx] = w

            # Convergence when all windows are no longer moving a lot
            if (lockedWindows == numWindows):
                break
            else:
                # Otherwise let's keep on going
                iterations += 1

        '''
            Now that we have our windows, and MS is over,
            we just need to recombine everything based on euclidean distance
        '''

        newIm = np.zeros(self.CLR_SHAPE, dtype=np.uint8)  # Color the output??
        for i in features:
            x, y = i.x, i.y
            newIm[x][y] = windows[i.bin].color.getColor()
        return newIm

    def getFeatures(self, im, imGray):
        '''
        This function will loop through each pixel in the image and then get the key features
        :return:
        '''
        # bins = np.histogram(im,range(256))
        intensityFeatures = []
        colorFeatures = []

        for i in range(len(im)):
            for j in range(len(im[i])):
                intensityFeatures.append(((i, j), imGray[i][j], 0))  # todo create a class for int features
                colorFeatures.append(ColorFeature(Color(im[i][j]), i, j, 0))
        if self.DEBUG:
            plt.hist(im.ravel(), 256, [0, 256]);
            plt.autoscale()
            plt.savefig("./images/histogram.png")
            hist = cv2.imread("./images/histogram.png")
            self.outputImages.append((hist, "Intensity Histogram"))

        return intensityFeatures, colorFeatures

    def intensitySegmentation(self, features):
        '''
        Using the intensity feature space, run all of the segmentation algorithms
        :param features: A list of features, 1 for each x,y coordinate
        :return: the ms, kmeans and gaus images all segmented
        '''
        # Do Mean Shift
        print "Mean shift with intensity features"
        mS_intensity = self.meanShift_Intensity(features, self.MS_WINDOW_SIZE)

        # Do kmeans

        # Do gaus

        return mS_intensity

    def colorSegmentation(self, features):
        print "Mean shift with color features"
        ms_Color = self.meanShift_Color(features)

        return ms_Color

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
        print ("Getting the image features")
        intensity_features, color_features = self.getFeatures(im_color, im_gray)

        # Then do the segmentations for the intensity features
        # print "Run intensity segmentation algorithms"
        # ms_int = self.intensitySegmentation(intensity_features)
        # self.outputImages.append((ms_int,"Mean Shift Using Intensity Feature Space"))

        # Then do segmentations for the color features
        print "Run color feature space algorithms"
        ms_color = self.colorSegmentation(color_features)
        self.outputImages.append((ms_color, "Mean Shift Using Color Feature Space"))

        # Display output images
        self.displayImages(self.outputImages)


# %pylab inline
# %matplotlib inline
plt.rcParams['figure.figsize'] = (60, 30)
c = Segment()
c.main()
