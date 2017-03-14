import cv2
from matplotlib import pyplot as plt
import os
import numpy as np


class bWords:
    DEBUG = True

    def resizeImage(self, image, dbug=DEBUG):
        if dbug:
            image = cv2.resize(image, (120, 160))
        else:
            image = cv2.resize(image, (2400, 1200))
        return image

    def displayImage(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img)

    def displayImages(self, images):
        print("Displaying outs")
        for im in images:
            ima = im[0]
            ima = cv2.cvtColor(ima, cv2.COLOR_BGR2RGB)
            plt.figure()
            plt.title(im[1], fontsize=self.FONT_SIZE, fontweight='bold', color="green")
            plt.imshow(ima)

    def importImages(self, path):
        for i in os.walk(path):
            print i

    def main(self):
        pathToImages = "./images/101_ObjectCategories"

        print "Importing the images"
        self.importImages()


# $pylab
# %matplotlib inline
plt.rcParams['figure.figsize'] = (20, 60.0)
b = bWords()
b.main()
