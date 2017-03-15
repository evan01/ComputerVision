import cv2
from matplotlib import pyplot as plt
import os
from collections import defaultdict
import numpy as np
import itertools



class bWords:
    DEBUG = True
    SKIPIMPORT = True  # Means to only import a minimal ammount of images
    IMAGES = defaultdict()

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
        # for each category, we need to get all the pictures associated with it
        categories = [i[0] for i in os.walk(path)][1:]
        categories = categories[int(len(categories) / 2):]
        cs = [i.rsplit('/', 1)[1] for i in categories]

        # Create a dictionary where we have category as dict key, returning the array of related images
        for i in range(len(categories)):
            imNames = [j[2] for j in os.walk(categories[i] + "/")][0]
            if self.SKIPIMPORT and len(imNames):
                # No point importing ALL PICTURES when testing...
                if (i % 2 == 0 or i % 5 == 0):
                    continue
                imNames = imNames[:1]

            print ("\t importing: " + "(" + str(len(imNames)) + ") " + str(cs[i]) + ":"),
            importedImages = []
            for idx, image in enumerate(imNames):
                if (idx % 10 == 0):
                    print ('.'),
                # import each image and categorize it
                im = cv2.imread(categories[i] + "/" + image)
                # Resize each image for your sanity
                im = self.resizeImage(im)

                # Compute the sift features for each image
                s = cv2.xfeatures2d.SIFT_create()
                sifts = s.detectAndCompute(im, None)

                # Append imported images to list
                imageAndFeatures = dict({'image': im, 'sift': sifts})
                importedImages.append(imageAndFeatures)
            print ""
            self.IMAGES[cs[i]] = importedImages

    def kmeans(self):
        pass
    def main(self):
        pathToImages = "./images/101_ObjectCategories"

        print "Importing the images"
        self.importImages(pathToImages)

        print "For each image, run kmeans"
        self.kmeans()


# $pylab
# %matplotlib inline
plt.rcParams['figure.figsize'] = (20, 60.0)
b = bWords()
b.main()
