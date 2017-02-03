import cv2 as cv2
import numpy as np

from Assignment1.pictureLogger import imageLogger

il = imageLogger()


def main():
    im = cv2.imread("./images/Rebecca1.jpg")
    il.log(im,"original")

     #Resize the image
    # im = resize(im,10)
    # il.log(im,"resized")

    #Shift the image
    # im = shiftImage(im)
    # il.log(im,"Shifted")

def resize(img,scale):
    shp = img.shape
    return cv2.resize(img,((shp[1]/scale),(shp[0]/scale)),interpolation=cv2.INTER_CUBIC)

def shiftImage(img):
    #Create the transformation matrix,
    tx, ty = 100, 50 #The shift values
    trans = np.array([[1,0,tx],[1,0,ty]],dtype=np.float32)
    dsz = img.shape
    return cv2.warpAffine(img, trans,(dsz[1],dsz[0]))

def rotateImage(img):
    #Will need the coordinates of 3 points we want to place in original image to trans imaged
    inputs = np.float23([[100,200],[100,300],[100,400]])

def doGradients(img):
    return cv2.Sobel(img,cv2.CV_64F)


if __name__ == '__main__':
    main()