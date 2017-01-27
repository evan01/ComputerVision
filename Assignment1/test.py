import cv2
import numpy as np
from lib.pictureLogger import imageLogger

il = imageLogger()
def main():
    iris = cv2.imread("./images/Iris.png")
    il.log(iris,"iris")

if __name__ == '__main__':
    main()