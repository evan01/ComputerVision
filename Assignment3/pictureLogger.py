'''
    This class is a tool that lets you see the history of images as you apply
    transformations to them.

    IT IS SO UNBELIEVABLY USEFUL FOR COMPUTER VISION IT CAN'T BE IGNORED

'''
import logging
from logging import FileHandler

import imutils
from vlogging import VisualRecord

LOGGING_ON = True


class imageLogger():
    '''
        Class that enables you to log images. To use this class just import it, instantiate it 'imageLogger()'\n
        and then use the il.log(image,**args)
    '''

    def __init__(self):
        url = "Logging.html"
        self.url = url
        self.logger = logging.getLogger(url)
        if LOGGING_ON:
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.INFO)
        fh = FileHandler("Logging.html", mode="w")
        self.logger.addHandler(fh)

    def log(self, image, desc="no description", fmt="png", size=1600, title="Logging"):
        '''
        Takes as input a raw image and logs it to an html file for easy viewing
        'image':The raw image
        'desc':A description of the image (optional)
        'fmt':The format to output the image (optional)
        'size':The size to output the image to (optional)
        'title':The title of the image
        '''
        if LOGGING_ON:
            try:
                image = imutils.resize(image, width=size)
                self.logger.debug(VisualRecord(title, image, desc, fmt=fmt))
            except Exception as e:
                print (e)
                print ("catch")

    if __name__ == '__main__':
        print("logging")
