#This program provides some examples of how to check the version of
#the programs used in this repository
#Your environement might require other methods.
#Check the websites of the packages for more information.

import platform
print("python:",platform.python_version())
#import sys                     #alternative version information 
#print("Python:",sys.version_info)

import numpy
print("numpy:",numpy.version.version)

import matplotlib
print("matplotlib:",matplotlib.__version__)

import pandas as pd
print("pandas:",pd.__version__)

import scipy
print("scipy:",scipy.__version__)

import sklearn as sk
print("skearn:",sk.__version__)

import pydotplus
#dot -V from your command line will check the Graphviz version 
#Troubleshooting: try !dot -V on Google Colaboratory with the follwing output, for example:
#dot - graphviz version 2.40.1 (20161225.0304)

import keras
print("keras:",keras.__version__)

import PIL
print("PIL:",PIL.VERSION)

import cv2
print("cv2:",cv2.__version__)


