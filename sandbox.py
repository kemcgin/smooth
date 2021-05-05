#!/usr/bin/env python3


import sys
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import math
import cv2

filename = str(sys.argv[1])
input_img = cv2.imread(filename,1)

#cv2.imshow('image', input_img)
#cv2.waitKey(0)

#print (input_img)


def print_bw(input_img):
  output = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
  cv2.imshow('image', output)
  cv2.waitKey(0)

def disp(img):
  cv2.imshow('image', img)
  cv2.waitKey(0)

  
# resizes image so that its width is the second arg, and its height is kept proportional     
def resize(img, w):
  original_width = img.shape[1]
  original_height = img.shape[0]
  scale_ratio = float(w / original_width)
  h = int(original_height * scale_ratio)
  dims = (w, h)
  print (original_width)
  print (scale_ratio)
  print(dims)
  return cv2.resize(img, dims, interpolation = cv2.INTER_AREA)




def main():
#  print(input_img.shape)
  resized = resize(input_img, 900)
  disp(resized)


  cv2.destroyAllWindows()
  exit 




main()
