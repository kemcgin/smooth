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

# create a test matrix nxn, 1 - n^2
def testMatrix(n):
  t = (n*n)
  m = np.arange(t)
  m.shape=(-1,n)
  
  return m
  
#takes a matrix, and the sigma value for the submatrix,
#then returns an nxn matrix of kernal matrices
def kernal(m, sigma):
  if  (sigma % 2 == 0):
    sys.exit("sigma must be odd number")

  th = int(np.floor(sigma / 2)) #threshold is the floor of have the number (assuming odd number)
  res = np.empty(np.shape(m), dtype=object)

  #iterate over each element and get its kernal, k
  for idn, val in np.ndenumerate(m):
    x = idn[0]
    y = idn[1]

    k = m[max(0, y-th) : y + (th+1), max(0, x-th) : x + (th+1)]
#    print(x,y)
#    print(k)
    res[x,y] = k

  return res


#takes in a matrix, and a sigma value, then returns a matrix averaged over that kernal
def blockFilter(m, sigma):
  k = kernal(m, sigma)
  mat = k[1,1]
  avg = np.average(mat)
  res = np.empty(np.shape(m))
  
  for idn, val in np.ndenumerate(k):
    x = idn[0]
    y = idn[1]
  
    res[x,y] = np.average(k[x,y])
 
  
  return(res)



def main():
#  print(input_img.shape)

#  resized = resize(input_img, 900)
#  disp(resized)


#  cv2.destroyAllWindows()

  m = testMatrix(10)
#  k = kernal(m, 5)
  blockFilter(m, 5)
  print (m)
  

  exit 




main()
