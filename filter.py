#!/usr/bin/env python3


import sys
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import math
import cv2

filename = str(sys.argv[1])
input_img = cv2.imread(filename,1)


#prints entire matrices without truncation
#np.set_printoptions(threshold=np.inf)

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



#takes a matrix, and the sigma value for the submatrix,
#then returns an nxn matrix of kernal matrices
def kernal(m, sigma):
  if  (sigma % 2 == 0):
    sys.exit("sigma must be odd number")

  th = int(np.floor(sigma / 2)) #threshold is the floor of have the number (assuming odd number)
  res = np.empty(np.shape(m), dtype=object)


  _,_,z = np.shape(m)
#  z = z - 1

  #for each layer in the matrix
  for i in range (0,z):
    #iterate over each element and get its kernal, k
    for idn, val in np.ndenumerate(m):
      x = idn[0]
      y = idn[1]
            

      k = m[max(0, x-th) : x + (th+1), max(0, y-th) : y + (th+1), i]
      #    print(x,y)
      #    print(k)
      res[x,y,i] = k

  return res


#takes in a matrix, and a sigma value, then returns a matrix averaged over that kernal
def boxFilter(m, sigma):
  

  k = kernal(m, sigma)
#  mat = k[1,1]
#  avg = np.average(mat)
  res = np.empty(np.shape(m))

  _,_,z = np.shape(k)

  for i in range (0,z):
    for idn, val in np.ndenumerate(k):
      x = idn[0]
      y = idn[1]
  
      res[x,y,i] = np.average(k[x,y,i])
 
  # 255 for rgb 
#  res = res / 255   
  return(res)

# creates a box filter using intensity component of lab colorspace
def boxFilterColor(m, sigma):
  #create copy of matrix
  c = m

  #confirm its a color image
  if(m.ndim != 3):
    sys.exit("not a color image")
  
  x, y, z = np.shape(c)

  print(x,y,z)

  intensity = np.empty(x,y)
  intensity = z[0]



#def intensity_matrix(m):
#  x, y , z = np.shape(m)
#  res = np.zeros(x,y)
#  for x in 


def saveTxt(filename, matrix):
  file = open(filename, "w+")
  content = str(matrix)
  file.write(content)
  file.close()


def main():
#  print(input_img.shape)
#  resized = resize(input_img, 900)
#  disp(resized)
#  print_bw(input_img)

  bw = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
  lab = cv2.cvtColor(input_img, cv2.COLOR_BGR2Lab)


#  print(bw)
#  newM = boxFilter(lab, 101)
#  newM = boxFilterColor(lab, 15)
  newM = boxFilter(lab, 25)


# these are for converting back to BRG color, since the dtype when i 
# manipulate the matrices seems to change to float64, which apparently
# opencv can't handle
#  orig = cv2.cvtColor(newM.astype('float32'), cv2.COLOR_Lab2BGR)
#  orig = cv2.cvtColor(newM.astype('uint8') * 255, cv2.COLOR_Lab2BGR)


#  newM = newM.astype(int)
#  k = kernal(input_img, 5)
#  print(np.shape(newM)) 

#  saveTxt("og.txt", k[:,:,2])
#  saveTxt("new.txt",newM)
 
#  print(input_img[200,200,0]) 
#  print(k[200,200,0])
#  print(np.shape(newM)) 
 
#  disp(bw) 
  disp(orig)

  cv2.destroyAllWindows()
  exit 



main()
