# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 16:02:33 2018

@author: clausmichele
"""
import os
import numpy as np
import cv2
import glob
from PIL import Image
 
def main():
  
  src_dir='./data/train2'
  save_dir = './data/train2'
  src_dir_test='./data/test2'
  save_dir_test = './data/test2'
  
  filepaths = glob.glob(src_dir + '/*.jpg')
  filepaths_test = glob.glob(src_dir_test + '/*.jpg')
  def sortKeyFunc(s):
    return int(os.path.basename(s)[:-4])
    
  filepaths_test.sort(key=sortKeyFunc)
  filepaths.sort(key=sortKeyFunc)
  
  
  print("[*] Reading train files...")  
  
  # if not os.path.exists(save_dir):
  #       os.mkdir(save_dir)
  #       os.mkdir(save_dir_test)
  #       os.mkdir('./data/train2/noisy')
  #       os.mkdir('./data/train2/original')
  #       os.mkdir('./data/test2/noisy')
  #       os.mkdir('./data/test2/original')

  print("[*] Applying noise...")

  sig = np.linspace(0,50,len(filepaths))
  np.random.shuffle(sig)
  sig_test = np.linspace(0,50,len(filepaths_test))
  np.random.shuffle(sig_test)

  for i in range(len(filepaths)):
      # im = Image.open(filepaths[i])
      #
      # if im.width > im.height:
      #     im.rotate(90).show()
      #     image1 = img.transpose(Image.ROTATE_90)
      #     image1.save(filepaths[i])

      image = cv2.imread(filepaths[i],cv2.IMREAD_GRAYSCALE)
      image = cv2.resize(image,(180,180), interpolation = cv2.INTER_CUBIC)
      # row, col, ch = image.shape
      row, col= image.shape
      # row1, col1 = row, col
      # if row < col:
      #   image1 = image.transpose(Image.ROTATE_90)
      # else:
      #   image1 = image
      mean = 0
      sigma = sig[i]
      # gauss = np.random.normal(mean,sigma,(row,col,ch))
      # gauss = gauss.reshape(row,col,ch)
      gauss = np.random.normal(mean, sigma, (row, col))
      gauss = gauss.reshape(row, col)
      noisy = image + gauss
      noisy = np.clip(noisy, 0, 255)
      noisy = noisy.astype('uint8')
      cv2.imwrite(os.path.join(save_dir, "noisy/%04d.png" %i), noisy)
      cv2.imwrite(os.path.join(save_dir, "original/%04d.png" %i), image)
        
  for i in range(len(filepaths_test)):
        image = cv2.imread(filepaths_test[i],cv2.IMREAD_GRAYSCALE)
        # image = cv2.resize(image,(180,180), interpolation = cv2.INTER_CUBIC)
        row,col,ch = image.shape
        mean = 0
        sigma = sig[i]
        gauss = np.random.normal(mean,sigma,(row,col,ch))
        gauss = gauss.reshape(row,col,ch)
        noisy = image + gauss
        noisy = np.clip(noisy, 0, 255)
        noisy = noisy.astype('uint8')
        cv2.imwrite(os.path.join(save_dir_test, "noisy/%d.png" %i), noisy)
        cv2.imwrite(os.path.join(save_dir_test, "original/%d.png" %i), image)
  
  print("[*] Noisy and original images saved")

if __name__ == "__main__":
 main()