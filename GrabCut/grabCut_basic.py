#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 22:29:54 2016

@author: Hande
This is doing grabcut in a less cool way(histogram)
This one uses the grayscale intensity for the N-links weight function
"""
import warnings
import numpy as np
import cv2
import maxflow
import sys
import matplotlib.pyplot as plt
from skimage.color import rgb2grey
from math import exp
# class for histogram.
def fxn():
    warnings.warn("deprecated", DeprecationWarning)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()
        
class GrabCut:
    #foreground and background values 
    fore_ground = 1
    back_ground = 0
    #initialize step of the grabCut class
    def __init__(self,img,rec):
        self.img = img
        self.grayImg = rgb2grey(self.img)
        self.num_rows = img.shape[0]
        self.num_cols = img.shape[1]
        #parameters for the energy minimazation
        self.sigma = 10
        self.iteration = 3
        self.energy = np.infty
        self.inif = np.inf
        #the mask same size of the image indicating foreground or background
        #foreground is set to 1, background 0
        #the image layout is 
         #the rect region is set as (x,y,width,height)  
        self.mask = np.zeros(img.shape[:2],np.uint8)
        self.mask[rec[1]:rec[3],rec[0]:rec[2]] = 1
        #initialize the histograms with given mask
        self.foreHistogram,self.backHistogram = self.init_histograms(self.mask)
        #normalize the histograms
        self.normalize_histogram()    
        self.neighbor_xs = [0, 1, 1,  1,  0, -1, -1, -1]
        self.neighbor_ys = [1, 1, 0, -1, -1, -1,  0,  1]
        self.num_neighbors = len(self.neighbor_xs)
        
        #structure to store the N-links
        self.leftW = np.zeros(img.shape[:2],np.float64)
        self.upleftW = np.zeros(img.shape[:2],np.float64)
        self.upW = np.zeros(img.shape[:2],np.float64)
        self.uprightW = np.zeros(img.shape[:2],np.float64)
 
        self.compute()
        self.compute()
#        self.compute()
#        self.compute()   
        #send to the UI to handle
        testlabelmask2 = self.labelmask
        testlabelmask2 = np.int8(testlabelmask2)
#        self.displayMaskImage(self.mask,testlabelmask2)
#        
    def compute(self):
        self.foreHistogram,self.backHistogram = self.init_histograms(self.mask)
        self.normalize_histogram() 
        self.compute_labels()
        self.labelmask = self.buildGraph()
        self.mask= self.labelmask
        maskToReturn = np.int8(self.labelmask)
        return maskToReturn
    
    def get_mask(self):  
        return self.mask

    # initialize the histograms with the given mask 
    def init_histograms(self,mask):
        foreHistogram = np.zeros(shape=(16,16,16))
        backHistogram = np.zeros(shape=(16,16,16))
#        print self.img.shape[0] * self.img.shape[1]
        self.foreGroundPixels = np.transpose(np.where(mask==1))
#        print foreGroundPixels.shape
        self.backGroundPixels = np.transpose(np.where(mask==0))
#        print backGroundPixels.shape
        for i in self.foreGroundPixels:
            pixelColor = self.img[i[0],i[1]]
            r = np.rint(np.floor(pixelColor[0]/16))
            g = np.rint(np.floor(pixelColor[1]/16))
            b = np.rint(np.floor(pixelColor[2]/16))
            foreHistogram[r,g,b] = foreHistogram[r,g,b] + 1
        for j in self.backGroundPixels:
            pixelColor2 = self.img[j[0],j[1]]
            r2 = np.rint(np.floor(pixelColor2[0]/16))
            g2 = np.rint(np.floor(pixelColor2[1]/16))
            b2 = np.rint(np.floor(pixelColor2[2]/16))
            backHistogram[r2,g2,b2] = backHistogram[r2,g2,b2] + 1
        return foreHistogram,backHistogram
    
    #normalize the histogram
    def normalize_histogram(self):
        # not do this...way....class...
        self.foreHistogram = self.foreHistogram / np.count_nonzero(self.mask)
        #number of the pixels in the background histogram would just be the total number of pixels - foreGround count
        self.backHistogram = self.backHistogram / ((self.num_rows * self.num_cols) - np.count_nonzero(self.mask))
        
    def compute_labels(self):
        for i in self.foreGroundPixels:
            pixelColor = self.img[i[0],i[1]]
            r = np.rint(np.floor(pixelColor[0]/16))
            g = np.rint(np.floor(pixelColor[0]/16))
            b = np.rint(np.floor(pixelColor[0]/16))
#           case where the pixel appears with a higher chance in the foreground histogram model
            if self.foreHistogram[r,g,b] < self.backHistogram[r,g,b]:
                self.mask[i[0],i[1]] = 0
    def buildGraph(self):
#create the graph using the maxflow libaries 
        graph = maxflow.Graph[float](self.num_rows, self.num_cols)
        nodes = graph.add_grid_nodes((self.num_rows, self.num_cols))
 #        #assign T links to the clearly background pixels
        for j in self.backGroundPixels:
            graph.add_tedge(nodes[j[0],j[1]],self.inif,0)
#        for j in self.backGroundPixels:
        for i in self.foreGroundPixels:
            pixelColor = self.img[i[0],i[1]]
        ## adding in the t-links
            p_f = -np.log(self.foreHistogram[np.floor(pixelColor[0]/16),np.floor(pixelColor[1]/16),np.floor(pixelColor[2]/16)])
            p_b = -np.log(self.backHistogram[np.floor(pixelColor[0]/16),np.floor(pixelColor[1]/16),np.floor(pixelColor[2]/16)])
            graph.add_tedge(nodes[i[0],i[1]],p_f,p_b)
        ##assign N links  
        for x in range(self.num_rows):  
            for y in range(self.num_cols):
                for k in xrange(self.num_neighbors):
                    nx = x + self.neighbor_xs[k]
                    ny = y + self.neighbor_ys[k]
                    #test to see if the the pixal point is out of the picture
                    if nx < 0 or ny < 0 or nx >= self.num_rows or ny >= self.num_cols:
                        continue 
                    weight_n = exp(-((self.grayImg[nx,ny] - self.grayImg[x,y])/(2*self.sigma**2)))
                    #add in the n-links
                    graph.add_edge(nodes[x,y], nodes[nx,ny], weight_n, weight_n)
        print 'the current energy is'
        print graph.maxflow()
        label_mask = graph.get_grid_segments(nodes)
          ##return the result..
        return label_mask
        
        #only compute it once can increase performance a lot!
#    def computeNlinks(self):
        
        
    def displayMaskImage(self,mask,labelmask):
        img = self.img
        masked_img = cv2.bitwise_and(img,img,mask = labelmask)
#        masked_img2 = cv2.bitwise_and(img,img,mask = labelmask)
        plt.subplot(221)
        plt.imshow(img)
        plt.subplot(222)
        plt.imshow(mask,'gray')
        plt.subplot(223)
        plt.imshow(masked_img)
#        plt.subplot(224)
#        plt.imshow(masked_img2)
        plt.show()
            
def main():  
        #Test the mask 
        img2 = cv2.imread('rose.bmp')
#        img2 = cv2.imread('monkey.jpg')
        b,g,r =cv2.split(img2)
        img2 = cv2.merge([r,g,b])
#        rect = (70,20,400,250)
        GrabCut(img2,rect)
        
#main()
# At first, in input window, draw a rectangle around the object using
# mouse right button. Then press 'n' to segment the object (once or a few times)
# For any finer touch-ups, you can press any of the keys below and draw lines on
# the areas you want. Then again press 'n' for updating the output.
# Key '0' - To select areas of sure background
# Key '1' - To select areas of sure foreground
# Key '2' - To select areas of probable background
# Key '3' - To select areas of probable foreground
# Key 'n' - To update the segmentation
# Key 'r' - To reset the setup
# Key 's' - To save the results       
        
#UI interface 
BLUE = [255,0,0]        # rectangle color
RED = [0,0,255]         # PR BG
GREEN = [0,255,0]       # PR FG
BLACK = [0,0,0]         # sure BG
WHITE = [255,255,255]   # sure FG

DRAW_BG = {'color' : BLACK, 'val' : 0}
DRAW_FG = {'color' : WHITE, 'val' : 1}
DRAW_PR_FG = {'color' : GREEN, 'val' : 3}
DRAW_PR_BG = {'color' : RED, 'val' : 2}

# setting up flags
rect = (0,0,1,1)
drawing = False         # flag for drawing curves
rectangle = False       # flag for drawing rect
rect_over = False       # flag to check if rect drawn
rect_or_mask = 100      # flag for selecting rect or mask mode
value = DRAW_FG         # drawing initialized to FG
thickness = 3           # brush thickness

def onmouse(event,x,y,flags,param):
    global img,img2,drawing,value,mask,rectangle,rect,rect_or_mask,ix,iy,rect_over

    # Draw Rectangle
    if event == cv2.EVENT_RBUTTONDOWN:
        rectangle = True
        ix,iy = x,y

    elif event == cv2.EVENT_MOUSEMOVE:
        if rectangle == True:
            img = img2.copy()
            cv2.rectangle(img,(ix,iy),(x,y),BLUE,2)
            rect = (min(ix,x),min(iy,y),abs(ix-x),abs(iy-y))
            rect_or_mask = 0

    elif event == cv2.EVENT_RBUTTONUP:
        rectangle = False
        rect_over = True
        cv2.rectangle(img,(ix,iy),(x,y),BLUE,2)
        rect = (min(ix,x),min(iy,y),abs(ix-x),abs(iy-y))
        rect_or_mask = 0
        print(" Now press the key 'n' a few times until no further change \n")

    # draw touchup curves

    if event == cv2.EVENT_LBUTTONDOWN:
        if rect_over == False:
            print("first draw rectangle \n")
        else:
            drawing = True
            cv2.circle(img,(x,y),thickness,value['color'],-1)
            cv2.circle(mask,(x,y),thickness,value['val'],-1)

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            cv2.circle(img,(x,y),thickness,value['color'],-1)
            cv2.circle(mask,(x,y),thickness,value['val'],-1)

    elif event == cv2.EVENT_LBUTTONUP:
        if drawing == True:
            drawing = False
            cv2.circle(img,(x,y),thickness,value['color'],-1)
            cv2.circle(mask,(x,y),thickness,value['val'],-1)

if __name__ == '__main__':

    # print documentation
    print(__doc__)

    # Loading images
    if len(sys.argv) == 2:
        filename = sys.argv[1] # for drawing purposes
    else:
        print("No input image given, so loading default image, ../data/lena.jpg \n")
        print("Correct Usage: python grabcut.py <filename> \n")
        filename = 'rose.bmp'

    img = cv2.imread(filename)
    img2 = img.copy()                               # a copy of original image
    mask = np.zeros(img.shape[:2],dtype = np.uint8) # mask initialized to PR_BG
    output = np.zeros(img.shape,np.uint8)           # output image to be shown

    # input and output windows
    cv2.namedWindow('output')
    cv2.namedWindow('input')
    cv2.setMouseCallback('input',onmouse)
    cv2.moveWindow('input',img.shape[1]+10,90)

    print(" Instructions: \n")
    print(" Draw a rectangle around the object using right mouse button \n")

    while(1):

        cv2.imshow('output',output)
        cv2.imshow('input',img)
        k = 0xFF & cv2.waitKey(1)

        # key bindings
        if k == 27:         # esc to exit
            break
        elif k == ord('0'): # BG drawing
            print(" mark background regions with left mouse button \n")
            value = DRAW_BG
        elif k == ord('1'): # FG drawing
            print(" mark foreground regions with left mouse button \n")
            value = DRAW_FG
        elif k == ord('2'): # PR_BG drawing
            value = DRAW_PR_BG
        elif k == ord('3'): # PR_FG drawing
            value = DRAW_PR_FG
        elif k == ord('s'): # save image
            bar = np.zeros((img.shape[0],5,3),np.uint8)
            res = np.hstack((img2,bar,img,bar,output))
            cv2.imwrite('grabcut_output.png',res)
            print(" Result saved as image \n")
        elif k == ord('r'): # reset everything
            print("resetting \n")
            rect = (0,0,1,1)
            drawing = False
            rectangle = False
            rect_or_mask = 100
            rect_over = False
            value = DRAW_FG
            img = img2.copy()
            mask = np.zeros(img.shape[:2],dtype = np.uint8) # mask initialized to PR_BG
            output = np.zeros(img.shape,np.uint8)           # output image to be shown
        elif k == ord('n'): # segment the image
            print(""" For finer touchups, mark foreground and background after pressing keys 0-3
            and again press 'n' \n""")
            myGrabCut = GrabCut(img,rect)
            mask = myGrabCut.compute()
        output = cv2.bitwise_and(img,img,mask=mask)

    cv2.destroyAllWindows()

    
#main()
#UI()
