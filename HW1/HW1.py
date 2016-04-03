import numpy as np
import copy
import cv2
from matplotlib import pyplot as plt
from scipy import signal
import tkinter as tk
from tkinter import filedialog
import sys
from PIL import Image

def selectfile():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename()
    return file_path

def getCount():
    return int(input('Enter parts number (2 - 4): '))


BLUE = [255,0,0]        # part 1
RED = [0,0,255]         # part 2
GREEN = [0,255,0]       # part 3
YELLOW = [0,255,255]    # part 4

DRAW_P1 = {'color' : BLUE, 'val' : 1}
DRAW_P2 = {'color' : RED, 'val' : 1}
DRAW_P3 = {'color' : GREEN, 'val' : 1}
DRAW_P4 = {'color' : YELLOW, 'val' : 1}

drawing = False         # flag for drawing curves
thickness = 3           # brush thickness
value = DRAW_P1

def onmouse(event,x,y,flags,param):
    global imgInput,img2,drawing,value,mask
        
    # draw touchup curves
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        cv2.circle(imgInput,(x,y),thickness, value['color'],-1)
        cv2.circle(mask,(x,y),thickness, value['val'],-1)

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            cv2.circle(imgInput,(x,y), thickness, value['color'],-1)
            cv2.circle(mask,(x,y), thickness, value['val'],-1)

    elif event == cv2.EVENT_LBUTTONUP:
        if drawing == True:
            drawing = False
            cv2.circle(imgInput,(x,y), thickness, value['color'],-1)
            cv2.circle(mask,(x,y), thickness, value['val'],-1)

def getUserInput(image, count, masks):  # Get parts cound and user-marked masks
    global value, mask
    cv2.namedWindow('input')
    #cv2.moveWindow('input', 0, 0)
    bgdmodel = np.zeros((1,65),np.float64)
    fgdmodel = np.zeros((1,65),np.float64)
    
    cv2.setMouseCallback('input', onmouse)
    for i in range(0, count):
        # first - get the ALREADY filled mask, from the previous user attempt
        mask = copy.deepcopy(masks[i]);
        if i == 0:
            value = DRAW_P1
        elif i == 1:
            value = DRAW_P2
        elif i == 2:
            value = DRAW_P3
        elif i == 3:
            value = DRAW_P4

        print(" mark part %d regions with left mouse button and after press 'n'\n" % (i + 1))
        while(1):
            k = 0xFF & cv2.waitKey(1)
            
            cv2.imshow('input',image)
            # key bindings
            if k == 27:    # esc to exit
                cv2.destroyAllWindows()
                setStop()
                return
            elif k == ord('n'): # segment the image
                masks[i] = mask
                #mask = 2+np.zeros(image.shape[:2],dtype = np.uint8)
                #plt.imshow(masks[i]),plt.colorbar(),plt.show()
                break   
    cv2.destroyAllWindows()          
  
                  
def grabImagesZ(image, masks, count):  # Get 4 masks, the image should be divided into 4 parts
    grabMasks = []
    for i in range(0, count): 
        bgdmodel = np.zeros((1,65),np.float64)
        fgdmodel = np.zeros((1,65),np.float64)
        (grabMask, bgdModel, fgdModel) = cv2.grabCut(image,masks[i],None,bgdmodel,fgdmodel,1,cv2.GC_INIT_WITH_MASK)
        grabMasks.append(grabMask)
        #plt.imshow(grabMask),plt.colorbar(),plt.show()
        #plt.imshow(masks[i]),plt.colorbar(),plt.show()

    #set 1 for foreground and possible foreground
    #set 0 fro background and possible background
    for i in range (0, count):
        grabMasks[i] = np.where((grabMasks[i]==2)|(grabMasks[i]==0),0,1).astype('uint8')
        #plt.imshow(grabMasks[i]),plt.colorbar(),plt.show()

    for i in range (1, count):
        for j in range (0, i):
            grabMasks[i] = np.where((grabMasks[j]==1),0, grabMasks[i]).astype('uint8')
        #plt.imshow(grabMasks[i]),plt.colorbar(),plt.show()

    totalMask = np.where(grabMasks[0] == 1, 0, 0)
    for i in range(0, count -1):
        totalMask = np.where(grabMasks[i] == 1, 1, totalMask)

    grabMasks[count - 1] = np.where(totalMask == 1, 0, 1)
    return grabMasks

def calcBorderForMask(grabMask):  
    (height, width) = grabMask.shape

    for j in range(1, width-2):
        for i in range(1, height-2):
            if grabMask[i][j] != 0:
                if grabMask[i-1][j] == 0 or grabMask[i][j-1] == 0 or grabMask[i + 1][j] == 0 or grabMask[i][j+1] == 0:
                    grabMask[i][j] = 255     
    for i in range(0, height-1):
        for j in range(0, width-1):
            if grabMask[i][j] != 255:
                grabMask[i][j] = 0
            else:
                grabMask[i][j] = 1  
    return  grabMask

def calcBorderForMaskZ(grabMask):  
    conv = scharr = np.array([[-1, -1, -1],
                              [-1, 8, -1],
                              [-1, -1, -1]])
    result = signal.convolve2d(grabMask, conv, 'same')
    result = np.where(result < 0, 0, result)
    result = np.where(result > 0, 1, result)

    #plt.imshow(result),plt.colorbar(),plt.show()
    return result




def calculateBorderMask(grabMasks, count): # Calculkate borders from previous masks
    borderMasks = []
    for i in range(0, count):
        borderMask = calcBorderForMaskZ(grabMasks[i])
        borderMasks.append(borderMask)
    return borderMasks

def drawBorder(image, value, mask):
    (height, width) = mask.shape
    for row in range(0, height):
        for col in range(0, width):
            if mask[row][col] == 1:
                image[row][col] = value['color']
    return image

def showResult(image, borderMask, count):  # draw borders on the image
    global value
    im = copy.deepcopy(image);
    for i in range(0, count):
        if i == 0:
            value = DRAW_P1
        elif i == 1:
            value = DRAW_P2
        elif i == 2:
            value = DRAW_P3
        elif i == 3:
            value = DRAW_P4
        im = drawBorder(im, value, borderMask[i])
    plt.imshow(im),plt.colorbar(),plt.show()
    cv2.namedWindow('output')
    cv2.imshow('output',im)


def shouldStop():
    global stopFlag
    return stopFlag

def setStop():
    global stopFlag;
    stopFlag = 1

filename = selectfile()
imgInput = cv2.imread(filename)
imgInputOriginal = copy.deepcopy(imgInput)
count = getCount()
masks = []
stopFlag = 0;
for i in range(0, count):
    mask = 2+np.zeros(imgInput.shape[:2],dtype = np.uint8)
    masks.append(mask)
while(1):
    getUserInput(imgInput, count, masks)  # Get parts cound and user-marked masks
    if shouldStop() == 1:
        break
    #plt.imshow(imgInputOriginal),plt.colorbar(),plt.show()
    masksCopy = masks 
    #for i in range(0, count):
    #    plt.imshow(masks[i]),plt.colorbar(),plt.show()
    grabMasks = grabImagesZ(imgInputOriginal, masks, count) # Get 4 masks, the image should be divided into 4 parts
   # for i in range(0, count):
    #    plt.imshow(grabMasks[i]),plt.colorbar(),plt.show()


    borderMask = calculateBorderMask(grabMasks, count) # Calculkate borders from previous masks
    showResult(imgInputOriginal, borderMask, count) # draw borders on the image


    #print("Press n to continue, Esc to exit\n") 
    #k = 0xFF & cv2.waitKey(1)
    #if k == 27:    # esc to exit
     #   break
#cv2.destroyAllWindows()
