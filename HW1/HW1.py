import numpy as np
import copy
import cv2
from matplotlib import pyplot as plt
from scipy import signal
import tkinter as tk
from tkinter import filedialog
import sys
from PIL import Image

#
# Global variables
#
BLUE = [255,0,0]        # part 1
GREEN = [0,255,0]       # part 2
RED = [0,0,255]         # part 3
YELLOW = [0,255,255]    # part 4
DRAW_P1 = {'color' : BLUE, 'val' : 1, 'colorname' : 'BLUE'}
DRAW_P2 = {'color' : GREEN, 'val' : 1, 'colorname' : 'GREEN'}
DRAW_P3 = {'color' : RED, 'val' : 1, 'colorname' : 'RED'}
DRAW_P4 = {'color' : YELLOW, 'val' : 1, 'colorname' : 'YELLOW'}

drawing = False         # flag for drawing curves
thickness = 3           # brush thickness
value = DRAW_P1
masks = []
stopFlag = 0
DEBUG = 0
#
# Functions below
#

#
# A simple function to select an input file
#
def selectfile():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename()
    return file_path

#
# A simple functin to get from the user the number of partitions
#
def getCount():
    return int(input('Enter parts number (2 - 4): '))

#
# Mouse click listener.  Used to listen to clicks in the input window
#
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

#
# A function to process masks and ensure that pixels selected in one mask
# surely marked as BACKGROUOND in all other masks
#
def prepareMasks(masks):
    preparedMasks = []
    for i in range(0, count):
        preparedMask = copy.deepcopy(masks[i])
        for j in range(0, count):
            if i != j:
                preparedMask = np.where((masks[j] == 1),0,preparedMask).astype('uint8')
        preparedMasks.append(preparedMask)
    return preparedMasks

#
# A function, which displays the original image and previously marked user masks
# enables a user to refine masks by adding more points in the same order.
#
def getUserInput(image, previousResult, count, masks):  # Get parts count and user-marked masks
    global value, mask
    showImageCV2(previousResult, 'Segmented so far')

    inputWindowTitle = 'Use mouse to mark segments'
    cv2.namedWindow(inputWindowTitle)
    bgdmodel = np.zeros((1,65),np.float64)
    fgdmodel = np.zeros((1,65),np.float64)
    
    cv2.setMouseCallback(inputWindowTitle, onmouse)
    for i in range(0, count):
        # first - get the ALREADY filled mask, from the previous user attempt
        mask = copy.deepcopy(masks[i])
        if i == 0:
            value = DRAW_P1
        elif i == 1:
            value = DRAW_P2
        elif i == 2:
            value = DRAW_P3
        elif i == 3:
            value = DRAW_P4

        print("Mark part %d %s regions with left mouse button and after press 'n'\n" % ((i + 1), value['colorname']))
        while(1):
            k = 0xFF & cv2.waitKey(1)
            
            cv2.imshow(inputWindowTitle,image)
            if k == 27:    # esc to exit
                cv2.destroyAllWindows()
                setStop()
                return
            elif k == ord('n'): # segment the image
                masks[i] = mask
                showImageDebug(masks[i], 'user masks after selection')
                break  

    preparedMasks = prepareMasks(masks)
    cv2.destroyAllWindows()  
    return preparedMasks        
   
#
#  A function which actually calls the grabCut()
#  returns masks for each area
#  The last mask contains all the unclaimed area!
#
def grabImages(image, masks, count):  # Get 4 masks, the image should be divided into 4 parts
    grabMasks = []
    for i in range(0, count): 
        bgdmodel = np.zeros((1,65),np.float64)
        fgdmodel = np.zeros((1,65),np.float64)
        (grabMask, bgdModel, fgdModel) = cv2.grabCut(image,masks[i], None, bgdmodel, fgdmodel, 1, cv2.GC_INIT_WITH_MASK)
        grabMasks.append(grabMask)
        showImageDebug(grabMask, 'grabmask ' + str(i))

    #
    # Normalization - remove possibly FG and BG
    # set 1 for foreground and possible foreground and 0 for background and
    # possible background
    #
    for i in range(0, count):
        grabMasks[i] = np.where((grabMasks[i] == 2) | (grabMasks[i] == 0), 0, 1).astype('uint8')
        showImageDebug(grabMasks[i], 'grabmask normalized ' + str(i))

    #
    # Remove conflicts in the resulting masks.  priority order is FIFO
    #
    for i in range(1, count):
        for j in range(0, i):
            grabMasks[i] = np.where((grabMasks[j] == 1), 0, grabMasks[i]).astype('uint8')
        showImageDebug(grabMasks[i], 'grabmask with no conflicts ' + str(i))

    #
    # Prepare the last mask to include all area not included in the previous
    # masks
    #
    totalMask = np.where(grabMasks[0] == 1, 0, 0)
    for i in range(0, count - 1):
        totalMask = np.where(grabMasks[i] == 1, 1, totalMask)
    grabMasks[count - 1] = np.where(totalMask == 1, 0, 1)

    return grabMasks

#
#  This function calculates mask for area border. Mask thickness is 2 pixels
#
def calcBorderForMaskZ(grabMask):  
    #conv = np.array([[-1, -1, -1],
    #                 [-1, 8, -1],
    #                 [-1, -1, -1]])
    conv = np.array([[-1, -1, -1, -1, -1],
                     [-1, -1, -1, -1, -1],
                     [-1, -1, 16, -1, -1],
                     [-1, -1, -1, -1, -1],
                     [-1, -1, -1, -1, -1]])

    result = signal.convolve2d(grabMask, conv, 'same')
    result = np.where(result < 0, 0, result)
    result = np.where(result > 0, 1, result)
    showImageDebug(result, 'border mask')
    return result

#
# Calculate border masks for each area and return them
#
def calculateBorderMasks(grabMasks, count):
    borderMasks = []
    for i in range(0, count):
        borderMask = calcBorderForMaskZ(grabMasks[i])
        borderMasks.append(borderMask)
    return borderMasks

#
# Draw a border on the given image according to the given mask
# value is the border color
#
def drawBorder(image, value, mask):
    (height, width) = mask.shape
    for row in range(0, height):
        for col in range(0, width):
            if mask[row][col] == 1:
                image[row][col] = value['color']
    return image

#
# Will display the image ALLOWING TO SAVE IT
#
def showImage(image, title):
    fig = plt.gcf()
    fig.canvas.set_window_title(title + '. Use Save As to save the image')
    plt.imshow(image),plt.colorbar(),plt.show() 

def showImageRGB(image, title):
    fig = plt.gcf()
    fig.canvas.set_window_title(title + '. Use Save As to save the image')
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)),plt.colorbar(),plt.show() 

#
# Will display the image ALLOWING TO SAVE IT only if DEBUG != 0
#
def showImageDebug(image, title):
    if DEBUG != 0:
        fig = plt.gcf()
        fig.canvas.set_window_title(title + '. Use Save As to save the image')
        plt.imshow(image),plt.colorbar(),plt.show() 

def showImageCV2(image, title):
    cv2.namedWindow(title);
    cv2.imshow(title, image);

#
#
#
def showMasksTogether(masksToGlue):
    result = np.where(masksToGlue[0] == 0, 0, 0)
    for i in range (0, count):
        result = np.where(masksToGlue[i] == 1, i, result)
    showImage(result, 'All masks together')


#
# Display the image with bordered areas. It is possible to Save As the image
#
def calculateSegmentedImage(image, borderMask, count):
    global value
    for i in range(0, count):
        if i == 0:
            value = DRAW_P1
        elif i == 1:
            value = DRAW_P2
        elif i == 2:
            value = DRAW_P3
        elif i == 3:
            value = DRAW_P4
        im = drawBorder(image, value, borderMask[i])
        color = i
    return im


#
# Queries the stop flag, which indicates whether we have to run another iteration
#
def shouldStop():
    global stopFlag
    return stopFlag

#
# Sets the stop flag
#
def setStop():
    global stopFlag
    stopFlag = 1


##################################################################
#
#        PROGRAM START
#
##################################################################

filename = selectfile()
imgInput = cv2.imread(filename)
imgInputOriginal = copy.deepcopy(imgInput)
imBorders = copy.deepcopy(imgInput)
count = getCount()

#
# Init the masks array to Possible Foreground
#
for i in range(0, count):
    mask = 3 + np.zeros(imgInput.shape[:2], dtype = np.uint8)
    masks.append(mask)

#
# Start the iterative process
#
while(1):

    #
    # Read user mouse input (including the previous input)
    #
    preparedMasks = getUserInput(imgInput, imBorders, count, masks)
    if shouldStop() == 1:
        break

    #
    # Grab areas
    #
    grabMasks = grabImages(imgInputOriginal, preparedMasks, count)
    for i in range(0, count):
        showImageDebug(grabMasks[i], 'Mask number ' + str(i))

    #
    # Show glued mask
    #
    showMasksTogether(grabMasks)

    #
    # Calculate area borders
    #
    borderMask = calculateBorderMasks(grabMasks, count) 

    #
    # Draw borders and display the result
    #
    imageToDisplay = copy.deepcopy(imgInputOriginal)
    imBorders = calculateSegmentedImage(imageToDisplay, borderMask, count)
    showImageRGB(imBorders, 'Segmentation result')
    
    print("Press n to continue, Esc to exit\n") 
    k = 0xFF & cv2.waitKey(1)
    if k == 27:    # esc to exit
        break
cv2.destroyAllWindows()
