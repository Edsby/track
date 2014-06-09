# Code based off of https://github.com/dereks/motion_tracking/blob/master/track.py

import cv
import time
import numpy

# Initialize camera source
capture = cv.CaptureFromCAM(0)

# Get an image and start initializing
frame = cv.QueryFrame(capture)
frameSize = cv.GetSize(frame)

displayImage = cv.QueryFrame(capture)

# Get a greyscale image for a motion mask
greyImage = cv.CreateImage(frameSize, cv.IPL_DEPTH_8U, 1)

runningAverageImage = cv.CreateImage(frameSize, cv.IPL_DEPTH_32F, 3)
runningAverageInDisplayColourDepth = cv.CloneImage(displayImage)

memStorage = cv.CreateMemStorage(0)

difference = cv.CloneImage(displayImage)

targetCount = 1
lastTarhetCount = 1
latTargetChangeT = 0.0
kOrGuess = 1
cookbook = []
frameCount = 0
lastFrameEntityList = []
maxTargets = 3

t0 = time.time()


while True:
    
    # Get the camera image
    cameraImage = cv.QueryFrame(capture)
    
    frameCount += 1
    frameT0 = time.time()
    
    displayImage = cv.CloneImage(cameraImage)    
    colourImage = cv.CloneImage(displayImage)
    
    cv.Smooth(colourImage, colourImage, cv.CV_GUASSIAN, 19, 0)
    
    # Use the Running Average as the static background            
    # a = 0.020 leaves artifacts lingering way too long.
    # a = 0.320 works well at 320x240, 15fps.  (1/a is roughly num frames.)
    cv.RunningAvg(colourImage, runningAverageImage, 0.320, None)
    cv.ConvertScale(runningAverageImage, runningAverageInDisplayColourDepth, 1.0, 0)
    
    #Get the difference between the content and the running average
    cv.AbsDiff(colourImage, runningAverageInDisplayColourDepth, difference)
    
    #Convert the difference image to grayscale
    cv.CvtColor(difference, greyImage, cv.CV_RGB2GRAY)
    
    # Threshold to difference image to a black and whit motion mask
    cv.Threshold(greyImage, greyImage, 2, 255, cv.CV_THRESH_BINARY)
    # Smooth and Threshold again to eliminate artifacts
    cv.Smooth(greyImage, greyImage, cv.CV_GUASSIAN, 19, 0)
    cv.Threshold(greyImage, greyImage, 240, 255, cv.CV_THRESH_BINARY)
    
    # Turn the greay image into an array
    greyImageAsArray = numpy.asarray(cv.GetMat(greyImage))
    nonBlackCoordinatesArray = numpy.where(greyImageAsArray > 3)
    nonBlackCoordinatesArray = zip (nonBlackCoordinatesArray[1], nonBlackCoordinatesArray[0])
    
    boundingBoxList = []
    
    # Use the white pixels as the motion and find the contours
    contour = cv.FindContours(greyImage, memStorage. cv.CV_RETR_CCOMP, cv.CV_CHAIN_APPROX_SIMPLE)
    
    while contour:
        
        boundingRect = cv.BoundingRect(list(contour))
        p1 = (boundingRect[0], boundingRect[1])
        p2 = (boundingRect[0] + boundingRect[2], boundingRect[1] + boundingRect[3])
        
        boundingBoxList.append((p1, p2))
        polygonPoints = cv.ApproxPoly( list(contour), memStorage, cv.CV_POLY_APPROX_DP )
        
        #Show the contours
        cv.FillPoly(greyImage, [ list(polygonPoints), ], cv.CV_RGB(255, 255, 255), 0, 0)
        cv.PolyLine(displayImage, [polygonPoints, ], 0, cv.CV_RGB(255,255, 255), 1, 0, 0)
        
        contour = contour.h_next()
    
    # Find the average size of the bounding box targets and remove ones that are 10% or less than the average as noise
    boxAreas = []
    for box in boundingBoxList:
        boxWidth = box[right][0] - box[left][0]
        boxHeight = box[bottom][0] - box[top][0]
        boxAreas.append(boxWidth * boxHeight)
        
    averageBoxArea = 0.0
    if len(boxAreas):
        averageBoxArea = float(sum(boxAreas) / len(boxAreas))
    
    trimmedBoxList = []
    for box in boundingBoxList:
        boxWidth = box[right][0] - box[left][0]
        boxHeight = box[bottom][0] - box[top][0]
        
        # remove the box if it's smaller than our noise threshold
        if (boxWidth * boxHeight) > 0.1 * averageBoxArea:
            trimmedBoxList.append(box)
    
    
cv.DestroyAllWindows()