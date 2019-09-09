# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 15:20:01 2019

@author: MONSTER
"""

import numpy as np
import cv2

video_path = 'video.avi'
cv2.ocl.setUseOpenCL(False)
    
version = cv2.__version__.split('.')[0]
print(version) 


#read video file
cap = cv2.VideoCapture(video_path)

#check opencv version
if version == '2' :
	fgbg = cv2.BackgroundSubtractorMOG2()
if version == '3': 
	fgbg = cv2.createBackgroundSubtractorMOG2()
	

while (cap.isOpened):

	#if ret is true than no error with cap.isOpened
    ret, frame = cap.read()
	
    if ret==True:

		#apply background substraction
        fgmask = fgbg.apply(frame)
        ret1,th1 = cv2.threshold(fgmask,150,200,cv2.THRESH_BINARY)				
		#check opencv version
        if version == '2' : 
            (contours, hierarchy) = cv2.findContours(th1.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        if version == '3' : 
            (im2, contours, hierarchy) = cv2.findContours(th1.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            
        i = 0       	
		#looping for contours
        for c in contours:
            if cv2.contourArea(c) < 500:
                continue
				
			#get bounding box from countour
            (x, y, w, h) = cv2.boundingRect(c)
			
            i = i + 1
            
			#draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
          
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame,str(i),(100,400),font,4,(255,255,255),2,cv2.LINE_AA)
        cv2.imshow('foreground and background',th1)
        cv2.imshow('rgb',frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
