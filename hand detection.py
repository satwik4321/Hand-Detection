#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import mediapipe as mp
import time


# In[2]:


cap=cv2.VideoCapture(0)


# In[3]:


cap.isOpened()


# In[4]:


mpDraw=mp.solutions.drawing_utils


# In[5]:


mphands=mp.solutions.hands


# In[6]:


hands=mphands.Hands()


# In[7]:


ctime=0
ptime=0


# In[8]:


while True:
    success, img=cap.read()
    imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    result=hands.process(imgRGB)
    if result.multi_hand_landmarks:    
        for mlhands in result.multi_hand_landmarks:
            for id,lm in enumerate(mlhands.landmark):
                h,w,c=img.shape
                cx,cy=int(lm.x*w),int(lm.y*h)
                print(id,cx,cy)
                if(id==4):
                    cv2.circle(imgRGB,(cx,cy),25,(255,0,255),cv2.FILLED)
            mpDraw.draw_landmarks(imgRGB,mlhands,mphands.HAND_CONNECTIONS)
    cv2.imshow("Image",imgRGB)
    ctime=time.time()
    fps=1/(ctime-ptime)
    ptime=ctime
    cv2.putText(imgRGB,str(int(fps)),(100,100),cv2.FONT_HERSHEY_PLAIN,3,(0,0,0),3)
    cv2.waitKey(1)
cap.release()
cv2.destroyAllWindows()

