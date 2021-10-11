import cv2
import numpy as np
import time

video=cv2.VideoCapture(0,cv2.CAP_DSHOW)                 # cv2.VideoCapture() gets the video capture object for the camera.
time.sleep(3)
for i in range(60):                                     #background frames are stored in a loop
    check,background = video.read()                     
background = np.flip(background, axis=1)                #real image is flipped
                                                        
while(video.isOpened()):                        
    check,img = video.read()
    if check==False:
        break
    img = np.flip(img, axis=1)                          #axis = 1 laterally inverts the image
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)          #convert the image to hsv

    low_red = np.array([0, 120,50])
    high_red = np.array([10, 255,255])
    mask1 = cv2.inRange(hsv, low_red, high_red)      #cv2.inRange(hsv, hsv_lower, hsv_higher) is the syntax
    lower_red = np.array([170,120,70])
    higher_red = np.array([1806,255,255])
    mask2 = cv2.inRange(hsv, lower_red, higher_red)
    mask1= mask1+mask2
    mask1 = cv2.morphologyEx(mask1,cv2.MORPH_OPEN, np.ones((3,3),np.uint8))
    mask1 = cv2.morphologyEx(mask1,cv2.MORPH_DILATE, np.ones((3,3),np.uint8))
    mask2 = cv2.bitwise_not(mask1)
    res1 = cv2.bitwise_and(img,img, mask = mask2)
    res2 = cv2.bitwise_and(background,background, mask=mask1)

    final = cv2.addWeighted(res1, 1, res2, 1, 0)
    cv2.imshow("final",final)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
