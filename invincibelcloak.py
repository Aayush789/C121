import cv2
import time
import numpy as np

#to save the output in the file called output.avi.
 

fourcc = cv2.VideoWriter_fourcc (*'XVID')
output_file = cv2.VideoWriter('output.avi',fourcc,20.0,(640,480))

#starting the web cam

cap = cv2.VideoCapture(0)

#allowing the webcam to start by making the sleep for 2 seconds

time.sleep(2)
bg = 0

#capuring the bg for 60 frames

for i in range(60):

    ret,bg = cap.read()

#fliping the bg coz the camera flicks the background inverted

bg = np.flip(bg,axis = 1)

#reading the capture frame or every frame the bg until the camera is open

#cap.isOpened() is to check the camera is open. 

#ret returns value true or false

while (cap.isOpened()):
    ret,img = cap.read()
    if not ret:
        break
    #flipping the img for consistance
    img  = np.flip(img,axis=1)

    #converting thr color from bgr to hsv

    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

    #generating the marks to detect red color

    #the values can change as per the color
        
    lower_red = np.array([0,120,50])

    upper_red = np.array([10,255,255])

    mask_1 = cv2.inRange(hsv,lower_red,upper_red)

    lower_red = np.array([170,120,70])

    upper_red = np.array([180,255,255])

    mask_2 = cv2.inRange(hsv,lower_red,upper_red)

    mask_1 = mask_1 + mask_2

    #open and expand the img where the is mask 1(color)

    mask_1 = cv2.morphologyEx(mask_1,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))

    mask_1 = cv2.morphologyEx(mask_1,cv2.MORPH_DILATE,np.ones((3,3),np.uint8))


    #selecting the part doesn't have the mask 1 and saving it to mask 2

    mask_2 = cv2.bitwise_not(mask_1)

    #keeping the only part of without red color.
     
    res_1 = cv2.bitwise_and(img,img, mask = mask_2)

    #keeping te only part of images with red color
     
    res_2 = cv2.bitwise_and(bg,bg, mask = mask_1)

    #generating the final output by merging res_1 and res_2

    final_output = cv2.addWeighted(res_1,1,res_2,1,0)

    output_file.write(final_output)

    #displaying the output to the user.

    cv2.imshow("magic",final_output)
    cv2.waitKey(1)

cap.release()
out.release()
cv2.destroyAllWindows()


    




    
 