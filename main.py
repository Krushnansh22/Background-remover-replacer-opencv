#Python version >=3.4 & <=3.9
#Import required libraries
import cvzone
import cv2  
from cvzone.SelfiSegmentationModule import SelfiSegmentation  
import os  

# Accessing the webcam
cap = cv2.VideoCapture(0)

# Setting the width and height of the video capture
cap.set(3, 640)
cap.set(4, 480)

# Creating a SelfiSegmentation object for background removal
segmentor = SelfiSegmentation()

# Creating an object to calculate frames per second
fps = cvzone.FPS()

# Loading a background image
imgBg = cv2.imread("backgrounds/hel.jpg")

# Listing all images in the 'backgrounds' folder
listImg = os.listdir("backgrounds")
print(listImg)

# Initializing variables
index = 0
imglist = []

# Loading all images from the 'backgrounds' folder
for imgPath in listImg:
    img = cv2.imread(f'backgrounds/{imgPath}')
    imglist.append(img)

# Main loop for video capture and background removal
while True:
    # Capturing video frame by frame
    success, img = cap.read()
    
    # Removing the background from the current frame using the SelfiSegmentation module
    imgOut = segmentor.removeBG(img, imglist[index], threshold=0.8)

    # Stacking the original frame and the frame with background removed vertically
    imgstacked = cvzone.stackImages([img, imgOut], 2, 1)
    
    # Updating and displaying the FPS counter on the stacked image
    _, imgstacked = fps.update(imgstacked)
    cv2.imshow("Image", imgstacked)
    
    # Waiting for a key press to change the background image or quit the program
    key = cv2.waitKey(1)
    if key == ord('a'):
        if index > 0:
            index -= 1
    elif key == ord('d'):
        if index < (len(imglist) - 1):
            index += 1
    elif key == ord('q'):
        break

# Releasing the webcam and closing all OpenCV windows
cap.release()
cv2.destroyAllWindows()
