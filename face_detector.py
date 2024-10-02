import cv2
import numpy as np
import face_recognition
import matplotlib.pyplot as plt 

#this loads the target image of the person we are looking for
target_image = face_recognition.load_image_file("target.jpg")
               #this loads the image


#loading the crowd image
crowd_image = face_recognition.load_image_file("crowd.jpg")


#display the images
plt.subplot(1, 2, 1)
plt.imshow(target_image)
plt.title('crowd iamge')

#this shows the images side by side
plt.show()

#find all face locations 
target_face_location = face_recognition.face_location(target_image)
crowd_face_locations = face_recognition.face_location(crowd_image)

#print the locations (coordinates) of faces detected
print("Target face location:", target_face_location)
print("crowd_face_locations:", crowd_face_locations)
