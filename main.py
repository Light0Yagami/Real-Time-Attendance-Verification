import os
import pickle

import cv2
import face_recognition

cap = cv2.VideoCapture(0)
# setting the HEIGHT of webcam frame
cap.set(4, 480)
# setting the WIDTH of webcam frame
cap.set(3, 640)

imgBackground = cv2.imread('Resources/images/background_image/background.png')

# ********************************IMPORTING THE MODE IMAGES ************************************************************

folderModePath = "Resources/status"
modePathList = os.listdir(folderModePath)
imgModeList = []
# check imgModeList for all status mode
# print(modePathList)
path: str
for path in modePathList:
    imgModeList.append(cv2.imread(os.path.join(folderModePath, path)))
# to check number of images
# print(len(imgModeList))

#*************************************** LOAD ENCODING FILES ***********************************************************

print("Loading Encode File ......")
file = open('encodeFile.p','rb')
encodeListKnownWithIds = pickle.load(file)
file.close()
encodeListKnown, studentsIds = encodeListKnownWithIds
#print(studentsIds)
print("Encode File Loaded .........")



# ****************************************** VIDEO CAPTURING ***********************************************************
while True:
    success, img = cap.read()

    imgS = cv2.resize(img, (0,0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    faceCurFrame = face_recognition.face_locations(imgS)
    encodeCurFrame = face_recognition.face_encodings(imgS,faceCurFrame)

    for encodeFace, faceLoc in zip(encodeCurFrame, faceCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace) #lower the face distance better the recognition
        print("matches", matches)
        print("faceDis", faceDis)



    # use img = cv2.resize(<height>,<width>)  if resolution is not supported
    imgBackground[162:162 + 480, 55:55 + 640] = img
    imgBackground[44:44 + 633, 808:808 + 414] = imgModeList[1]
    cv2.imshow("Face Attendance", imgBackground)

    # (Webcam test)
    # ccv2.imshow("webcam", img)

    # This line is checking if the ‘q’ key is pressed.
    # The cv2.waitKey(1) function waits for 1 millisecond for a key event on an OpenCV window12.
    # If pressed it breaks the loops and stops capturing
    # Wait indefinitely for a key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break



# Release the capture and destroy all windows
cap.release()
cv2.destroyAllWindows()
