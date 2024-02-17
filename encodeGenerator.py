import os
import cv2
import face_recognition
import pickle

folderPath = 'Resources/images/students_images_(studentIds)'
pathList = os.listdir(folderPath)
imgList = []
studentIds = []
for path in pathList:
    imgList.append(cv2.imread(os.path.join(folderPath, path)))
    # img are saved as their student ids and now to remove .png from it
    studentIds.append(os.path.splitext(path)[0])

print(studentIds)

def findEncodings(imageList):
    encodeList = []
    for img in imageList:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)

    return encodeList

print("Encoding Started....")
encodeListKnown = findEncodings(imgList)
encodeListKnownWithIds = [encodeListKnown, studentIds]
print(encodeListKnownWithIds)
print("Encoding Completed")

file = open("encodeFile.p", 'wb')
pickle.dump(encodeListKnownWithIds,file)
file.close()
print("File Saved")

