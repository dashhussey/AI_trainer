import cv2
import numpy as np
import time
import PoseModule as pm
import csv
import pandas as pd

count = 0
dir = 0
pTime = 0

#cap = cv2.VideoCapture("F:\\Personal AI trainer\\c6.mp4")
cap = cv2.VideoCapture(0)
detector = pm.poseDetector()

while True:
    #img = cv2.imread("F:\\Personal AI trainer\\img15.jpg")
    success, img = cap.read()
    img = cv2.resize(img, (1200, 720))
    img = detector.findPose(img, False)
    lmList = detector.findPosition(img, False)
    #print(lmList)

    if len(lmList) !=0:
        angle = detector.findAngle(img, 11, 13, 15)
        angle_2 = detector.findAngle2(0, 11)
        dist = detector.findDist(img, 16, 15)
        per = np.interp(angle, (80, 145), (100, 0)) #to get the percentage values for the angle 0 - 100%
        bar = np.interp(angle, (80, 145), (100, 400)) #Max first, Min second
        #print(angle, per)

        color = (252, 29, 29)
        if per == 100:
            color = (0, 255, 255)
            if dir == 0:
                count += 1
                dir = 1

        if per == 0:
            if dir == 1:
                dir = 0

        if cv2.waitKey(1) & 0xFF == ord('r'):
            count = 0

        print(count, "%.2f" % per, int(angle), dist, round(angle_2, 2))

        #Save data to CSV file
        with open('values1.csv', 'a', encoding='UTF8', newline='') as f:
            # row1 = ('count', 'per', 'angle')
            row = (count, "%.2f" % per, int(angle), dist)
            writer = csv.writer(f)
            writer.writerow(row)
            f.close()

        #Rename the headers of CSV file and save to same CSV file
        file = pd.read_csv("values1.csv")
        headerList = ['COUNT', 'PER%', 'ANGLE', 'DIST']
        file.to_csv("values1.csv", header=headerList, index=False)

        #Draw bar
        cv2.rectangle(img, (1130, 100), (1100, 400), color, 2)
        cv2.rectangle(img, (1130, int(bar)), (1100, 400), color, cv2.FILLED)
        cv2.putText(img, f'{int(per)}%', (1080, 70), cv2.FONT_HERSHEY_COMPLEX, 1, color, 2)

        #Draw curl counts
        cv2.rectangle(img, (5, 620), (450, 700), (255, 255, 255), cv2.FILLED)
        cv2.putText(img, str("Curl count:"), (30, 675), cv2.FONT_HERSHEY_COMPLEX, 1.5, (255, 0, 0), 2)
        #cv2.putText(img, str(int(count)), (380, 675), cv2.FONT_HERSHEY_COMPLEX, 1.5, (0, 0, 255), 2)
        cv2.putText(img, str(int(dist)), (380, 675), cv2.FONT_HERSHEY_COMPLEX, 1.5, (0, 0, 255), 2)

        # cTime = time.time()
        # fps = 1 / (cTime - pTime)  # fps calculation formula
        # pTime = cTime
        #
        # cv2.putText(img, str(int(fps)), (40, 60), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0),
        #             2)  # calculated fps is displayed on screen with given position

    cv2.imshow("AI Trainer for Bicep Curls", img)
    #cv2.imwrite('savedimage_15.jpeg', img)
    cv2.waitKey(1)
