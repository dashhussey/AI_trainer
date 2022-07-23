import cv2
import mediapipe as mp
import time
import math

wCam, hCam = 1280, 720
class poseDetector:

    def __init__(self, mode=False, upBody=False, smooth=True, detectionCon=False, trackCon=0.5):

        self.lmList = None
        self.results = None
        self.mode = mode
        self.upBody = upBody
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode, self.upBody, self.smooth, self.detectionCon, self.trackCon)

    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS, self.mpDraw.DrawingSpec(color=(0, 0, 255), thickness=8, circle_radius=8), self.mpDraw.DrawingSpec(color=(0, 255, 0), thickness=7))
        return img

    def findPosition(self, img, draw=True):
        self.lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                # print(id, lm)
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 255, 255), cv2.FILLED)
        return self.lmList

    def findAngle(self, img, p3, p2, p1, draw=True): #(p3, p2, p1) for left hand

        # Get the landmarks
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        x3, y3 = self.lmList[p3][1:]

        #Calculate the Angle
        angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
        if angle < 0:
            angle += 360
        #print(angle)

        # Draw
        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 3)
            cv2.line(img, (x2, y2), (x3, y3), (255, 255, 255), 3)
            cv2.circle(img, (x1, y1), 10, (0, 255, 0), cv2.FILLED)
            cv2.circle(img, (x1, y1), 15, (0, 0, 255), 2)
            cv2.circle(img, (x2, y2), 10, (0, 255, 0), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (0, 0, 255), 2)
            cv2.circle(img, (x3, y3), 10, (0, 255, 0), cv2.FILLED)
            cv2.circle(img, (x3, y3), 15, (0, 0, 255), 2)
            cv2.putText(img, str(int(angle)), (x2 - 85, y2 - 0), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5, (0, 255, 255), 2)
        return angle

    def findAngle2(self, p6, p7):

        # Get the landmarks
        x1, y1 = self.lmList[p6][1:]
        x2, y2 = self.lmList[p7][1:]
        theta = math.acos((y2 - y1) * (-y1) / (math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) * y1))
        degree = int(180 / math.pi) * theta
        return degree

    def findDist(self, img, p4, p5, draw=True):

        x4, y4 = self.lmList[p4][1:]
        x5, y5 = self.lmList[p5][1:]
        #x6, y6 = self.lmList[p6][1:]

        # Calculate the distance
        dist = math.dist((x4, y4), (x5, y5))
        #print(dist)

        if draw:
            cv2.line(img, (x4, y4), (x5, y5), (255, 255, 255), 3)
            cv2.putText(img, str(int(dist)), (x5 - 85, y5 - 0), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5, (0, 255, 255), 2)
            #cv2.circle(img, (x5, y5), 5, (255, 255, 255), 3)
        return dist

def main():
    cap = cv2.VideoCapture("F:\\New test 01\\c5.mp4")
    #cap = cv2.VideoCapture(1)
    #cap.set(3, wCam)
    #cap.set(4, hCam)
    pTime = 0
    detector = poseDetector()
    while True:
        success, img = cap.read()
        #cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        #cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        img = cv2.resize(img, (640, 480))
        img = detector.findPose(img)
        lmList = detector.findPosition(img, draw=True)
        if len(lmList) !=0:
            print(lmList[11])
            cv2.circle(img, (lmList[11][1], lmList[11][2]), 10, (255, 0, 255), cv2.FILLED)

        cTime = time.time()
        fps = 1 / (cTime - pTime)  # fps calculation formula
        pTime = cTime

        cv2.putText(img, str(int(fps)), (40, 60), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 2)  # calculated fps is displayed on screen with given position
        cv2.imshow('Mediapipe Feed', img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()