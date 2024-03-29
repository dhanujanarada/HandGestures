import cv2
import mediapipe as mp
import time
import HandGestureTrackingModule as htm


pTime = 0
cTime = 0
cap = cv2.VideoCapture(0)
detector = htm.HandDetector()

while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmlst = detector.findpositions(img)
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_TRIPLEX, 3, (255, 0, 255), 3)

    cv2.imshow("Hand Tracking", img)
    cv2.waitKey(1)
