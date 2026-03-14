import cv2
from ultralytics import YOLO
from playsound import playsound
import threading

model = YOLO(r"D:\projects\runs\detect\train5\weights\best.pt")
print("model loaded")

cam=cv2.VideoCapture(0)

if not cam.isOpened():
    print("cam not open")
    exit()

while True:
    ret,frame=cam.read()
    if not ret:
        print("Fail to grab frame")
        break

    results = model(frame)
    annoted_frame=results[0].plot()
    cv2.imshow("Knife Detected",annoted_frame)
    # print(results)
    if results[0].boxes is not None:
        classes=results[0].boxes.cls.tolist()

        #--Thread to isolated or not distrub the camera
        if 2 in classes:
            threading.Thread(
                target=playsound,
                args=(r"A:\rec\gif\audio\alarm.wav",),
                daemon=True
            ).start()

        # playsound(r"A:\rec\gif\audio\alarm.wav") to play in sync and not in thread

    if cv2.waitKey(1)==27:
        break
cam.release()
cv2.destroyAllWindows()