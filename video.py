import cv2
from screeninfo import get_monitors
import os

def capture_video():
    monitors = get_monitors()
    monitor = monitors[0]
    width, height = monitor.width, monitor.height
    cap = cv2.VideoCapture(0)
    cap.set(3, width)
    cap.set(4, height)
    cap.set(10, 150) #brightness

    # Check if the webcam is opened successfully
    if not cap.isOpened():
        print("Error: Failed to open webcam.")
        return

    while True:
        ret, frame = cap.read()
        cv2.imshow('Live Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    
capture_video()