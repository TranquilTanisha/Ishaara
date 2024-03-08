import cv2
import mediapipe as mp
import os
import numpy

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Assuming you want to process the first frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, frame = cap.read()
    if not ret:
        print("Failed to read frame")
        return

    # Save the original frame
    cv2.imwrite('original_frame.jpg', frame)

    # Convert the frame to RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    black_image = numpy.zeros((image.shape[0], image.shape[1], 3), dtype=numpy.uint8)

    # Process the image for hand landmarks
    results = hands.process(image)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(black_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Process the image for pose landmarks
    results = pose.process(image)
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(black_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Convert the image back to BGR for saving
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    black_image = cv2.cvtColor(black_image, cv2.COLOR_RGB2BGR)
    # Save the frame with landmarks drawn
    cv2.imwrite('frame_with_landmarks.jpg', black_image)

    cap.release()

# Example usage
video_path = 'Words\Can\Can_4.mp4'
process_video(video_path)
