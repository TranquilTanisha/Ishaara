import cv2
import mediapipe as mp
import os
import csv
import numpy as np
import pandas as pd

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils
df = pd.DataFrame(columns=['Index', 'Class Label', 'Landmarks'])

def look_for_videos_in_folder(folder_path):
    for folder in os.listdir(folder_path):
        folder_path = os.path.join(folder_path, folder)
        if os.path.isdir(folder_path):
            process_videos_in_folder(folder_path)

def process_videos_in_folder(folder_path):
    for video_file in os.listdir(folder_path):
        data = []
        if video_file.endswith(('.mp4')):
            video_path = os.path.join(folder_path, video_file)
            print(f"Processing {video_file}...")
            frame_list = process_video(video_path)
            print(frame_list)
            data.append(frame_list)

            folder_name = video_path.split('\\')[0]
            data.append(folder_name)

            df.loc[len(df)] = data

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Calculate the number of frames to skip
    skip_frames = max(int(fps / 8), 4)

    frame_c = 0
    frame_list = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        temp = [0] * 198  # 42*3 for hands, 24*3 for poses
        # temp=[]
        # Convert the BGR image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame to detect hands
        results_hands = hands.process(image)
        # Process the frame to detect pose
        results_pose = pose.process(image)

        if frame_c % skip_frames == 0:
            # Extract and print landmark coordinates
            left_hand_landmarks = None
            right_hand_landmarks = None
            pose_landmarks = None

            if results_hands.multi_hand_landmarks:
                for id, hand_landmarks in enumerate(results_hands.multi_hand_landmarks):
                    if results_hands.multi_handedness[id].classification[0].label == 'Left':
                        # left_hand_landmarks = hand_landmarks
                        
                        buffer=[]
                        left_hand_landmarks = hand_landmarks.landmark
                        for lm in left_hand_landmarks:
                            buffer+= [lm.x, lm.y, lm.z]
                    else:
                        buffer=[0]*63
                    temp+=buffer
                    
                    if results_hands.multi_handedness[id].classification[0].label == 'Right':
                        # right_hand_landmarks = hand_landmarks
                        buffer=[]
                        right_hand_landmarks = hand_landmarks.landmark
                        for lm in right_hand_landmarks:
                            buffer+= [lm.x, lm.y, lm.z]
                    else:
                        buffer=[0]*63
                    temp+=buffer
                    

                if results_pose.pose_landmarks:
                    buffer=[]
                    pose_landmarks = results_pose.pose_landmarks
                    if pose_landmarks is not None:
                        for lm in pose_landmarks.landmark:
                            buffer+= [lm.x, lm.y, lm.z]
                    else:
                        buffer=[0]*72

                # append_landmarks(temp, left_hand_landmarks, right_hand_landmarks, pose_landmarks)

            frame_list += temp
        frame_c += 1

    cap.release()
    return frame_list

# Specify the folder path containing your videos
folder_path = 'Words/Sister'
# look_for_videos_in_folder(folder_path)
process_videos_in_folder(folder_path)
df.to_csv('sign_language_data.csv')