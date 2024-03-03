import cv2
import mediapipe as mp
import os
import csv
import numpy as np
import pandas as pd
import json

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils
# X = pd.DataFrame(columns=['Class Label', 'Landmarks'])
X = {'Landmarks':[]}
Y={'Class Label':[]}

def process_videos_in_folder(folder_path):
    for video_file in os.listdir(folder_path):
        # data = []
        if video_file.endswith(('.mp4')):
            video_path = os.path.join(folder_path, video_file)
            print(video_path)
            print(f"Processing {video_file}")
            
            # print(video_path)
            folder_name = video_path.split('/')[-2]
            # data.append(folder_name)
            Y['Class Label'] += [labels[folder_name]]

            frame_list = process_video(video_path)
            frame_list = np.array([frame_list])
            # print(frame_list.shape)
            X['Landmarks'] += [frame_list] #shape of (1, 3564)
            # data.append(frame_list)

def append_landmarks(frame, left_hand_landmarks, right_hand_landmarks, pose_landmarks):
    if left_hand_landmarks is not None:
        for id, lm in enumerate(left_hand_landmarks.landmark):
            # The landmark coordinates are in normalized image space.
            x, y, z = lm.x, lm.y, lm.z
            frame[id] = x
            frame[id+66] = y
            frame[id+132] = z

    if right_hand_landmarks is not None:
        for id, lm in enumerate(right_hand_landmarks.landmark):
            # The landmark coordinates are in normalized image space.
            x, y, z = lm.x, lm.y, lm.z
            frame[id+21] = x
            frame[id+87] = y
            frame[id+153] = z
    
    if pose_landmarks is not None:
        for id, lm in enumerate(pose_landmarks.landmark):
            # The landmark coordinates are in normalized image space.
            if id == 24:
                break
            x, y, z = lm.x, lm.y, lm.z
            frame[id+42] = x
            frame[id+108] = y
            frame[id+174] = z

    #Shift origin of the points relative to the nose
    for i in range(66):
        frame[i] -= frame[54]
    for i in range(66, 132):
        frame[i] -= frame[120]
    for i in range(132, 198):
        frame[i] -= frame[186]

    #Normalize the points with respect to the shoulder landmarks (index 53 and index 54)
    for i in range(66):
        frame[i] /= frame[53]
    for i in range(66, 132):
        frame[i] /= frame[119]
    for i in range(132, 198):
        frame[i] /= frame[185]

    return frame  

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
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results_hands = hands.process(image) #detect hands
        results_pose = pose.process(image) #detect poses

        if frame_c % skip_frames == 0:
            # Extract and print landmark coordinates
            left_hand_landmarks = None
            right_hand_landmarks = None
            pose_landmarks = None

            if results_hands.multi_hand_landmarks:
                for id, hand_landmarks in enumerate(results_hands.multi_hand_landmarks):
                    if results_hands.multi_handedness[id].classification[0].label == 'Left':
                        left_hand_landmarks = hand_landmarks
                    
                    if results_hands.multi_handedness[id].classification[0].label == 'Right':
                        right_hand_landmarks = hand_landmarks

                if results_pose.pose_landmarks:
                    pose_landmarks = results_pose.pose_landmarks

                append_landmarks(temp, left_hand_landmarks, right_hand_landmarks, pose_landmarks)

                frame_list += temp
        frame_c += 1

    cap.release()
    return frame_list

def look_for_videos_in_folder():
    for folder in os.listdir('dataset_generator/words/'):
        folder_path = 'dataset_generator/words/' + folder + '/'
        # if os.path.isdir(folder_path):
        process_videos_in_folder(folder_path)


with open('data.json', 'r') as f:
    labels = json.load(f)

look_for_videos_in_folder()
process_videos_in_folder('dataset_generator/words/Boy')

X=pd.DataFrame(X)
X.to_csv('landmarks.csv', index=False)
Y=pd.DataFrame(Y)
Y.to_csv('labels.csv', index=False)