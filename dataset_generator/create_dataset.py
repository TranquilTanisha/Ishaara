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
# df = pd.DataFrame(columns=['Class Label', 'Landmarks'])

def look_for_videos_in_folder(folder_path):
    for folder in os.listdir(folder_path):
        print(folder)
        current_folder_path = os.path.join(folder_path, folder)
        if os.path.isdir(current_folder_path):
            process_videos_in_folder(current_folder_path)

def process_videos_in_folder(folder_path):
    final_data = []
    for video_file in os.listdir(folder_path):
        data = []
        if video_file.endswith(('.mp4')):
            video_path = os.path.join(folder_path, video_file)
            print(f"Processing {video_file}...")


            frame_list = process_video(video_path)

            
            final_data.append(frame_list)
            data = np.array(data, dtype=object)

            # np.save('dataset\\' + video_path.split('\\')[-1].replace(".mp4", "") + '.npy', data)
            # df.loc[len(df)] = data
    np.save('dataset.npy', final_data)

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    folder_name = video_path.split('\\')[-2]

    # Calculate the number of frames to skip
    target_frames = 20
    skip_frames = (int(frame_count/target_frames))
    frame_c = 0
    frame_list = {'Class Label': folder_name}
    n_frame = 1

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # temp = [0] * 198  # 42*3 for hands, 24*3 for poses
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
                        left_hand_landmarks = hand_landmarks
                    
                    if results_hands.multi_handedness[id].classification[0].label == 'Right':
                        right_hand_landmarks = hand_landmarks

            if results_pose.pose_landmarks:
                pose_landmarks = results_pose.pose_landmarks

            frame_list = append_landmarks(left_hand_landmarks, right_hand_landmarks, pose_landmarks, n_frame, frame_list)
            
            # Adjust skip_frames based on the current progress
            if(len(frame_list) != target_frames):
                skip_frames = max(1, int((frame_count - frame_c) / (target_frames - len(frame_list))))
            n_frame += 1
        frame_c += 1

    cap.release()
    return frame_list

def append_landmarks(left_hand_landmarks, right_hand_landmarks, pose_landmarks, n_frame, frame_list):
    for i in range(66):
        frame_list[f'X{"{:02d}".format(i)}{"{:02d}".format(n_frame)}'] = 0
        frame_list[f'Y{"{:02d}".format(i)}{"{:02d}".format(n_frame)}'] = 0
        frame_list[f'Z{"{:02d}".format(i)}{"{:02d}".format(n_frame)}'] = 0

    if left_hand_landmarks is not None:
        for id, lm in enumerate(left_hand_landmarks.landmark):
            # The landmark coordinates are in normalized image space.
            x, y, z = lm.x, lm.y, lm.z
            frame_list[f'X{"{:02d}".format(id)}{"{:02d}".format(n_frame)}'] = x
            frame_list[f'Y{"{:02d}".format(id)}{"{:02d}".format(n_frame)}'] = y
            frame_list[f'Z{"{:02d}".format(id)}{"{:02d}".format(n_frame)}'] = z

    if right_hand_landmarks is not None:
        for id, lm in enumerate(right_hand_landmarks.landmark):
            # The landmark coordinates are in normalized image space.
            x, y, z = lm.x, lm.y, lm.z
            frame_list[f'X{(id+21)}{"{:02d}".format(n_frame)}'] = x
            frame_list[f'Y{(id+21)}{"{:02d}".format(n_frame)}'] = y
            frame_list[f'Z{(id+21)}{"{:02d}".format(n_frame)}'] = z
    
    if pose_landmarks is not None:
        for id, lm in enumerate(pose_landmarks.landmark):
            # The landmark coordinates are in normalized image space.
            if id == 24:
                break
            x, y, z = lm.x, lm.y, lm.z
            frame_list[f'X{(id+42)}{"{:02d}".format(n_frame)}'] = x
            frame_list[f'Y{(id+42)}{"{:02d}".format(n_frame)}'] = y
            frame_list[f'Z{(id+42)}{"{:02d}".format(n_frame)}'] = z

    #Shift origin of the points relative to the shoulder (index 54--)
    for i in range(66):
        frame_list[f'X{"{:02d}".format(i)}{"{:02d}".format(n_frame)}'] -= frame_list[f'X54{"{:02d}".format(n_frame)}']
        frame_list[f'Y{"{:02d}".format(i)}{"{:02d}".format(n_frame)}'] -= frame_list[f'Y54{"{:02d}".format(n_frame)}']
        frame_list[f'Z{"{:02d}".format(i)}{"{:02d}".format(n_frame)}'] -= frame_list[f'Z54{"{:02d}".format(n_frame)}']
    #Normalize the points with respect to the shoulder landmarks (index 53--)
    for i in range(66):
        frame_list[f'X{"{:02d}".format(i)}{"{:02d}".format(n_frame)}'] /= frame_list[f'X53{"{:02d}".format(n_frame)}']
        frame_list[f'Y{"{:02d}".format(i)}{"{:02d}".format(n_frame)}'] /= frame_list[f'Y53{"{:02d}".format(n_frame)}']
        frame_list[f'Z{"{:02d}".format(i)}{"{:02d}".format(n_frame)}'] /= frame_list[f'Z53{"{:02d}".format(n_frame)}']

    return frame_list 


# folder_path = 'words'
folder_path = 'dataset_generator\words'
look_for_videos_in_folder(folder_path)
process_videos_in_folder(folder_path)
# df.to_csv('sign_language_data.csv')


data = np.load('dataset.npy')

# Save the NumPy array as a CSV file
np.savetxt('dataset.csv', data, delimiter=',')