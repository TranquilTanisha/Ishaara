import cv2
import mediapipe as mp
import os
import numpy as np
import pandas as pd

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils
final = [ ]

def look_for_videos_in_folder(folder_path):
    for folder in os.listdir(folder_path):
        print(folder)
        current_folder_path = os.path.join(folder_path, folder)
        if os.path.isdir(current_folder_path):
            process_videos_in_folder(current_folder_path)

def process_videos_in_folder(folder_path):
    final_data = []
    for video_file in os.listdir(folder_path):
        if video_file.endswith(('.mp4')):
            video_path = os.path.join(folder_path, video_file)
            print(f"Processing {video_file}...")

            frames = process_video(video_path)

            final_data += frames

            file_path = 'dataset\\' + folder_path.split("\\")[-1] + '.npy'
            np.save(file_path, final_data)
    np.save('dataset.npy', final_data)
    df = pd.DataFrame.from_records(final_data)
    df.to_csv('dataset.csv', index=False)

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    folder_name = video_path.split('\\')[-2]

    # Calculate the number of frames to skip
    target_frames = 20
    skip_frames = max(1, frame_count // target_frames)
    frame_c = 0
    frames = []
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
            frame_list = {'Word': folder_name}

            if results_hands.multi_hand_landmarks:
                for id, hand_landmarks in enumerate(results_hands.multi_hand_landmarks):
                    if results_hands.multi_handedness[id].classification[0].label == 'Left':
                        left_hand_landmarks = hand_landmarks
                    
                    if results_hands.multi_handedness[id].classification[0].label == 'Right':
                        right_hand_landmarks = hand_landmarks

            if results_pose.pose_landmarks:
                pose_landmarks = results_pose.pose_landmarks

            frames.append(append_landmarks(left_hand_landmarks, right_hand_landmarks, pose_landmarks, n_frame, frame_list))
            # Adjust skip_frames based on the current progress
            if(n_frame != target_frames):
                skip_frames = max(1, int((frame_count - frame_c) / (target_frames - n_frame)))
            n_frame += 1
        frame_c += 1

    cap.release()
    return frames

def append_landmarks(left_hand_landmarks, right_hand_landmarks, pose_landmarks, n_frame, frame_list):
    frame_list['Frame'] = n_frame

    for i in range(66):
        frame_list[f'X{"{:02d}".format(i)}'] = 0
        frame_list[f'Y{"{:02d}".format(i)}'] = 0
        frame_list[f'Z{"{:02d}".format(i)}'] = 0

    if left_hand_landmarks is not None:
        for id, lm in enumerate(left_hand_landmarks.landmark):
            # The landmark coordinates are in normalized image space.
            x, y, z = lm.x, lm.y, lm.z
            frame_list[f'X{"{:02d}".format(id)}'] = x
            frame_list[f'Y{"{:02d}".format(id)}'] = y
            frame_list[f'Z{"{:02d}".format(id)}'] = z

    if right_hand_landmarks is not None:
        for id, lm in enumerate(right_hand_landmarks.landmark):
            # The landmark coordinates are in normalized image space.
            x, y, z = lm.x, lm.y, lm.z
            frame_list[f'X{(id+21)}'] = x
            frame_list[f'Y{(id+21)}'] = y
            frame_list[f'Z{(id+21)}'] = z
    
    if pose_landmarks is not None:
        for id, lm in enumerate(pose_landmarks.landmark):
            # The landmark coordinates are in normalized image space.
            if id == 24:
                break
            x, y, z = lm.x, lm.y, lm.z
            frame_list[f'X{(id+42)}'] = x
            frame_list[f'Y{(id+42)}'] = y
            frame_list[f'Z{(id+42)}'] = z

    #Shift origin of the points relative to the shoulder (index 54--)
    for i in range(66):
        frame_list[f'X{"{:02d}".format(i)}'] -= frame_list[f'X54']
        frame_list[f'Y{"{:02d}".format(i)}'] -= frame_list[f'Y54']
        frame_list[f'Z{"{:02d}".format(i)}'] -= frame_list[f'Z54']
    #Normalize the points with respect to the shoulder landmarks (index 53--)
    for i in range(66):
        frame_list[f'X{"{:02d}".format(i)}'] /= frame_list[f'X53']
        frame_list[f'Y{"{:02d}".format(i)}'] /= frame_list[f'Y53']
        frame_list[f'Z{"{:02d}".format(i)}'] /= frame_list[f'Z53']

    return frame_list 


folder_path = 'Words'
look_for_videos_in_folder(folder_path)
# process_videos_in_folder(folder_path)
# df.to_csv('sign_language_data.csv')
