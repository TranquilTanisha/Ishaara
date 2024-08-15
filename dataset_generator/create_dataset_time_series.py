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

    indices = [0, 1, 4, 5, 8, 9, 12, 13, 16, 17, 20]

    for i in range(27):
        frame_list[f'X{"{:02d}".format(i)}'] = 0
        frame_list[f'Y{"{:02d}".format(i)}'] = 0
        frame_list[f'Z{"{:02d}".format(i)}'] = 0

    index = 0
    if left_hand_landmarks is not None:
        for id, lm in enumerate(left_hand_landmarks.landmark):
            # The landmark coordinates are in normalized image space.
            if id in indices:
                x, y, z = lm.x, lm.y, lm.z
                frame_list[f'X{"{:02d}".format(index)}'] = x
                frame_list[f'Y{"{:02d}".format(index)}'] = y
                frame_list[f'Z{"{:02d}".format(index)}'] = z
                index += 1

    index = 0
    if right_hand_landmarks is not None:
        for id, lm in enumerate(right_hand_landmarks.landmark):
            # The landmark coordinates are in normalized image space.
            if id in indices:
                x, y, z = lm.x, lm.y, lm.z
                frame_list[f'X{(index+11)}'] = x
                frame_list[f'Y{(index+11)}'] = y
                frame_list[f'Z{(index+11)}'] = z
                index += 1
    
    index = 0
    if pose_landmarks is not None:
        for id, lm in enumerate(pose_landmarks.landmark):
            # The landmark coordinates are in normalized image space.
            if id > 14:
                break
            if id in [0, 11, 12, 13, 14]:
                x, y, z = lm.x, lm.y, lm.z
                frame_list[f'X{(index+22)}'] = x
                frame_list[f'Y{(index+22)}'] = y
                frame_list[f'Z{(index+22)}'] = z
                index += 1

    #Shift origin of the points relative to the shoulder (index 54--)
    for i in range(27):
        frame_list[f'X{"{:02d}".format(i)}'] -= frame_list[f'X24']
        frame_list[f'Y{"{:02d}".format(i)}'] -= frame_list[f'Y24']
        frame_list[f'Z{"{:02d}".format(i)}'] -= frame_list[f'Z24']
    
    if frame_list[f'X23'] == 0:
        frame_list[f'X23'] = 1
    else:
        #Normalize the points with respect to the shoulder landmarks (index 53--)
        for i in range(27):
            frame_list[f'X{"{:02d}".format(i)}'] /= frame_list[f'X23']
            frame_list[f'Y{"{:02d}".format(i)}'] /= frame_list[f'Y23']
            frame_list[f'Z{"{:02d}".format(i)}'] /= frame_list[f'Z23']

    return frame_list 


folder_path = 'Words'
look_for_videos_in_folder(folder_path)
