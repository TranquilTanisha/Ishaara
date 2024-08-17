import cv2
import mediapipe as mp
import os
import numpy as np

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
            final_data.extend(frames)
            file_path = 'dataset\\' + folder_path.split("\\")[-1] + '.npy'
            np.save(file_path, final_data)

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
            frame_list = {'Word': folder_name, 'Frame': n_frame}
            indices = [0, 1, 4, 5, 8, 9, 12, 13, 16, 17, 20]
            landmarks = extract_landmarks(results_hands, results_pose, indices)
            frame_list.update(landmarks)
            frames.append(frame_list)
            # Adjust skip_frames based on the current progress
            if(n_frame < target_frames):
                skip_frames = max(1, int((frame_count - frame_c) / (target_frames - n_frame)))
            n_frame += 1
        frame_c += 1

    cap.release()
    return frames

def extract_landmarks(results_hands, results_pose, indices):
    landmarks = {}
    if results_hands.multi_hand_landmarks:
        for id, hand_landmarks in enumerate(results_hands.multi_hand_landmarks):
            label = results_hands.multi_handedness[id].classification[0].label
            if label == 'Left':
                landmarks.update(extract_hand_landmarks(hand_landmarks, "L", indices))
            elif label == 'Right':
                landmarks.update(extract_hand_landmarks(hand_landmarks, "R", indices))
    if results_pose.pose_landmarks:
        landmarks.update(extract_pose_landmarks(results_pose.pose_landmarks))
    return landmarks

def extract_hand_landmarks(hand_landmarks, side, indices):
    landmarks = {}
    for i, lm in enumerate(hand_landmarks.landmark):
        if i in indices:
            landmarks[f'{side}x{i:02d}'] = lm.x
            landmarks[f'{side}y{i:02d}'] = lm.y
            landmarks[f'{side}z{i:02d}'] = lm.z
    return landmarks

def extract_pose_landmarks(pose_landmarks):
    landmarks = {}
    for i, lm in enumerate(pose_landmarks.landmark):
        if i in [0, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]:
            landmarks[f'Px{i:02d}'] = lm.x
            landmarks[f'Py{i:02d}'] = lm.y
            landmarks[f'Pz{i:02d}'] = lm.z
    return landmarks


folder_path = 'Words'
look_for_videos_in_folder(folder_path)
