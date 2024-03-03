import cv2
import mediapipe as mp
import os
import csv

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils
frames = []

def look_for_videos_in_folder(folder_path):
    for folder in os.listdir(folder_path):
        folder_path = os.path.join(folder_path, folder)
        if os.path.isdir(folder_path):
            process_videos_in_folder(folder_path)

def process_videos_in_folder(folder_path):
    for video_file in os.listdir(folder_path):
        if video_file.endswith(('.mp4')):
            video_path = os.path.join(folder_path, video_file)
            print(f"Processing {video_file}...")
            process_video(video_path)
            add_to_csv(frames)


def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Calculate the number of frames to skip
    skip_frames = max(int(fps / 6), 3)

    frame_c = 0
    frame_count = 1

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_list = [0]*200 # 1 for Frame Count, 42*3 for hands, 24*3 for poses, 1 for class label
        # Convert the BGR image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame to detect hands
        results_hands = hands.process(image)
        # Process the frame to detect pose
        results_pose = pose.process(image)

        if frame_c % (skip_frames) == 0:
            # Extract and print landmark coordinates
            if results_hands.multi_hand_landmarks:
                for id, hand_landmarks in enumerate(results_hands.multi_hand_landmarks):
                    append_landmarks(hand_landmarks, 'Hand', frame_list, results_hands.multi_handedness[id].classification[0].label)

            if results_pose.pose_landmarks:
                append_landmarks(results_pose.pose_landmarks, 'Pose', frame_list, ' ')

            folder_name = video_path.split('\\')[1]

            frame_list[0] = frame_count
            frame_list[-1] = folder_name
            
            frames.append(frame_list)
            frame_count += 1
        frame_c += 1

    cap.release()

def append_landmarks(landmarks, landmark_type, frame_list, hand_label):
    if landmark_type == 'Hand':
        for id, lm in enumerate(landmarks.landmark):
            # The landmark coordinates are in normalized image space.
            x, y, z = lm.x, lm.y, lm.z
            if hand_label == 'Left':
                frame_list[id+1] = x
                frame_list[id+67] = y
                frame_list[id+133] = z
            else:
                frame_list[id+22] = x
                frame_list[id+88] = y
                frame_list[id+154] = z
    
    else:
        for id, lm in enumerate(landmarks.landmark):
            if id == 24:
                break
            x, y, z = lm.x, lm.y, lm.z
            frame_list[id+43] = x
            frame_list[id+109] = y
            frame_list[id+175] = z
    print(frame_list[43:67], frame_list[109:133], frame_list[175:])

def add_to_csv(frames):
    csv_file_path = 'gesture_landmarks.csv'

    # Open the CSV file in write mode
    with open(csv_file_path, 'a', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        # Write the header
        # csv_writer.writerow(['Frame Count'] + [f'X{i}' for i in range(1, 67)] + [f'Y{i}' for i in range(1, 67)] + [f'Z{i}' for i in range(1, 67)] + ['Class Label'])
        # Write the data
        for row in frames:
            csv_writer.writerow(row)      

# Specify the folder path containing your videos
folder_path = 'Words/Boy'
# look_for_videos_in_folder(folder_path)
process_videos_in_folder(folder_path)

