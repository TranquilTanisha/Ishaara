import cv2
import mediapipe as mp
import os
import csv
import numpy as np
import pandas as pd
import json
from screeninfo import get_monitors

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils
# X = pd.DataFrame(columns=['Class Label', 'Landmarks'])
# X = {'Landmarks':[], 'Length':[]}
# Y={'Class Label':[]}
landmarks={}
length={}
label={}
count=0

def process_videos_in_folder(folder_path):
    global count
    for video_file in os.listdir(folder_path):
        # data = []
        if video_file.endswith(('.mp4')):
            label[count]=[]
            length[count]=[]
            landmarks[count]=[]
            
            video_path = os.path.join(folder_path, video_file)
            print(f"Processing {video_file}")
            
            folder_name = video_path.split('/')[-2]
            # Y['Class Label'] += [labels[folder_name]]
            label[count] += [labels[folder_name]]

            frame_list = process_video(video_path)
            print(len(frame_list))
            length[count]+=[len(frame_list)]
            landmarks[count] += [frame_list] #1D nd array
            count+=1

def append_landmarks(frame, left_hand_landmarks, right_hand_landmarks, pose_landmarks):
    if left_hand_landmarks is not None:
        for id, lm in enumerate(left_hand_landmarks.landmark):
            x, y, z = lm.x, lm.y, lm.z
            frame[id] = x
            frame[id+66] = y
            frame[id+132] = z

    if right_hand_landmarks is not None:
        for id, lm in enumerate(right_hand_landmarks.landmark):
            x, y, z = lm.x, lm.y, lm.z
            frame[id+21] = x
            frame[id+87] = y
            frame[id+153] = z
    
    if pose_landmarks is not None:
        for id, lm in enumerate(pose_landmarks.landmark):
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

def capture_video():
    monitors = get_monitors()
    monitor = monitors[0]
    width, height = monitor.width, monitor.height
    cap = cv2.VideoCapture(0)
    cap.set(3, width)
    cap.set(4, height)
    cap.set(10, 150) #brightness
    return cap

def process_video(video_path):
    if type(video_path)==str:
        cap = cv2.VideoCapture(video_path)
    else:
        cap=capture_video()
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate the number of frames to skip
    skip_frames = (int(frame_count/20))
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
        process_videos_in_folder(folder_path)


with open('data.json', 'r') as f:
    labels = json.load(f)

# look_for_videos_in_folder()
# process_videos_in_folder('dataset_generator/words/Boy/')
# process_videos_in_folder('dataset_generator/words/Can/')
# process_videos_in_folder('dataset_generator/words/Eat/')
# process_videos_in_folder('dataset_generator/words/Fine/')
# process_videos_in_folder('dataset_generator/words/Girl/')
# process_videos_in_folder('dataset_generator/words/Help/')
# process_videos_in_folder('dataset_generator/words/How/')
# process_videos_in_folder('dataset_generator/words/Hungry/')
# process_videos_in_folder('dataset_generator/words/I/')
# process_videos_in_folder('dataset_generator/words/Name/')
# process_videos_in_folder('dataset_generator/words/Parents/')
process_videos_in_folder('dataset_generator/words/Sister/')
# process_videos_in_folder('dataset_generator/words/Sleep/')
# process_videos_in_folder('dataset_generator/words/This/')
# process_videos_in_folder('dataset_generator/words/You/')


# process_videos_in_folder('dataset_generator/words/Mother/')
# process_videos_in_folder('dataset_generator/words/Namaste/')

# X=pd.DataFrame(X)
# Y=pd.DataFrame(Y)

# if os.path.exists('landmarks.csv'):
#     print(True)
#     landmarks=pd.read_csv('landmarks.csv')
#     labels=pd.read_csv('labels.csv')
#     landmarks=pd.concat([landmarks, X], ignore_index=True)
#     labels=pd.concat([labels, Y], ignore_index=True)
#     os.remove('landmarks.csv')
#     os.remove('labels.csv')
#     landmarks.to_csv('landmarks.csv', index=False)
#     labels.to_csv('labels.csv', index=False)
# else:
#     print(False)
#     X.to_csv('landmarks.csv', index=False)
#     Y.to_csv('labels.csv', index=False)

with open('landmarks.json', 'w') as f:
    json.dump(landmarks, f)
    
with open('label.json', 'w') as f:
    json.dump(landmarks, f)
    
with open('length.json', 'w') as f:
    json.dump(landmarks, f)