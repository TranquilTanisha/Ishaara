import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from screeninfo import get_monitors
from preprocess import translate_to_english
import pyttsx3
import joblib


import tensorflow as tf

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils
    
scaler = joblib.load('scaler.gz')
model= tf.keras.models.load_model('lstmmodel.keras')

# scaler= tf.keras.models.load_model('scaler.h5')
    
with open ('words.txt', 'r') as f:
    words = f.read().splitlines()


class _TTS:

    engine = None
    rate = None
    def __init__(self):
        self.engine = pyttsx3.init()

    def start(self,text_):
        self.engine.say(text_)
        self.engine.runAndWait()

def handle_nans(features_reshaped):
    # Replace NaNs with zero or the mean of the column
    if np.isnan(features_reshaped).any():
        print("Handling NaNs in input data")
        features_reshaped = np.nan_to_num(features_reshaped, nan=0.5)
    return features_reshaped

def predict_with_check(features_reshaped):
    features_reshaped = handle_nans(features_reshaped)
    prediction = model.predict(features_reshaped)

    if np.isnan(prediction).any():
        print("Warning: Model prediction contains NaNs")
        return None  # Handle NaNs appropriately

    return prediction

def extract_landmarks(image, indices):
    landmarks = {}
    results_hands = hands.process(image)
    results_pose = pose.process(image)
    if results_hands.multi_hand_landmarks:
        for id, hand_landmarks in enumerate(results_hands.multi_hand_landmarks):
            label = results_hands.multi_handedness[id].classification[0].label
            if label == 'Left':
                landmarks.update(extract_hand_landmarks(hand_landmarks, "L", indices))
                for i in range(21):
                    if i in indices:
                        landmarks[f'Rx{i:02d}'] = None
                        landmarks[f'Ry{i:02d}'] = None
                        landmarks[f'Rz{i:02d}'] = None
            elif label == 'Right':
                landmarks.update(extract_hand_landmarks(hand_landmarks, "R", indices))
                for i in range(21):
                    if i in indices:
                        landmarks[f'Lx{i:02d}'] = None
                        landmarks[f'Ly{i:02d}'] = None
                        landmarks[f'Lz{i:02d}'] = None
    else:
        for i in range(21):
            if i in indices:
                landmarks[f'Lx{i:02d}'] = None
                landmarks[f'Ly{i:02d}'] = None
                landmarks[f'Lz{i:02d}'] = None
                landmarks[f'Rx{i:02d}'] = None
                landmarks[f'Ry{i:02d}'] = None
                landmarks[f'Rz{i:02d}'] = None

    if results_pose.pose_landmarks:
        landmarks.update(extract_pose_landmarks(results_pose.pose_landmarks))
    else:
        for i in range(23):
            if i in [0, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]:
                landmarks[f'Px{i:02d}'] = None
                landmarks[f'Py{i:02d}'] = None
                landmarks[f'Pz{i:02d}'] = None
    return landmarks

def extract_hand_landmarks(hand_landmarks, side, indices):
    landmarks = {}
    for i, lm in enumerate(hand_landmarks.landmark):
        if i in indices:
            landmarks[f'{side}x{i:02d}'] = lm.x
            landmarks[f'{side}y{i:02d}'] = lm.y
            landmarks[f'{side}z{i:02d}'] = lm.z
    # print("Hand Landmarks: ", len(landmarks))
    return landmarks

def extract_pose_landmarks(pose_landmarks):
    landmarks = {}
    for i, lm in enumerate(pose_landmarks.landmark):
        if i in [0, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]:
            landmarks[f'Px{i:02d}'] = lm.x
            landmarks[f'Py{i:02d}'] = lm.y
            landmarks[f'Pz{i:02d}'] = lm.z
    # print("Pose Landmarks: ", len(landmarks))
    return landmarks

def process_video(results, final, lang):
    frame_count = len(results)
    target_frames = 20
    skip_frames = max(1, int(frame_count/target_frames))
    frame_c = 0
    coord_frames = []
    n_frame = 1

    

    for result in results:
        if frame_c % skip_frames == 0:
             # Extract and print landmark coordinates
            frame_list = {'Frame': n_frame}
            indices = [0, 1, 4, 5, 8, 9, 12, 13, 16, 17, 20]
            landmarks = extract_landmarks(result, indices)
            frame_list.update(landmarks)
            coord_frames.append(frame_list)
            # Adjust skip_frames based on the current progress
            if(n_frame < target_frames):
                skip_frames = max(1, int((frame_count - frame_c) / (target_frames - n_frame)))
            n_frame += 1
        frame_c += 1
    
    padded_df = pd.DataFrame.from_records(coord_frames)
    x_columns = ['Px00', 'Px09', 'Px10', 'Px11', 'Px12', 'Px13', 'Px14', 'Px15', 'Px16', 'Px17', 'Px18', 'Px19', 'Px20', 'Px21', 'Px22', 'Lx00', 'Lx01', 'Lx04', 'Lx05', 'Lx08', 'Lx09', 'Lx12', 'Lx13', 'Lx16', 'Lx17', 'Lx20', 'Rx00', 'Rx01', 'Rx04', 'Rx05', 'Rx08', 'Rx09', 'Rx12', 'Rx13', 'Rx16', 'Rx17', 'Rx20']
    y_columns = ['Py00', 'Py09', 'Py10', 'Py11', 'Py12', 'Py13', 'Py14', 'Py15', 'Py16', 'Py17', 'Py18', 'Py19', 'Py20', 'Py21', 'Py22', 'Ly00', 'Ly01', 'Ly04', 'Ly05', 'Ly08', 'Ly09', 'Ly12', 'Ly13', 'Ly16', 'Ly17', 'Ly20', 'Ry00', 'Ry01', 'Ry04', 'Ry05', 'Ry08', 'Ry09', 'Ry12', 'Ry13', 'Ry16', 'Ry17', 'Ry20']
    z_columns = ['Pz00', 'Pz09', 'Pz10', 'Pz11', 'Pz12', 'Pz13', 'Pz14', 'Pz15', 'Pz16', 'Pz17', 'Pz18', 'Pz19', 'Pz20', 'Pz21', 'Pz22', 'Lz00', 'Lz01', 'Lz04', 'Lz05', 'Lz08', 'Lz09', 'Lz12', 'Lz13', 'Lz16', 'Lz17', 'Lz20', 'Rz00', 'Rz01', 'Rz04', 'Rz05', 'Rz08', 'Rz09', 'Rz12', 'Rz13', 'Rz16', 'Rz17', 'Rz20']

    padded_df[x_columns] = padded_df[x_columns].fillna(padded_df[x_columns].mean())
    padded_df[y_columns] = padded_df[y_columns].fillna(1.2)
    padded_df[z_columns] = padded_df[z_columns].fillna(padded_df[z_columns].mean())

    padded_df[x_columns] = padded_df[x_columns].div(padded_df['Px12'], axis=0)
    padded_df[y_columns] = padded_df[y_columns].div(padded_df['Py12'], axis=0)
    padded_df[z_columns] = padded_df[z_columns].div(padded_df['Pz12'], axis=0)

    padded_df[x_columns] = padded_df[x_columns].sub(padded_df['Px11'], axis=0)
    padded_df[y_columns] = padded_df[y_columns].sub(padded_df['Py11'], axis=0)
    padded_df[z_columns] = padded_df[z_columns].sub(padded_df['Pz11'], axis=0)

    print(padded_df)
    
    print(f'Preprocessed frames: {len(padded_df)}')

    # Excerpt from active file capture_video_LSTM.py, lines 147 to 172
    padded_df = padded_df[:19]

    features_scaled = scaler.transform(padded_df[['Frame'] + x_columns + y_columns + z_columns])
    features_reshaped = features_scaled.reshape(1, 19, 112)

    prediction = predict_with_check(features_reshaped)
    if prediction is not None:
        pred = np.argmax(prediction, axis=1)
        final.append(words[pred[0]])
        print(prediction.round(2))
        return words[pred[0]], final
    else:
        print("Prediction failed due to NaNs in the output.")
        return None, final

def capture_video(lang):    
    monitors = get_monitors()
    monitor = monitors[0]
    width, height = monitor.width, monitor.height
    cap = cv2.VideoCapture(0)
    cap.set(3, width)
    cap.set(4, height)
    cap.set(5, 30) 
    cap.set(10, 150) #brightness

    final=[]
    results=[]
    threshold = 0
    res = None
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results_hands = hands.process(image)
        results_pose = pose.process(image)
        
        if results_pose.pose_landmarks and results_pose.pose_landmarks.landmark[11].visibility>=0.86 and results_pose.pose_landmarks.landmark[12].visibility>=0.86:
            if results_hands.multi_hand_landmarks:
                results.append(image)
                threshold = 0
                cv2.putText(frame, "Detecting gestures", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                
            else:
                if threshold < 8 or len(results) == 0:
                    threshold += 1
                    cv2.putText(frame, "No hands detected", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                else:
                    cv2.putText(frame, "Preprocessing your frame. Kindly wait.", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    print('Collected frames')
                    # print(frames)
                    print(len(results))
                    if(len(results)>=19):
                        res, final = process_video(results, final, lang)
                    elif (len(results) > 16):
                        for i in range(19 - len(results)):
                            results.append(frame)
                        res, final = process_video(results, final, lang)
                    else:
                        res='Please gesture slowly'

                    tts = _TTS()
                    tts.start(res)
                    del(tts)
                    results=[]
            
        else:
            cv2.putText(frame, "Shoulders not detected", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 5)
            results=[]
        
        if res is not None:
            cv2.putText(frame , res, (10, 680), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        cv2.imshow('Sign Detection', frame)
        cv2.waitKey(1)
        
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    
    print(final)
    cap.release()
    cv2.destroyAllWindows()
    # return ' '.join(final)   
    return final
    
# capture_video('Hindi')