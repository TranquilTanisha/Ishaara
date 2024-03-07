import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from screeninfo import get_monitors
import pyttsx3
import pickle
from app import translate_to_english

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

speaker=pyttsx3.init()
voices=speaker.getProperty('voices')
speaker.setProperty('voice', voices[1].id)
speaker.setProperty('rate', 120)

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
    
with open ('words.txt', 'r') as f:
    words = f.read().splitlines()
    
languages={'Hindi': 'hi-IN',
'Bengali': 'bn-IN',
'Telugu': 'te-IN',
'Marathi': 'mr-IN',
'Tamil': 'ta-IN',
'Urdu': 'ur-IN',
'Gujarati': 'gu-IN',
'Malayalam': 'ml-IN',
'Kannada': 'kn-IN',
'Odia': 'or-IN',
'Punjabi': 'pa-IN',
'Assamese': 'as-IN'}

def normalize_dict(data):
    values=list(data.values())
    min_value = min(values)
    max_value = max(values)
    range_value = max_value - min_value

    # Normalize numeric values
    # normalized_data = {"ClassLabel": data["ClassLabel"]}
    for key, value in data.items():
        data[key] = (value - min_value) / range_value
    return data
    
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
        frame_list[f'X{"{:02d}".format(i)}{"{:02d}".format(n_frame)}'] -= frame_list[f'X53{"{:02d}".format(n_frame)}']
        frame_list[f'Y{"{:02d}".format(i)}{"{:02d}".format(n_frame)}'] -= frame_list[f'Y53{"{:02d}".format(n_frame)}']
        frame_list[f'Z{"{:02d}".format(i)}{"{:02d}".format(n_frame)}'] -= frame_list[f'Z53{"{:02d}".format(n_frame)}']

    return frame_list 

def process_video(frames, final, lang):
    frame_count = len(frames)
    target_frames = 20
    skip_frames = max(1, int(frame_count/target_frames))
    # print(f'Interval: {skip_frames}')
    frame_c = 0
    frame_list = {}
    n_frame = 1

    for frame in frames:
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame to detect hands
        results_hands = hands.process(image)
        # Process the frame to detect pose
        results_pose = pose.process(image)
        # print(f"Skip?: {frame_c % skip_frames != 0}")
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
            if(len(frame_list)/198 < target_frames - 1):
                skip_frames = max(1, int((frame_count - frame_c) / (target_frames - len(frame_list)/198)))
            else:
                break
            n_frame += 1
        frame_c += 1

    print(f'Preprocessed frames:{len(frame_list)/198}')
    # print(frame_list.keys())
    frame_list=normalize_dict(frame_list)
    #prediction of the model
    res=model.predict([list(frame_list.values())])
    # res=model.predict(np.array([list(frame_list.values())]))
    final.append(words[res[0]])
    speaker.say(translate_to_english(words[res[0]], lang, 'en'))

def capture_video(lang):    
    monitors = get_monitors()
    monitor = monitors[0]
    width, height = monitor.width, monitor.height
    cap = cv2.VideoCapture(0)
    cap.set(3, width)
    cap.set(4, height)
    cap.set(5, 20) 
    cap.set(10, 150) #brightness
    
    frames=[]
    final=[]
    threshold = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results_hands = hands.process(image)
        results_pose = pose.process(image)
        
        if results_pose.pose_landmarks:
            left_shoulder = results_pose.pose_landmarks.landmark[11].visibility>=0.86
            right_shoulder=results_pose.pose_landmarks.landmark[12].visibility>=0.86
        else:
            left_shoulder = False
            right_shoulder = False
        
        if left_shoulder and right_shoulder:
            if results_hands.multi_hand_landmarks:
                frames.append(frame)
                threshold = 0
                cv2.putText(frame, "Detecting gestures", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                
            else:
                if threshold < 5 or len(frames) == 0:
                    threshold += 1
                    cv2.putText(frame, "No hands detected", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                else:
                    cv2.putText(frame, "Preprocessing your frame. Kindly wait.", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    print('Collected frames')
                    # print(frames)
                    print(len(frames))
                    if(len(frames)>18):
                        process_video(frames, final, lang)
                    else:
                        cv2.putText(frame, "Please gesture slowly, the system could not catch that", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    frames=[]
            
        else:
            # speaker.say(translate_to_english("Shouders not detected", lang, 'en'))
            # speaker.say(translate_to_english("Start now", lang, 'en'))
            cv2.putText(frame, "Both shoulders not detected. Kindly repeat your gesture.", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 5)
            frames=[]
        cv2.imshow('Sign Detection', frame)
        
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    print(final)
    cap.release()
    cv2.destroyAllWindows()
    
capture_video('Hindi')