import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from screeninfo import get_monitors
import pyttsx3
import pickle

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

speaker=pyttsx3.init()
voices=speaker.getProperty('voices')
speaker.setProperty('voice', voices[1].id)
speaker.setProperty('rate', 120)

with open('lstmmodel.pkl', 'rb') as f:
    model = pickle.load(f)
    
with open ('words.txt', 'r') as f:
    words = f.read().splitlines()

with open ('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

    
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

def process_video(frames, final, lang):
    frame_count = len(frames)
    target_frames = 21
    skip_frames = max(1, int(frame_count/target_frames))
    # print(f'Interval: {skip_frames}')
    frame_c = 0
    coord_frames = []
    n_frame = 1

    for frame in frames:
        if frame_c % skip_frames == 0:
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process the frame
            results_hands = hands.process(image)
            results_pose = pose.process(image)

            frame_list = {}

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

            coord_frames.append(append_landmarks(left_hand_landmarks, right_hand_landmarks, pose_landmarks, n_frame, frame_list))
            # Adjust skip_frames based on the current progress
            if(n_frame != target_frames):
                skip_frames = max(1, int((frame_count - frame_c) / (target_frames - n_frame)))
            n_frame += 1
        frame_c += 1

    import numpy as np

    # Assuming frames is a list of dictionaries where each dictionary represents a frame
    # and contains the same keys (features)

    # Extract the keys (features) from the first dictionary
    features = list(coord_frames[0].keys())

    # Initialize an empty list to hold the transformed frames
    frames_2d = []

    # Iterate over each frame and append its values to frames_2d
    for coord_frame in coord_frames:
        frames_2d.append([coord_frame[feature] for feature in features])

    # Convert frames_2d to a NumPy array
    frames_array = np.array(frames_2d)

    print(f'Preprocessed frames:{len(coord_frames)}')
    if(len(coord_frames)==21):
        #delete the last frame
        frames_array = np.delete(frames_array, -1, axis=0)

    new_data_scaled = scaler.transform(frames_array)
    new_data_reshaped = new_data_scaled.reshape((-1, 20, new_data_scaled.shape[1]))
    


    prediction = model.predict(new_data_reshaped)
    pred = np.argmax(prediction, axis=1) 
    final.append(words[pred[0]])
    print(words[pred[0]])
    speaker.say(words[pred[0]])
    speaker.runAndWait()

def capture_video(lang):    
    monitors = get_monitors()
    monitor = monitors[0]
    width, height = monitor.width, monitor.height
    cap = cv2.VideoCapture(1)
    cap.set(3, width)
    cap.set(4, height)
    cap.set(5, 20) 
    cap.set(10, 150) #brightness

    final=[]
    frames=[]
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
                    if(len(frames)>19):
                        process_video(frames, final, lang)
                    else:
                        cv2.putText(frame, "Wait", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                        speaker.say('Please gesture slowly')
                        speaker.runAndWait()
                    frames=[]
            
        else:
            # speaker.say(translate_to_english("Shouders not detected", lang, 'en'))
            # speaker.say(translate_to_english("Start now", lang, 'en'))
            cv2.putText(frame, "Both shoulders not detected. Kindly repeat your gesture.", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 5)
            frames=[]
        cv2.imshow('Sign Detection', frame)
        
        
        if cv2.waitKey(2) & 0xFF == ord('q'):
            break
    
    print(final)
    cap.release()
    cv2.destroyAllWindows()
    # return ' '.join(final)   
    return final
    
# capture_video('Hindi')