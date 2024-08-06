from vidaug import augmentors as va
import cv2
import numpy as np
import os
import itertools


operations = [va.CenterCrop(size=(1080, 1080)), 
    va.HorizontalFlip(),
    va.Upsample(),
    va.RandomRotate(-10, 10),
    va.RandomShear([-50, 50], [-50, 50]),
    va.RandomTranslate([-50, 50], [-50, 50])]
# Define the augmentation pipeline
seq1 = va.Sequential([
    va.CenterCrop(size=(1080, 1080)), 
    va.HorizontalFlip() # horizontally flip the video with 50% probability
])

def get_unique_combinations(operations):
    """Generate all unique combinations of operations."""
    return list(itertools.combinations(operations, 2)) + list(itertools.combinations(operations, 3)) + list(itertools.combinations(operations, 4))

def look_for_videos_in_folder(folder_path, combinations):
    for folder in os.listdir(folder_path):
        folder_p = os.path.join(folder_path, folder)
        if os.path.isdir(folder_p):
            process_videos_in_folder(folder_p, combinations)

def process_videos_in_folder(folder_path, combinations):
    for video_file in os.listdir(folder_path):
        if video_file.endswith(('.mp4')):
            video_path = os.path.join(folder_path, video_file)
            print(f"Processing {video_file}...")
            process_video(video_path, combinations)

def process_video(video_path, combinations):
    frames = []

    cap = cv2.VideoCapture(video_path)
    ret = True
    while ret:
        ret, img = cap.read() # read one frame from the 'capture' object; img is (H, W, C)
        if ret:
            frames.append(img)
    video = np.stack(frames, axis=0) # dimensions (T, H, W, C)
    video_name = video_path.split('/')[-1].split('.')[0]
    for i, combo in enumerate(combinations):
        seq = va.Sequential(combo)
        video_augmented = seq(video_path)
        
        # Save the augmented video with a unique name
        save_video(video_augmented, video_path, f'{video_name}_{i + 1}')
    


def save_video(video, video_path, video_name):
     # Ensure video is a NumPy array if it's not already
    if not isinstance(video, np.ndarray):
        video = np.array(video)
    num_frames, height, width, _ = video.shape

    filename = f"{video_name}.mp4"
    codec_id = "mp4v"
    fourcc = cv2.VideoWriter_fourcc(*codec_id)
    out = cv2.VideoWriter(filename, fourcc=fourcc, fps=30, frameSize=(width, height))

    # Write each frame to the video file
    for frame in video:
        out.write(frame)

    # Release the VideoWriter object
    out.release()


combinations = get_unique_combinations(operations)
folder_path = 'Words'
# Look for videos in each folder and apply the augmentation pipeline
look_for_videos_in_folder(folder_path, combinations)
# process_videos_in_folder(folder_path)