from vidaug import augmentors as va
import cv2
import numpy as np
import os
import itertools


operations = [
    va.CenterCrop(size=(1080, 1080)),
    va.HorizontalFlip(),
    va.RandomTranslate(x = 50, y = 50)]

def get_unique_combinations(operations):
    """Generate all unique combinations of operations."""
    return list(itertools.chain.from_iterable(itertools.combinations(operations, n) for n in range(1, len(operations) + 1)))

def process_videos_in_folder(folder_path, combinations):
    for entry in os.scandir(folder_path):
        if entry.is_dir():
            process_videos_in_subfolder(entry.path, combinations)

def process_videos_in_subfolder(subfolder_path, combinations):
    for entry in os.scandir(subfolder_path):
        if entry.name.endswith(".mp4"):
            video_path = entry.path
            print(f"Processing {entry.name}...")
            process_video(video_path, combinations)

def process_video(video_path, combinations):
    frames = []

    cap = cv2.VideoCapture(video_path)
    ret = True
    while ret:
        ret, img = cap.read() # read one frame from the 'capture' object; img is (H, W, C)
        
        if ret:
            h, w = img.shape[:2]
            aspect_ratio = w / h
            img = cv2.resize(img, (int(1080*aspect_ratio), 1080))
            frames.append(img)
    cap.release()
    video = np.stack(frames, axis=0) # dimensions (T, H, W, C)
    video_name = video_path.split('/')[-1].split('.')[0]
    for i, combo in enumerate(combinations):
        seq = va.Sequential(combo)
        video_augmented = seq(video)
        
        # Save the augmented video with a unique name
        save_video(video_augmented, video_path, f'{video_name}_ver{i + 1}')
    


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
process_videos_in_folder(folder_path, combinations)
# process_videos_in_folder(folder_path)