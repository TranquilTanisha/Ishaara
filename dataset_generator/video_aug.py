from vidaug import augmentors as va
import cv2
import numpy as np
import os

# Define the augmentation pipeline
seq1 = va.Sequential([
    va.CenterCrop(size=(1080, 1080)), 
    va.HorizontalFlip() # horizontally flip the video with 50% probability
])
seq4 = va.Sequential([
    va.CenterCrop(size=(1080, 1080)), # randomly crop video with a size of (240 x 180)
])
seq7 = va.Sequential([
    va.Downsample(), # downsample the video by a factor of 2
])
seq8 = va.Sequential([
    va.Upsample(), # upsample the video by a factor of 2
])
seq9 = va.Sequential([
    va.Downsample(), # downsample the video by a factor of 2
    va.HorizontalFlip() # horizontally flip the video with 50% probability
])
seq10 = va.Sequential([
    va.Upsample(), # upsample the video by a factor of 2
    va.HorizontalFlip() # horizontally flip the video with 50% probability
])

def look_for_videos_in_folder(folder_path):
    for folder in os.listdir(folder_path):
        folder_p = os.path.join(folder_path, folder)
        if os.path.isdir(folder_p):
            process_videos_in_folder(folder_p)

def process_videos_in_folder(folder_path):
    for video_file in os.listdir(folder_path):
        if video_file.endswith(('.mp4')):
            video_path = os.path.join(folder_path, video_file)
            print(f"Processing {video_file}...")
            process_video(video_path)

def process_video(video_path):
    frames = []

    cap = cv2.VideoCapture(video_path)
    ret = True
    while ret:
        ret, img = cap.read() # read one frame from the 'capture' object; img is (H, W, C)
        if ret:
            frames.append(img)
    video = np.stack(frames, axis=0) # dimensions (T, H, W, C)
    # Apply the augmentation pipeline to the video
    video_aug1 = seq1(video)
    video_aug4 = seq4(video)
    video_aug7 = seq7(video)
    video_aug8 = seq8(video)
    video_aug9 = seq9(video)
    video_aug10 = seq10(video)

    video_name = video_path.split('/')[-1].split('.')[0]

    # Save the augmented videos
    save_video(video_aug1, video_path, f'{video_name}_1')
    save_video(video_aug4, video_path, f'{video_name}_4')
    save_video(video_aug7, video_path, f'{video_name}_7')
    save_video(video_aug8, video_path, f'{video_name}_8')
    save_video(video_aug9, video_path, f'{video_name}_9')
    save_video(video_aug10, video_path, f'{video_name}_10')


def save_video(video, video_path, video_name):
     # Ensure video is a NumPy array if it's not already
    if not isinstance(video, np.ndarray):
        video = np.array(video)
    num_frames, height, width, _ = video.shape

    filename = f"{video_name}.mp4"
    codec_id = "mp4v"
    fourcc = cv2.VideoWriter_fourcc(*codec_id)
    out = cv2.VideoWriter(filename, fourcc=fourcc, fps=20, frameSize=(width, height))

    # Write each frame to the video file
    for frame in video:
        out.write(frame)

    # Release the VideoWriter object
    out.release()



folder_path = 'Words'
# Look for videos in each folder and apply the augmentation pipeline
look_for_videos_in_folder(folder_path)
# process_videos_in_folder(folder_path)