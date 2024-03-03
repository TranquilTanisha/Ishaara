import os

def look_for_videos_in_folder(folder_path):
    for folder in os.listdir(folder_path):
        print(folder)
        folder_p = os.path.join(folder_path, folder)
        if os.path.isdir(folder_p):
            process_videos_in_folder(folder_p)

def process_videos_in_folder(folder_path):
    for video_file in os.listdir(folder_path):
        if video_file.endswith(('7.mp4')):
            video_path = os.path.join(folder_path, video_file)
            print(f"Processing {video_file}...")
            process_video(video_path)
        if video_file.endswith(('10.mp4')):
            video_path = os.path.join(folder_path, video_file)
            print(f"Processing {video_file}...")
            process_video(video_path)
        if video_file.endswith(('8.mp4')):
            video_path = os.path.join(folder_path, video_file)
            print(f"Processing {video_file}...")
            process_video(video_path)
        if video_file.endswith(('9.mp4')):
            video_path = os.path.join(folder_path, video_file)
            print(f"Processing {video_file}...")
            process_video(video_path)

def process_video(video_path):
    os.remove(video_path) 


folder_path = 'Words'
look_for_videos_in_folder(folder_path)