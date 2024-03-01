import scrapetube
from pytube import YouTube
from moviepy.editor import VideoFileClip, concatenate_videoclips, vfx
import os

def create_duplicates(title):
    # path='videos/'+title+'.mp4'
    path=title+'.mp4'
    clip = VideoFileClip(path)
    dur=clip.duration
    print(dur)
    os.remove(path)
    title.lower()
    clip1 = clip.subclip(0, dur/2)
    clip1.write_videofile('videos/'+title+'1.mp4', codec="libx264", fps=24)
    clip2 = clip.subclip(dur/2, dur)
    clip2.write_videofile('videos/'+title+'2.mp4', codec="libx264", fps=24)
    mirrored_clip = clip1.fx(vfx.mirror_x)
    mirrored_clip.write_videofile('videos/'+title+'3.mp4', codec="libx264", fps=24)
    
def view_contents():
    for file in os.listdir('videos/'):
        create_duplicates(file.split('.')[0])

def extract_videos():
    videos = scrapetube.get_channel("UCmM7KPLEthAXiPVAgBF6rhA")
    
    # f=open('videos.txt', 'w')
    for video in videos:
        details=video['title']['accessibility']['accessibilityData']['label']
        title=video['title']['runs'][0]['text']
        details=details.split('ago')
        duration=details[1].split()
        if 'minutes,' not in duration and 'minute,' not in duration and 'hour,' not in duration and 'hours,' not in duration and 'seconds' in duration:
            if len(title.split())==1:
                # dur=duration[-2] #no of seconds
                video_url = 'https://www.youtube.com/watch?v='+video['videoId']
                yt = YouTube(video_url)

                try:
                    stream = yt.streams.get_highest_resolution()
                    stream.download()
                    # stream.download(output_path='videos/')
                    # f.write(title+' '+str(dur)+'\n')
                    create_duplicates(title)
                    
                except Exception as e:
                    print(f"Error downloading video: {e}")
                    
    # f.close()
                    
extract_videos()
# view_contents()
#create_duplicates('tie')