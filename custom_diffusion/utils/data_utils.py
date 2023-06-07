def download_video(
    video_url: str ="https://www.youtube.com/watch?v=8QRG4vzbdE0",
    output_path: str = 'output_videos',
    quality: str = '720p',
    filename: str = 'downloaded_video.mp4'
    ):
    """
    This function downloads a video and corresponding audio from YouTube and saves it in the designated output folder.

    Args:
    video_url (str): URL of the YouTube video to be downloaded.
    output_path (str): Directory path where the downloaded video will be saved.
    quality (str): Desired quality of the video to be downloaded. It can be '144p', '240p', '360p', '480p', '720p', '1080p', '1440p' (for 2K), or '2160p' (for 4K).
    filename (str): Desired filename of the downloaded video.

    Returns:
    None
    """
    from pytube import YouTube
    import os

    # Create a YouTube object
    yt = YouTube(video_url)

    # Find the stream with the desired quality that also contains audio
    video_stream = yt.streams.filter(progressive=True, file_extension='mp4', res=quality).first()
    
    if video_stream is not None:
        # Download the stream
        video_stream.download(output_path, filename=filename)
        print('Video downloaded successfully!')
    else:
        print(f'No video stream found for {quality} quality.')
    
    save_path = os.path.join(output_path, filename)
    return save_path


def video_to_frames(video_path, output_path):
    """
    This function takes a video file, separates it into frames, 
    and saves them in the designated output folder.

    Args:
    video_path (str): Path to the video file to be dissected.
    output_path (str): Directory path where the dissected frames will be saved.

    Returns:
    None
    """
    import cv2
    import os
    
    
    # Create VideoCapture object
    vidcap = cv2.VideoCapture(video_path)

    # Check if output directory exists, if not, create it.
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    count = 0
    success = True

    while success:
        # Read frame
        success, image = vidcap.read()

        if success:
            # Save frame to specified output path
            cv2.imwrite(os.path.join(output_path, f'frame{count}.jpg'), image)  
            count += 1


def trim_video(video_path: str, output_path: str, start_time: int, end_time: int):
    """
    This function trims a video clip from the given start time to the end time.

    Args:
    video_path (str): Path to the input video file.
    output_path (str): Path to save the output trimmed video file.
    start_time (int): The start time of the clip in seconds.
    end_time (int): The end time of the clip in seconds.

    Returns:
    None
    """
    from moviepy.editor import VideoFileClip

    # Load the video
    clip = VideoFileClip(video_path)

    # Trim the video
    trimmed_clip = clip.subclip(start_time, end_time)

    # Write the result to a file (without processing audio)
    trimmed_clip.write_videofile(output_path, audio=True)

    print('Video trimmed successfully!')
    
    return None
