def download_video(
    video_url: str = "https://www.youtube.com/watch?v=8QRG4vzbdE0",
    output_path: str = "output_videos",
    quality: str = "720p",
    filename: str = "downloaded_video.mp4",
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
    import os

    from pytube import YouTube

    # Create a YouTube object
    yt = YouTube(video_url)

    # Find the stream with the desired quality that also contains audio
    video_stream = yt.streams.filter(progressive=True, file_extension="mp4", res=quality).first()

    if video_stream is not None:
        # Download the stream
        video_stream.download(output_path, filename=filename)
        print("Video downloaded successfully!")
    else:
        print(f"No video stream found for {quality} quality.")

    save_path = os.path.join(output_path, filename)
    return save_path
