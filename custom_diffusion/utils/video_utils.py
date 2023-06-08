def video_to_frames(video_path, output_path, frame_rate=1):
    """
    This function takes a video file, separates it into frames,
    and saves them in the designated output folder.

    Args:
    video_path (str): Path to the video file to be dissected.
    output_path (str): Directory path where the dissected frames will be saved.
    frame_rate (int): Determines the frequency of frames to be saved, every 'frame_rate' frame will be saved.

    Returns:
    None
    """
    import os

    import cv2

    # Create VideoCapture object
    vidcap = cv2.VideoCapture(video_path)

    # Check if output directory exists, if not, create it.
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    count = 0
    success = True
    frame_id = 0

    while success:
        # Read frame
        success, image = vidcap.read()

        if success:
            # Only save frame if it is the right id (based on frame_rate)
            if frame_id % frame_rate == 0:
                # Save frame to specified output path with zero-padded file name
                cv2.imwrite(os.path.join(output_path, f"{str(count).zfill(5)}.jpg"), image)
                count += 1

            frame_id += 1

    print("Video frames saved successfully!")

    return output_path


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

    print("Video trimmed successfully!")

    return output_path


def frames_to_video(folder_path, output_folder, output_video_name="output.avi", duration=10):
    """
    This function takes a folder with image files, orders them, and creates a video file from them.
    The video is then saved in the designated output folder.

    Args:
    folder_path (str): Path to the folder with image files.
    output_folder (str): Directory path where the video will be saved.
    output_video_name (str): The name of the output video file.
    duration (int): The desired duration of the output video in seconds.

    Returns:
    None
    """
    import glob
    import os

    import cv2

    # Get the list of images
    img_array = []
    for filename in sorted(glob.glob(os.path.join(folder_path, "*.jpg"))):
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width, height)
        img_array.append(img)

    # Calculate fps based on the desired duration
    fps = len(img_array) / duration

    # Check if output directory exists, if not, create it.
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Create a VideoWriter object
    out = cv2.VideoWriter(os.path.join(output_folder, output_video_name), cv2.VideoWriter_fourcc(*"DIVX"), fps, size)

    for i in range(len(img_array)):
        out.write(img_array[i])

    out.release()

    print("Video created successfully!")

    return output_folder
