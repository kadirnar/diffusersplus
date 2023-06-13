from typing import List


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
                cv2.imwrite(os.path.join(output_path, f"{str(count).zfill(5)}.png"), image)
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


def convert_images_to_video(image_list: List[str], output_file: str, frame_rate: int = 30):
    """
    Converts a list of PIL Images to a video file.

    Args:
        image_list (List[Image]): A list of PIL Images.
        output_file (str): The name of the output video file.
        frame_rate (int, optional): The frame rate of the output video. Defaults to 30.
    """
    import cv2
    import numpy as np

    # Convert PIL Images to NumPy arrays
    frames = [np.array(img) for img in image_list]

    # Get the resolution of the video
    height, width, layers = frames[0].shape
    size = (width, height)

    # Create an OpenCV VideoWriter object
    video_writer = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*"mp4v"), frame_rate, size)

    # Write each frame to the video file
    for frame in frames:
        video_writer.write(frame)

    # Close the VideoWriter when done
    video_writer.release()

    print("Images converted to video successfully!")

    return output_file


def video_pipeline(
    video_path: str = "test.mp4",
    output_path: str = "output",
    start_time: int = 0,
    end_time: int = 5,
    frame_rate: int = 1,
):
    edit_video = trim_video(video_path=video_path, output_path=output_path, start_time=start_time, end_time=end_time)
    video2frame = video_to_frames(video_path=edit_video, output_path=output_path, frame_rate=frame_rate)

    return video2frame
