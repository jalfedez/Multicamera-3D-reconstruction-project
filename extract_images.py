import cv2
import os
from pathlib import Path

def extract_frames(video_file, camera):
    # Output folder name
    output_folder = f"Cross_ASTA/raw/output_frames_cam{camera}"

    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Open the video file
    cap = cv2.VideoCapture(video_file)

    frame_no = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        #frames that you want to extract(bigger number, less frames extracted)
        if frame_no % 2 == 0:

            # Save the undistorted image
            target = os.path.join(output_folder, f'frame_{frame_no}_cam{camera}.png')
            cv2.imwrite(target, frame)

        frame_no += 1

        if frame_no > 30 * 30:
            break

    # Release video capture object
    cap.release()

if __name__ == "__main__":
    #path to the videos
    video_files = [
        "Cross_ASTA/videos_cropped/cam0_edited.mp4",
        "Cross_ASTA/videos_cropped/cam1_edited.mp4",   #this camera does not work, in the for we dont take this file
        "Cross_ASTA/videos_cropped/cam2_edited.mp4",
        "Cross_ASTA/videos_cropped/cam3_edited.mp4",
        "Cross_ASTA/videos_cropped/cam4_edited.mp4",
        "Cross_ASTA/videos_cropped/cam5_edited.mp4"
    ]

    for i, video_file in enumerate(video_files):
        if i == 1:
            i = 2
        extract_frames(video_file, i)
        if i == 4:
            camera = 5
            extract_frames("Cross_ASTA/videos_cropped/cam5_edited.mp4", camera)
 
