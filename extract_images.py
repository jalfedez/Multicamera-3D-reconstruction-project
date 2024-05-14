import cv2
import os
from pathlib import Path

def extract_frames(video_file, output_folder, camera):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Open the video file
    cap = cv2.VideoCapture(str(video_file))

    frame_no = 0

    while (cap.isOpened()):
        ret, frame = cap.read()
        if frame_no % 30 == 0:
            target = os.path.join(output_folder, f'{frame_no}'+'cam'+f'{camera}.png')
            cv2.imwrite(target, frame)

        frame_no += 1

        if frame_no > 50 * 30:
            break

    # Release video capture object
    cap.release()

if __name__ == "__main__":
    #path to the videos
    video_files = [
        "pile_harbour/cam0_pile.mp4",
        "pile_harbour/cam2_pile.mp4",
        "pile_harbour/cam3_pile.mp4",
        "pile_harbour/cam4_pile.mp4",
        "pile_harbour/cam5_pile.mp4"
    ]

    for i, video_file in enumerate(video_files):
        if i == 1:
            i = 2
        camera = i
        output_folder = f"pile_harbour/notedited/output_frames_cam{camera}"
        extract_frames(video_file, output_folder, camera)
        if i == 4:
            camera = i + 1
            output_folder = f"output_framescam{i+1}"
            extract_frames("pile_harbour/cam5_pile.mp4", output_folder, camera)
