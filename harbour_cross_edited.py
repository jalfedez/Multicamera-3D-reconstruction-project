import cv2
import os
import numpy as np

# Intrinsic camera matrix parameters and distortion coefficients
camera_parameters = {
    "cam0": {
        "mtx": np.array([[1473.41024473, 0, 926.88076437],
                         [0, 1473.51080614, 556.93029095],
                         [0, 0, 1]]),
        "dist": np.array([-0.34702552, 0.28881827, -0.00115781, 0.00562749])
    },
    "cam2": {
        "mtx": np.array([[1469.26261021, 0, 939.59642256],
                         [0, 1466.570029, 542.36962045],
                         [0, 0, 1]]),
        "dist": np.array([-0.30777328, 0.16462143, -0.0023316, 0.00140443])
    },
    "cam3": {
        "mtx": np.array([[1479.19429782, 0, 957.42429659],
                         [0, 1479.00152361, 523.41014308],
                         [0, 0, 1]]),
        "dist": np.array([-0.3378171, 0.11128436, 0.00498374, 0.00450891])
    },
    "cam4": {
        "mtx": np.array([[1469.58502581, 0, 1021.85805913],
                         [0, 1466.81748786, 612.92579221],
                         [0, 0, 1]]),
        "dist": np.array([-0.32604075, 0.22071807, -0.00058352, 0.01175219])
    },
    "cam5": {
        "mtx": np.array([[1467.51784531, 0, 1024.52158245],
                         [0, 1464.61523789, 537.02103995],
                         [0, 0, 1]]),
        "dist": np.array([-0.33124292, 0.10792365, 0.00737645, 0.00802014])
    }
}


def extract_frames(video_file, camera):
    # Output folder name
    output_folder = f"Cross_Harbour/good_folder/output_frames_cam{camera}"

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

        if frame_no % 1 == 0:
            # Undistort the image
            undistorted_frame = undistort(frame, camera_parameters[f"cam{camera}"]["mtx"],
                                          camera_parameters[f"cam{camera}"]["dist"])

            # Preprocessing operations
            preprocessed_frame = preprocess(undistorted_frame)

            # Segment the cross in the preprocessed image
            segmented_img = segment_cross(preprocessed_frame)

            # Save the segmented image
            target = os.path.join(output_folder, f'frame_{frame_no}_cam{camera}.png')
            cv2.imwrite(target, segmented_img)

        frame_no += 1

        if frame_no > 30 * 30:
            break

    # Release video capture object
    cap.release()


def undistort(img, mtx, dist):
    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
    x, y, w, h = roi
    dst = dst[y:y + h, x:x + w]
    return dst


def preprocess(img):
    # Color balance and contrast
    balanced_frame = cv2.detailEnhance(img)

    blurred = cv2.GaussianBlur(balanced_frame, (5, 5), 0)

    # Create a mask with a white circle in the center and black background
    mask = np.zeros_like(img[:, :, 0])
    height, width = img.shape[:2]
    center_x, center_y = width // 2, height // 2
    radius = min(center_x, center_y)
    cv2.circle(mask, (center_x, center_y), radius, (255), thickness=-1)

    # Apply blur only to the region outside the mask
    blurred_background = cv2.blur(blurred, (30, 30), 0)
    blurred_img = np.where(mask[..., None] == 0, blurred_background, img)

    return blurred_img

def sharpen(img):
    # Define the sharpening kernel
    kernel = np.array([[-1, -1, -1],
                       [-1,  9, -1],
                       [-1, -1, -1]])

    # Apply the sharpening filter
    sharpened = cv2.filter2D(img, -1, kernel)

    return sharpened

def segment_cross(img):
    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to separate the cross from the background
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a black mask of the same size as the image
    mask = np.zeros_like(gray)

    # Draw contours on the mask
    cv2.drawContours(mask, contours, -1, 255, cv2.FILLED)

    # Apply blur to the region not filled with contours
    blurred_background = cv2.blur(img, (15, 15))
    blurred_background = cv2.addWeighted(blurred_background, 1 + 5 / 127, np.zeros(blurred_background.shape, blurred_background.dtype), 0, -25 - 5)

    # Combine the original image with the blurred image using the mask
    result = cv2.bitwise_and(img, img, mask=mask)
    result += cv2.bitwise_and(blurred_background, blurred_background, mask=~mask)

    return result

if __name__ == "__main__":
    # path to the videos
    video_files = [
        "Cross_Harbour/cam0_harbour_cropped.mp4",
        "Cross_Harbour/cam2_harbour_cropped.mp4",
        "Cross_Harbour/cam3_harbour_cropped.mp4",
        "Cross_Harbour/cam4_harbour_cropped.mp4",
        "Cross_Harbour/cam5_harbour_cropped.mp4"
    ]

    for i, video_file in enumerate(video_files):
        if i == 1:
            i = 2
        extract_frames(video_file, i)
        if i == 4:
            camera = 5
            extract_frames("Cross_Harbour/cam5_harbour_cropped.mp4", camera)
