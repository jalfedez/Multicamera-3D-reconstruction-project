import cv2
import os
import numpy as np

# Parámetros de la matriz de cámara intrínseca y coeficientes de distorsión
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
    # Nombre de la carpeta de salida
    output_folder = f"Cross_ASTA/output_frames_cam{camera}"

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

        if frame_no % 25 == 0:
            # Undistorsionar la imagen
            undistorted_frame = undistort(frame, camera_parameters[f"cam{camera}"]["mtx"],
                                          camera_parameters[f"cam{camera}"]["dist"])

            # Aplicar operaciones de preprocesamiento
            preprocessed_frame = preprocess(undistorted_frame)

            # Guardar la imagen preprocesada
            target = os.path.join(output_folder, f'frame_{frame_no}_cam{camera}.png')
            cv2.imwrite(target, preprocessed_frame)

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
    # Balance de color y contraste
    balanced_frame = cv2.detailEnhance(img)

    return balanced_frame


if __name__ == "__main__":
    video_files = [
        "Cross_Harbour/cam0_cropped.mp4",
        "Cross_AHarbour/cam2_cropped.mp4",
        "Cross_Harbour/cam3_cropped.mp4",
        "Cross_Harbour/cam4_cropped.mp4",
        "Cross_Harbour/cam5_cropped.mp4"
    ]

    for i, video_file in enumerate(video_files):
        if i == 1:
            i = 2
        extract_frames(video_file, i)
        if i == 4:
            camera = 5
            extract_frames("Cross_Harbour/cam5_cropped.mp4", camera)
