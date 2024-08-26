import cv2
import numpy as np

def read_video(video_path : str) -> list[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    return frames

def save_video(output_video_frames : list[np.ndarray], output_video_path : str):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, 24, 
        (output_video_frames[0].shape[1], output_video_frames[0].shape[0]))
    
    for frame in output_video_frames:
        out.write(frame)
    out.release()
