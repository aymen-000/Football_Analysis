import cv2 as cv 

def read_video(video_path):
    videoCapture = cv.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = videoCapture.read()
        if not ret:
            break
        frames.append(frame)
    videoCapture.release()  # Release the video capture object after reading
    return frames

def save_video(output_video_path, output_video_frames):
    if not output_video_frames:
        print("No frames to save.")
        return
    
    fourcc = cv.VideoWriter_fourcc(*"XVID")
    height, width = output_video_frames[0].shape[:2]
    out = cv.VideoWriter(output_video_path, fourcc, 24, (width, height))
    
    for frame in output_video_frames:
        out.write(frame)
    
    out.release()
    print(f"Video saved successfully at {output_video_path}")