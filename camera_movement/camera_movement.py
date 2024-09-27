import cv2 as cv
import numpy as np
import pickle
import sys
import os 

sys.path.append("../")
from utils import get_distance

class CameraMovementEstimator:
    def __init__(self, frame):
        self.min_distance = 5
        gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        self.mask_features = np.zeros_like(gray_frame)
        self.mask_features[:, 0:20] = 1
        self.mask_features[:, 900:1050] = 1

    def _adjust_position_tracks(self, tracks, camera_movement_per_frame):
        for object_id, object_track in tracks.items():
            for frame_num, track in enumerate(object_track):
                for track_id, track_info in track.items():
                    position = track_info['position']
                    position_adjusted = self.adjust_position(position, camera_movement_per_frame[frame_num])
                    track_info['position_adjusted'] = position_adjusted

    def adjust_position(self, position, camera_movement):
        return (position[0] - camera_movement[0], position[1] - camera_movement[1])

    def get_camera_movement(self, frames, read_from_stubs=False, stub_path=None):
        if read_from_stubs and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                return pickle.load(f)

        camera_movement = [[0, 0] for _ in range(len(frames))]

        old_gray = cv.cvtColor(frames[0], cv.COLOR_BGR2GRAY)
        old_features = cv.goodFeaturesToTrack(old_gray,
                                              maxCorners=100, 
                                              qualityLevel=0.3,
                                              minDistance=3,
                                              blockSize=7,
                                              mask=self.mask_features)

        for frame_num in range(1, len(frames)):
            frame_gray = cv.cvtColor(frames[frame_num], cv.COLOR_BGR2GRAY)
            new_features, status, _ = cv.calcOpticalFlowPyrLK(old_gray,
                                                              frame_gray,
                                                              old_features,
                                                              None,
                                                              winSize=(15, 15),
                                                              maxLevel=2,
                                                              criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

            if new_features is not None:
                max_distance = 0
                camera_movement_x, camera_movement_y = 0, 0

                good_new = new_features[status == 1]
                good_old = old_features[status == 1]

                for new, old in zip(good_new, good_old):
                    distance = get_distance(new.ravel(), old.ravel())

                    if distance > max_distance:
                        max_distance = distance
                        camera_movement_x = new[0] - old[0]
                        camera_movement_y = new[1] - old[1]

                if max_distance > self.min_distance:
                    camera_movement[frame_num] = [camera_movement_x, camera_movement_y]
                    old_features = cv.goodFeaturesToTrack(frame_gray, 
                                                          maxCorners=100, 
                                                          qualityLevel=0.3,
                                                          minDistance=3,
                                                          blockSize=7,
                                                          mask=self.mask_features)

            old_gray = frame_gray.copy()

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(camera_movement, f)

        return camera_movement

    def draw_camera_movement_estimator(self, output_video_frames, camera_movement_per_frames):
        output_frames = []
        for frame_num, frame in enumerate(output_video_frames):
            frame_copy = frame.copy()
            overlay = frame.copy()

            cv.rectangle(overlay, (0, 0), (200, 100), (255, 255, 255), -1)
            alpha = 0.6
            cv.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame_copy)

            x_movement, y_movement = camera_movement_per_frames[frame_num]

            cv.putText(frame_copy, f"Camera movement X: {x_movement:.2f}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            cv.putText(frame_copy, f"Camera movement Y: {y_movement:.2f}", (10, 60), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

            output_frames.append(frame_copy)

        return output_frames