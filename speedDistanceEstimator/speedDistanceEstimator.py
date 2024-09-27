import cv2 as cv
import numpy as np
import sys
sys.path.append("../")
from utils import get_distance, get_foot_position

class SpeedDistanceEstimator:
    
    def __init__(self):
        self.frame_rate = 24  # Corrected typo (was `frame_reate`)
        self.frame_wind = 5  # Frame window size for speed calculation

    def add_speed_distance_to_tracks(self, tracks):
        total_distance = {}
        
        for object, object_tracks in tracks.items():
            if object == "ball" or object == "referees":
                continue  # Skip ball and referees
            
            num_frames = len(object_tracks)
            
            for frame_num in range(0, num_frames, self.frame_wind):
                last_frame = min(frame_num + self.frame_wind, num_frames)
                
                for track_id, _ in object_tracks[frame_num].items():
                    if track_id not in object_tracks[last_frame - 1]:  # Corrected for last frame access
                        continue 
                    
                    # Get start and end positions for speed calculation
                    start_position = object_tracks[frame_num][track_id]['position_transform']
                    end_position = object_tracks[last_frame - 1][track_id]['position_transform']
                    
                    if start_position is None or end_position is None:
                        continue
                    
                    # Calculate the distance and time between start and end positions
                    distance = get_distance(end_position, start_position)
                    time = (last_frame - frame_num) / self.frame_rate  # Time in seconds
                    
                    # Calculate speed in m/s and convert to km/h
                    speed_ms = distance / time
                    speed_kmh = speed_ms * 3.6
                    
                    # Initialize distance tracking for object
                    if object not in total_distance:
                        total_distance[object] = {}
                    
                    if track_id not in total_distance[object]:
                        total_distance[object][track_id] = 0
                    
                    # Update total distance traveled
                    total_distance[object][track_id] += distance
                    
                    # Update speed and distance for all frames in the window
                    for frame_num_batch in range(frame_num, last_frame):
                        if track_id not in tracks[object][frame_num_batch]:
                            continue
                        
                        tracks[object][frame_num_batch][track_id]["speed"] = speed_kmh
                        tracks[object][frame_num_batch][track_id]["distance"] = total_distance[object][track_id]

    def draw(self, frames, tracks):
        output_frames = []
        
        for frame_num, frame in enumerate(frames):
            for object, object_tracks in tracks.items():
                if object == "ball" or object == "referees":
                    continue  # Skip ball and referees
                
                for _, track_info in object_tracks[frame_num].items():
                    if "speed" in track_info:
                        speed = track_info.get('speed', None)
                        distance = track_info.get('distance', None)
                        
                        if speed is None or distance is None:
                            continue
                        
                        # Get bounding box and foot position to display text
                        bbox = track_info['bbox']
                        position = get_foot_position(bbox)
                        
                        # Adjust position slightly for text display
                        position = list(position)
                        position[1] += 40  # Move the text down a bit
                        position = tuple(map(int, position))  # Convert to integers
                        
                        # Display speed on the frame
                        cv.putText(frame, f"{speed:.2f} km/h", position, cv.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 2)
            
            output_frames.append(frame)
        
        return output_frames