from ultralytics import YOLO
import supervision as sv
import pickle
import os
import sys
import cv2 as cv
import numpy as np
import pandas as pd
sys.path.append("../")
from utils import get_center, get_bbox_width , get_foot_position

class Tracker: 
    def __init__(self, model_path) -> None:
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()
     
    
    def add_position_to_tracks(self, tracks):
        for object , object_tracks in tracks.items() : 
            for frame_num , track in enumerate(object_tracks):
                for track_id , track_info in track.items() : 
                        bbox = track_info['bbox']
                        if object == "ball" :
                           position = get_center(bbox)
                        else : 
                           position = get_foot_position(bbox)
                        tracks[object][frame_num][track_id]['position'] = position
    def get_bbox(self, frames): 
        detections = []
        batch_size = 20
        for i in range(0, len(frames), batch_size):
            detections_batch = self.model.predict(frames[i:i+batch_size],conf=0.1)
            detections += detections_batch
        return detections 
    
    def get_ball_interpolation(self , ball_pos) : 
        ball_positions = [x.get(1 , {}).get('bbox' , {}) for x in ball_pos]
        df_ball_positions = pd.DataFrame(ball_positions , columns=['x1','y1','x2','y2'])
        
        # interpolate missing values 
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()
        
        ball_positions = [{1: {"bbox":x}} for x in df_ball_positions.to_numpy().tolist()]
        
        return ball_positions
    
    def get_tracker(self, frames, read_from_stub=False, stub_path=None):
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                tracks = pickle.load(f)
            return tracks

        detections = self.get_bbox(frames)
        tracks = {
            "players": [],
            "referee": [],
            "ball": []
        }

        for frame_num, detection in enumerate(detections):
                cls_names = detection.names
                cls_names_inv = {v: k for k, v in cls_names.items()}  # Fix typo: 'itmes()' -> 'items()'
                
                # Convert to supervision Detections
                sv_detection = sv.Detections.from_ultralytics(detection)
                
                # Convert goalkeeper to player
                for obj_ind, obj_id in enumerate(sv_detection.class_id):
                    if cls_names[obj_id] == 'goalkeeper':
                        sv_detection.class_id[obj_ind] = cls_names_inv['player']
                
                # Track objects
                detection_with_tracks = self.tracker.update_with_detections(sv_detection)
                
                # Initialize frame-specific tracking
                tracks["players"].append({})
                tracks["ball"].append({})
                tracks["referee"].append({})
                
                for frame_detection in detection_with_tracks:
                    bbox = frame_detection[0].tolist()
                    cls_id = frame_detection[3]
                    track_id = frame_detection[4]
                    
                    if cls_id == cls_names_inv['player']:
                        tracks["players"][frame_num][track_id] = {"bbox": bbox}
                    elif cls_id == cls_names_inv['referee']:
                        tracks["referee"][frame_num][track_id] = {"bbox": bbox}
                
                # Handle ball tracking separately
                for frame_detection in sv_detection:
                    bbox = frame_detection[0].tolist()
                    cls_id = frame_detection[3]
                    
                    if cls_id == cls_names_inv["ball"]:
                        tracks["ball"][frame_num][1] = {"bbox": bbox}
            
        if stub_path is not None:
            with open(stub_path, "wb") as f:
                pickle.dump(tracks, f)

        return tracks
    
    def draw_ellipse(self, frame, bbox, color, track_id=None): 
        y2 = int(bbox[3])
        x_center, _ = get_center(bbox)
        width = get_bbox_width(bbox)
        cv.ellipse(
            frame,
            (x_center, y2),
            (int(width), int(0.3 * width)),
            angle=0.0 , 
            startAngle=-45,
            endAngle=235, 
            color=color,
            thickness=2,
            lineType=cv.LINE_4
        )
        if track_id is not None: 
            rec_width = 40 
            rec_height = 20 
            x1_rec = int(x_center - rec_width // 2)
            x2_rec = int(x_center + rec_width // 2)
            
            y1_rec = int(y2 - rec_height // 2) + 15
            y2_rec = int(y2 + rec_height // 2) + 15
            
            cv.rectangle(frame,
                         (x1_rec, y1_rec),
                         (x2_rec, y2_rec), 
                         color, 
                         cv.FILLED)
            
            x1_track = x1_rec + 12 
            if track_id > 99: 
                x1_track = x1_track - 10
            
            cv.putText(frame,
                       str(track_id),
                       (x1_track, y1_rec + 15), 
                       cv.FONT_HERSHEY_SIMPLEX, 
                       0.6, 
                       (0, 0, 0),
                       2)
        return frame
    
    def draw_triangle(self, frame, bbox, color): 
        x, _ = get_center(bbox)
        y = int(bbox[1])
        
        coordinates = np.array([
            [x, y], 
            [x - 10, y - 20], 
            [x + 10, y - 20]
        ])
        cv.drawContours(frame, [coordinates], 0, color, cv.FILLED)
        cv.drawContours(frame, [coordinates], 0, (0, 0, 0), 2)
        return frame
    def draw_team_control(self,frame , frame_num , team_control) : 
        overlay = frame.copy()
        cv.rectangle(overlay , (1350,850) , (1900,970) , (255,255,255) , -1)
        alpha = 0.4
        cv.addWeighted(overlay , alpha , frame , 1-alpha , 0 ,frame)
        # team control percent 
        
        team_control_till_frame = team_control[:frame_num+1]
        team_control_1 = 0 
        team_control_2 = 0 
        for item in team_control_till_frame : 
            if item == 1 : 
                team_control_1 = team_control_1 + 1
            else : 
                team_control_2 = team_control_2 + 1
                
        team1 = team_control_1 / (team_control_1+team_control_2)
        team2 = team_control_2 / (team_control_1+team_control_2)
        
        # put text 
        cv.putText(frame , f"team 1 ball control {team1*100:.2f}" , (1400 , 900) , cv.FONT_HERSHEY_COMPLEX , 1 , (0,0,0) , 3)
        cv.putText(frame , f"team 2 ball control {team2*100:.2f}" , (1400 , 950) , cv.FONT_HERSHEY_COMPLEX , 1 , (0,0,0) , 3)
        
        return frame
    def get_annotations(self, video_frames, tracks , team_control): 
        output_video_frames = []
        
        for frame_num, frame in enumerate(video_frames): 
            copy_frame = frame.copy()
            
            player_dict = tracks["players"][frame_num]
            ball_dict = tracks["ball"][frame_num]
            referee_dict = tracks["referee"][frame_num]
            
            for track_id, player in player_dict.items(): 
                bbox = player["bbox"]
                color = player.get("team_color", (0, 0, 255))
                has_ball = player.get("has_ball" , False)
                copy_frame = self.draw_ellipse(copy_frame, bbox, color, track_id)
                if has_ball : 
                    copy_frame = self.draw_triangle(copy_frame , bbox , (0,0,255))
            
            for referee_id, referee in referee_dict.items(): 
                bbox = referee['bbox']
                copy_frame = self.draw_ellipse(copy_frame, bbox, (0, 255, 255))
            
            for track_id, ball in ball_dict.items(): 
                copy_frame = self.draw_triangle(copy_frame, ball["bbox"], (0, 255, 0))
                    
            # team control 
            copy_frame = self.draw_team_control(copy_frame , frame_num, team_control)
            output_video_frames.append(copy_frame)
        
        return output_video_frames
