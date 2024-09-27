import os
import argparse
import uuid
from utils import read_video, save_video
from trackers import Tracker
from team_assigner import TeamAssigner
from assign_ball_player import AssignBallPlayer
from camera_movement import CameraMovementEstimator
from view_transforms import ViewTransforms
from speedDistanceEstimator import SpeedDistanceEstimator

def main(video_path, output_video_path):
    model_path = "./models/best.pt"

    # Ensure the output directory exists
    output_video_dir = os.path.dirname(output_video_path)
    os.makedirs(output_video_dir, exist_ok=True)

    # Generate a unique identifier for the stub files
    unique_id = uuid.uuid4()

    # Read video frames
    frames = read_video(video_path)
    tracker = Tracker(model_path)

    # Use unique stub path for this run
    track_stub_path = f"./stubs/track_stubs_{unique_id}.pkl"
    camera_stub_path = f"./stubs/camera_stub_{unique_id}.pkl"

    # Track players and ball using the tracker
    tracks = tracker.get_tracker(frames=frames,
                                 read_from_stub=True,
                                 stub_path=track_stub_path)
    
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(frames[0], tracks['players'][0])

    for frame_num, player_track in enumerate(tracks["players"]):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(frames[frame_num],
                                                 track['bbox'],
                                                 player_id)
            tracks['players'][frame_num][player_id]['team'] = team
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]

    tracks['ball'] = tracker.get_ball_interpolation(tracks['ball'])

    # Add position to tracks
    tracker.add_position_to_tracks(tracks)

    # Ball assigner
    ball_assigner = AssignBallPlayer()
    team_control = []
    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player = ball_assigner.assign_ball(player_track, ball_bbox)

        if assigned_player != -1:
            tracks["players"][frame_num][assigned_player]["has_ball"] = True
            team_control.append(tracks["players"][frame_num][assigned_player]["team"])
        else:
            team_control.append(team_control[-1])

    # Camera movement
    cameraMovement = CameraMovementEstimator(frame=frames[0])
    camera_movements = cameraMovement.get_camera_movement(frames, read_from_stubs=True, stub_path=camera_stub_path)
    new_frames = tracker.get_annotations(frames, tracks, team_control)

    # Adjust position
    cameraMovement._adjust_position_tracks(tracks, camera_movements)

    # Add view transforms
    viewTransform = ViewTransforms()
    viewTransform.add_position_transform_to_track(tracks)

    # Draw
    frames_with_camera = cameraMovement.draw_camera_movement_estimator(new_frames, camera_movements)

    # Distance/speed estimator
    distancSpeed = SpeedDistanceEstimator()
    distancSpeed.add_speed_distance_to_tracks(tracks)
    final_output_video = distancSpeed.draw(frames_with_camera, tracks)

    # Save the video
    save_video(output_video_path, final_output_video)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a video with player tracking and camera movement.")
    parser.add_argument("video_path", type=str, help="Path to the input video file.")
    parser.add_argument("output_video_path", type=str, help="Path to the output video file.")

    args = parser.parse_args()
    main(args.video_path, args.output_video_path)