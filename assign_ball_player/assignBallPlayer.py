import sys
sys.path.append('../')
from utils import get_center, get_distance

class AssignBallPlayer:
    def __init__(self):
        self.player_max_distance = 70
        
    def assign_ball(self, players, ball_bbox):
        ball_pos = get_center(ball_bbox)
        min_distance = 9999
        player = -1
        
        for player_id, player_track in players.items():
            player_bbox = player_track['bbox']
            distance_left = get_distance(ball_pos, (player_bbox[0], player_bbox[3]))
            distance_right = get_distance(ball_pos, (player_bbox[2], player_bbox[3]))
            distance = min(distance_left, distance_right)
            
            if int(distance) <=  int(self.player_max_distance):
                if int(distance) < int(min_distance):
                    min_distance = distance
                    player = player_id 
        
        return player