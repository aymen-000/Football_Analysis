from sklearn.cluster import KMeans
import numpy as np

class TeamAssigner:
    def __init__(self):
        self.team_colors = {}
        self.player_team_dict = {}
        self.kmeans = None

    def get_cluster_model(self, img):
        image_2d = img.reshape(-1, 3)
        
        kmean = KMeans(n_clusters=2, init="k-means++", n_init=10)
        kmean.fit(image_2d)
        
        return kmean
    
    def get_player_color(self, frame, bbox):
        image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
        
        top_half_image = image[:int(image.shape[0]/2), :]
        
        # get cluster 
        kmean = self.get_cluster_model(top_half_image)
        
        labels = kmean.labels_
        # Reshape the labels to the image shape
        clustered_image = labels.reshape(top_half_image.shape[:2])
        
        # Get the player cluster
        corner_clusters = [clustered_image[0,0], clustered_image[0,-1], clustered_image[-1,0], clustered_image[-1,-1]]
        non_player_cluster = max(set(corner_clusters), key=corner_clusters.count)
        player_cluster = 1 - non_player_cluster

        player_color = kmean.cluster_centers_[player_cluster]

        return player_color
    
    def assign_team_color(self, frame, players_detections):
        players_colors = []
        
        for _, player_detection in players_detections.items():
            bbox = player_detection["bbox"]
            player_color = self.get_player_color(frame, bbox)
            players_colors.append(player_color)
            
        # kmean to get just two colors 
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=10)
        kmeans.fit(players_colors)
        
        self.kmeans = kmeans

        self.team_colors[1] = kmeans.cluster_centers_[0]
        self.team_colors[2] = kmeans.cluster_centers_[1]
    
    def get_player_team(self, frame, player_bbox, player_id):
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]
        
        player_color = self.get_player_color(frame, player_bbox)
        
        team_id = self.kmeans.predict(player_color.reshape(1, -1))[0]
        team_id += 1
        if player_id == 91:
            team_id = 1
            
        self.player_team_dict[player_id] = team_id
        
        return team_id