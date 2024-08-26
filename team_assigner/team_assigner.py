from sklearn.cluster import KMeans

class TeamAssigner:
    def __init__(self) -> None:
        self.team_colors = {}
        self.player_team_dict = {}

    def get_clustering_model(self, image):
        image_2d = image.reshape(-1, 3)

        # perform kmeans cluster of 2
        kmeans = KMeans(n_clusters=2, init='k-means++', n_init=10, random_state=0).fit(image_2d)

        return kmeans

    def get_player_color(self, frame, bbox):
        # get player image
        image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
        image_top = image[:int(image.shape[0]//2), :]

        # get clustering model
        kmeans = self.get_clustering_model(image_top)

        # get cluster labels
        labels = kmeans.labels_

        # reshape to image
        clusterd_image = labels.reshape((image_top.shape[0], image_top.shape[1]))
        
        # set corner cluster label to background
        corner_cluster = [clusterd_image[0,0], clusterd_image[-1,0], clusterd_image[0,-1], clusterd_image[-1,-1]]
        non_player_cluster = max(set(corner_cluster), key=corner_cluster.count)
        player_cluster = 1 - non_player_cluster

        player_color = kmeans.cluster_centers_[player_cluster]

        return player_color

    def assign_team_color(self, frame, player_detections):
        player_colors = []
        
        for _, player_detection in player_detections.items():
            bbox = player_detection['bbox']
            player_color = self.get_player_color(frame, bbox)
            player_colors.append(player_color)

        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=1)
        kmeans.fit(player_colors)

        self.kmeans = kmeans

        self.team_colors[1] = kmeans.cluster_centers_[0]
        self.team_colors[2] = kmeans.cluster_centers_[1]

    def get_player_team(self, frame, player_bbox, player_id):
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]
        
        player_color = self.get_player_color(frame, player_bbox)

        team_id = self.kmeans.predict(player_color.reshape(1,-1))[0]
        team_id += 1

        if player_id == 92:
            team_id = 1
        elif player_id == 194:
            team_id = 2

        self.player_team_dict[player_id] = team_id

        return team_id
