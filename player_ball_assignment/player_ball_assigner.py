import sys
sys.path.append('../')
from utils import get_center_bbox, meassure_distance

class PlayerBallAssigner():
    def __init__(self) -> None:
        self.distance_threshold = 70

    def assign_ball_to_player(self, players, ball_bbox):
        # get ball position
        ball_postion = get_center_bbox(ball_bbox)

        minimum_distance = self.distance_threshold
        assigned_player = None

        # get player foot positons
        for player_id, player in players.items():
            player_bbox = player['bbox']

            # distance of ball from each foot
            distance_left = meassure_distance(ball_postion, [player_bbox[0], player_bbox[-1]])
            distance_right = meassure_distance(ball_postion, [player_bbox[2], player_bbox[-1]])

            distance = min(distance_right, distance_left)
            
            if distance < minimum_distance:
                minimum_distance = distance
                assigned_player = player_id

        return assigned_player