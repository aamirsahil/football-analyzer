from ultralytics import YOLO
from utils import read_video, save_video
from tracker import Tracker
import numpy as np
import cv2
from team_assigner import TeamAssigner
from player_ball_assignment import PlayerBallAssigner

def main():
    # read video
    input_path = 'input_video/08fd33_4.mp4'
    video_frames = read_video(input_path)

    # initialize tracker
    model_path = 'models/best.pt'
    tracker = Tracker(model_path)
    tracks = tracker.get_object_tracks(video_frames, read_from_stub=True, stub_path='stubs/track_stub.pkl')

    # interpolate ball
    tracks['ball'] = tracker.interpolate_ball(tracks['ball'])

    # assign team
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], tracks['players'][0])

    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num], track['bbox'], player_id)
            tracks['players'][frame_num][player_id]['team'] = team
            tracks['players'][frame_num][player_id]['team_color'] = [0, 0, 255] if team==1 else [255, 0, 0]

    # Assign ball aquisition
    player_assigner = PlayerBallAssigner()
    team_ball_control = []

    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)

        if assigned_player is not None:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
        else:
            team_ball_control.append(team_ball_control[-1])
    team_ball_control = np.array(team_ball_control)



    # Draw output
    output_video_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control)

    # save video
    output_path = 'output_video/output_video.avi'
    save_video(output_video_frames, output_path)

if __name__ == "__main__":
    main()