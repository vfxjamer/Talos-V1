"""
Talos v1 Replay Parsing & Binary Serialization
================================================
1. Scans a directory for *.replay files
2. Parses each with carball to extract frame-by-frame physics
3. Serializes to flat binary format for the C++ TalosStateSetter

Usage:
    pip install carball pandas scipy numpy
    python parse_replays.py --replay-dir ./replays --output serialized_replays.bin --max-frames 500000

Binary format (little-endian):
    [int64 n_frames]
    [for each frame]:
      [float*3 ball_pos] [float*3 ball_vel] [float*3 ball_ang_vel]
      [float*3 car1_pos] [float*3 car1_rot (yaw,pitch,roll)] [float*3 car1_vel] [float*3 car1_ang_vel]
      [float car1_boost] [uint8 car1_on_ground]
      [float*3 car2_pos] [float*3 car2_rot (yaw,pitch,roll)] [float*3 car2_vel] [float*3 car2_ang_vel]
      [float car2_boost] [uint8 car2_on_ground]
      [int32 blue_score] [int32 orange_score]
    Total: 114 bytes per frame (1v1)
"""

import struct
import argparse
import os
import glob
import numpy as np
from scipy.spatial.transform import Rotation as R


def euler_from_quat(w, x, y, z):
    """Convert quaternion (w,x,y,z) to Euler (yaw, pitch, roll) in radians."""
    r = R.from_quat([x, y, z, w])
    return r.as_euler('YXZ', degrees=False)  # yaw, pitch, roll


def process_replay(replay_path, max_frames=None, skip_frames=1):
    """
    Parse a single .replay file with carball and yield frame data.

    Args:
        replay_path: Path to .replay file
        max_frames: Maximum frames to extract (None = all)
        skip_frames: Keep every Nth frame (default: 1 = all)

    Yields:
        tuple of (ball_numpy_data, car_data_list, scores)
    """
    import carball

    try:
        game = carball.DecompileReplay(replay_path)
        proto = game.get_protobuf_data()
    except Exception as e:
        print(f"  ERROR: {e}")
        return

    # We need 1v1 replays or at least extract the first two cars
    players = list(proto.players)
    if len(players) < 2:
        print(f"  Skipping: only {len(players)} players found")
        return

    # Map player IDs to indices per team
    # RLGym convention: BLUE (team 0), ORANGE (team 1)
    team_map = {}
    for p in players:
        if p.is_orange:
            team_map[p.id.id] = 1
        else:
            team_map[p.id.id] = 0

    # Extract frame data from the game's tick data
    ticks = proto.game_metadata.tick_metadata if hasattr(proto, 'game_metadata') else None
    if ticks is not None and hasattr(proto, 'tick_data'):
        frames = []
        tick_data = proto.tick_data

        for i in range(0, len(tick_data.frame_data), skip_frames):
            if max_frames is not None and len(frames) >= max_frames:
                break

            frame = tick_data.frame_data[i]

            # Ball state
            ball = frame.ball
            ball_pos = np.array([ball.physics.x, ball.physics.y, ball.physics.z], dtype=np.float32)
            ball_vel = np.array([ball.physics.vx, ball.physics.vy, ball.physics.vz], dtype=np.float32)
            ball_ang_vel = np.array([ball.physics.ang_vx, ball.physics.vy, ball.physics.ang_vz], dtype=np.float32)

            # Car states - extract first player per team
            cars = {}
            for player_frame in frame.player_frames:
                pid = player_frame.player_id.id
                team = team_map.get(pid, 0)
                if team not in cars:
                    cars[team] = player_frame

            if 0 not in cars or 1 not in cars:
                continue  # need both teams

            car_data = {}
            for team_idx, player_frame in cars.items():
                pf = player_frame.physics
                pos = np.array([pf.x, pf.y, pf.z], dtype=np.float32)
                vel = np.array([pf.vx, pf.vy, pf.vz], dtype=np.float32)
                ang_vel = np.array([pf.ang_vx, pf.ang_vy, pf.ang_vz], dtype=np.float32)
                yaw, pitch, roll = euler_from_quat(
                    pf.rotation.w, pf.rotation.x, pf.rotation.y, pf.rotation.z
                )
                rot = np.array([yaw, pitch, roll], dtype=np.float32)
                boost = float(player_frame.boost) if hasattr(player_frame, 'boost') else 100.0
                on_ground = 1 if (hasattr(player_frame, 'on_ground') and player_frame.on_ground) else 0

                car_data[team_idx] = (pos, rot, vel, ang_vel, boost, on_ground)

            # Scores
            blue_score = int(frame.teams[0].score) if len(frame.teams) > 0 else 0
            orange_score = int(frame.teams[1].score) if len(frame.teams) > 1 else 0

            frames.append((ball_pos, ball_vel, ball_ang_vel,
                           car_data[0], car_data[1],
                           blue_score, orange_score))

        print(f"  Extracted {len(frames)} frames")
        yield frames
    else:
        # Alternate extraction method using advanced stats
        print(f"  Game metadata not found, trying alternate extraction...")
        # Fallback: try using carball's dataframe extraction
        try:
            df = game.get_data_frame()
            if df is None or len(df) == 0:
                print(f"  No data frame available")
                return

            frames = []
            for i in range(0, len(df), skip_frames):
                if max_frames is not None and len(frames) >= max_frames:
                    break
                row = df.iloc[i]

                # Map column names (ball_x, ball_y, ball_z, etc.)
                prefix_map = {
                    0: ('blue_', 'blue_'),  # team 0
                    1: ('orange_', 'orange_'),  # team 1
                }

                # Extract car data for each team
                car_data = {}
                for team_idx, (prefix, _) in prefix_map.items():
                    cols = {c: row[c] for c in df.columns if c.startswith(prefix)}

                    pos = np.array([cols.get(f'{prefix}pos_x', 0),
                                     cols.get(f'{prefix}pos_y', 0),
                                     cols.get(f'{prefix}pos_z', 0)], dtype=np.float32)
                    vel = np.array([cols.get(f'{prefix}vel_x', 0),
                                     cols.get(f'{prefix}vel_y', 0),
                                     cols.get(f'{prefix}vel_z', 0)], dtype=np.float32)
                    ang_vel = np.array([cols.get(f'{prefix}ang_vel_x', 0),
                                         cols.get(f'{prefix}ang_vel_y', 0),
                                         cols.get(f'{prefix}ang_vel_z', 0)], dtype=np.float32)

                    # Quaternion → Euler
                    qw = cols.get(f'{prefix}rot_w', 1)
                    qx = cols.get(f'{prefix}rot_x', 0)
                    qy = cols.get(f'{prefix}rot_y', 0)
                    qz = cols.get(f'{prefix}rot_z', 0)
                    yaw, pitch, roll = euler_from_quat(qw, qx, qy, qz)
                    rot = np.array([yaw, pitch, roll], dtype=np.float32)

                    boost = float(cols.get(f'{prefix}boost', 100))
                    on_ground = 1 if cols.get(f'{prefix}on_ground', True) else 0

                    car_data[team_idx] = (pos, rot, vel, ang_vel, boost, on_ground)

                # Ball data
                ball_pos = np.array([row.get('ball_pos_x', 0),
                                      row.get('ball_pos_y', 0),
                                      row.get('ball_pos_z', 0)], dtype=np.float32)
                ball_vel = np.array([row.get('ball_vel_x', 0),
                                      row.get('ball_vel_y', 0),
                                      row.get('ball_vel_z', 0)], dtype=np.float32)
                ball_ang_vel = np.array([row.get('ball_ang_vel_x', 0),
                                          row.get('ball_ang_vel_y', 0),
                                          row.get('ball_ang_vel_z', 0)], dtype=np.float32)

                # Scores (might need synthesis from goal events)
                blue_score = int(row.get('blue_score', 0))
                orange_score = int(row.get('orange_score', 0))

                frames.append((ball_pos, ball_vel, ball_ang_vel,
                               car_data.get(0), car_data.get(1),
                               blue_score, orange_score))

            if frames:
                print(f"  Extracted {len(frames)} frames (alternate method)")
                yield frames
            else:
                print(f"  No frames extracted")
        except Exception as e2:
            print(f"  Alternate extraction also failed: {e2}")


def serialize_frames(all_frames, output_path):
    """Serialize frames to binary format matching C++ reader."""
    total = len(all_frames)
    print(f"Serializing {total} frames to {output_path}...")

    with open(output_path, 'wb') as f:
        # Header: number of frames
        f.write(struct.pack('<q', total))

        for (ball_pos, ball_vel, ball_ang_vel,
             car0, car1,
             blue_score, orange_score) in all_frames:

            # Ball
            f.write(ball_pos.tobytes())  # 12 bytes
            f.write(ball_vel.tobytes())  # 12 bytes
            f.write(ball_ang_vel.tobytes())  # 12 bytes

            # Car 0 (blue team)
            for car_frame in [car0, car1]:
                pos, rot, vel, ang_vel, boost, on_ground = car_frame
                f.write(pos.tobytes())  # 12 bytes
                f.write(rot.tobytes())  # 12 bytes
                f.write(vel.tobytes())  # 12 bytes
                f.write(ang_vel.tobytes())  # 12 bytes
                f.write(struct.pack('<f', boost))  # 4 bytes
                f.write(struct.pack('<B', on_ground))  # 1 byte

            # Scores
            f.write(struct.pack('<ii', blue_score, orange_score))  # 8 bytes

    file_size = os.path.getsize(output_path)
    expected = 8 + total * 114
    print(f"  Written {file_size} bytes (expected ~{expected})")


def main():
    parser = argparse.ArgumentParser(description='Parse Rocket League replays into binary format for Talos v1')
    parser.add_argument('--replay-dir', required=True, help='Directory containing *.replay files')
    parser.add_argument('--output', default='serialized_replays.bin', help='Output binary file path')
    parser.add_argument('--max-frames', type=int, default=500000, help='Maximum total frames to extract')
    parser.add_argument('--skip-frames', type=int, default=2, help='Keep every Nth frame (higher = less data)')
    args = parser.parse_args()

    replay_files = sorted(glob.glob(os.path.join(args.replay_dir, '*.replay')))
    if not replay_files:
        print(f"No .replay files found in {args.replay_dir}")
        return

    print(f"Found {len(replay_files)} replay files")

    all_frames = []
    total_frames = 0
    frame_limit = args.max_frames

    for replay_path in replay_files:
        if total_frames >= frame_limit:
            print(f"Reached frame limit ({frame_limit}), stopping")
            break

        print(f"\nProcessing: {replay_path}")
        for frames in process_replay(replay_path, max_frames=frame_limit - total_frames, skip_frames=args.skip_frames):
            all_frames.extend(frames)
            total_frames = len(all_frames)
            print(f"  Total so far: {total_frames}")

    if all_frames:
        serialize_frames(all_frames, args.output)
        print(f"\nDone! {total_frames} frames serialized to {args.output}")
    else:
        print("\nNo frames extracted!")


if __name__ == '__main__':
    main()
