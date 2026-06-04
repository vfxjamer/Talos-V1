"""
Talos v1 Replay Parsing & Binary Serialization
================================================
Parses .replay files using carball (with boxcars backend), serializes to flat binary.

Usage:
    pip install carball
    python parse_replays.py --replay-dir ./replays --output serialized_replays.bin

Binary format (little-endian, 114 bytes per frame):
    [int64 n_frames]
    [for each frame]:
      [float*3 ball_pos] [float*3 ball_vel] [float*3 ball_ang_vel]
      [float*3 car1_pos] [float*3 car1_rot (yaw,pitch,roll)] [float*3 car1_vel] [float*3 car1_ang_vel]
      [float car1_boost] [uint8 car1_on_ground]
      [float*3 car2_pos] [float*3 car2_rot (yaw,pitch,roll)] [float*3 car2_vel] [float*3 car2_ang_vel]
      [float car2_boost] [uint8 car2_on_ground]
      [int32 blue_score] [int32 orange_score]
"""

import struct, argparse, os, glob, json, sys
import numpy as np

def quat_to_euler(w, x, y, z):
    """Quaternion (w,x,y,z) -> (yaw, pitch, roll) in radians."""
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    sinp = 2.0 * (w * y - z * x)
    pitch = np.arcsin(np.clip(sinp, -1.0, 1.0))
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return np.array([yaw, pitch, roll], dtype=np.float32)

def parse_with_boxcars(replay_path, max_frames, skip_frames):
    """Parse using carball.decompile_replay (carball >= 0.7 / boxcars backend)."""
    from carball import decompile_replay
    data = decompile_replay(replay_path)
    frames_data = data.get("frames") or data.get("tickData", {}).get("frames") or data.get("ticks", [])
    if not frames_data:
        return None

    players = data.get("players", [])
    team_map = {}
    for p in players:
        pid = p.get("id", {}).get("id", p.get("id"))
        team_map[pid] = 1 if p.get("isOrange") else 0

    def get_f32(v, idx, default=0.0):
        if isinstance(v, (list, tuple)):
            return float(v[idx]) if idx < len(v) else default
        return float(getattr(v, chr(120+idx), default))

    frames_out = []
    for i in range(0, len(frames_data), skip_frames):
        if max_frames and len(frames_out) >= max_frames:
            break
        f = frames_data[i]
        ball = f.get("ball", {})
        ball_pos = np.array([get_f32(ball.get("pos", {}), 0), get_f32(ball.get("pos", {}), 1), get_f32(ball.get("pos", {}), 2)], dtype=np.float32)
        ball_vel = np.array([get_f32(ball.get("vel", {}), 0), get_f32(ball.get("vel", {}), 1), get_f32(ball.get("vel", {}), 2)], dtype=np.float32)
        ball_ang_vel = np.array([get_f32(ball.get("angVel", {}), 0), get_f32(ball.get("angVel", {}), 1), get_f32(ball.get("angVel", {}), 2)], dtype=np.float32)

        player_frames = f.get("players", [])
        cars = {}
        for pf in player_frames:
            pid = pf.get("id", {}).get("id", pf.get("id"))
            team = team_map.get(pid, 0)
            if team not in cars:
                cars[team] = pf

        if 0 not in cars or 1 not in cars:
            continue

        car_data = {}
        for team_idx in (0, 1):
            pf = cars[team_idx]
            ph = pf.get("physics", pf)
            pos = np.array([get_f32(ph.get("pos", {}), 0), get_f32(ph.get("pos", {}), 1), get_f32(ph.get("pos", {}), 2)], dtype=np.float32)
            vel = np.array([get_f32(ph.get("vel", {}), 0), get_f32(ph.get("vel", {}), 1), get_f32(ph.get("vel", {}), 2)], dtype=np.float32)
            ang_vel = np.array([get_f32(ph.get("angVel", {}), 0), get_f32(ph.get("angVel", {}), 1), get_f32(ph.get("angVel", {}), 2)], dtype=np.float32)
            rot_q = ph.get("rot", ph.get("rotation", {}))
            qw = get_f32(rot_q, 0 if isinstance(rot_q, dict) else 0, 1.0)
            qx = get_f32(rot_q, 1 if isinstance(rot_q, dict) else 0, 0.0)
            qy = get_f32(rot_q, 2 if isinstance(rot_q, dict) else 0, 0.0)
            qz = get_f32(rot_q, 3 if isinstance(rot_q, dict) else 0, 0.0)
            rot = quat_to_euler(qw, qx, qy, qz)
            boost = float(pf.get("boost", 100))
            on_ground = 1 if pf.get("onGround", True) else 0
            car_data[team_idx] = (pos, rot, vel, ang_vel, boost, on_ground)

        teams = f.get("teams", [])
        if len(teams) >= 2:
            bs, os = int(teams[0].get("score", 0)), int(teams[1].get("score", 0))
        else:
            bs = os = 0

        frames_out.append((ball_pos, ball_vel, ball_ang_vel, car_data[0], car_data[1], bs, os))

    return frames_out

def parse_with_carball_old(replay_path, max_frames, skip_frames):
    """Fallback: use carball's protobuf-based DecompileReplay (carball < 0.7)."""
    import carball
    game = carball.DecompileReplay(replay_path)
    proto = game.get_protobuf_data()

    players = list(proto.players)
    if len(players) < 2:
        return None

    team_map = {}
    for p in players:
        team_map[p.id.id] = 1 if p.is_orange else 0

    ticks = getattr(proto, "game_metadata", None)
    if ticks is None or not hasattr(proto, "tick_data"):
        return None

    frames_out = []
    tick_data = proto.tick_data
    for i in range(0, len(tick_data.frame_data), skip_frames):
        if max_frames and len(frames_out) >= max_frames:
            break

        frame = tick_data.frame_data[i]
        ball = frame.ball
        ball_pos = np.array([ball.physics.x, ball.physics.y, ball.physics.z], dtype=np.float32)
        ball_vel = np.array([ball.physics.vx, ball.physics.vy, ball.physics.vz], dtype=np.float32)
        ball_ang_vel = np.array([ball.physics.ang_vx, ball.physics.ang_vy, ball.physics.ang_vz], dtype=np.float32)

        cars = {}
        for pf in frame.player_frames:
            pid = pf.player_id.id
            team = team_map.get(pid, 0)
            if team not in cars:
                cars[team] = pf

        if 0 not in cars or 1 not in cars:
            continue

        car_data = {}
        for team_idx in (0, 1):
            pf = cars[team_idx]
            ph = pf.physics
            pos = np.array([ph.x, ph.y, ph.z], dtype=np.float32)
            vel = np.array([ph.vx, ph.vy, ph.vz], dtype=np.float32)
            ang_vel = np.array([ph.ang_vx, ph.ang_vy, ph.ang_vz], dtype=np.float32)
            yaw, pitch, roll = quat_to_euler(ph.rotation.w, ph.rotation.x, ph.rotation.y, ph.rotation.z)
            rot = np.array([yaw, pitch, roll], dtype=np.float32)
            boost = float(getattr(pf, "boost", 100))
            on_ground = 1 if getattr(pf, "on_ground", True) else 0
            car_data[team_idx] = (pos, rot, vel, ang_vel, boost, on_ground)

        blue_score = int(frame.teams[0].score) if len(frame.teams) > 0 else 0
        orange_score = int(frame.teams[1].score) if len(frame.teams) > 1 else 0

        frames_out.append((ball_pos, ball_vel, ball_ang_vel, car_data[0], car_data[1], blue_score, orange_score))

    return frames_out

def process_replay(replay_path, max_frames=None, skip_frames=1):
    frames = parse_with_boxcars(replay_path, max_frames, skip_frames)
    if frames:
        yield frames
        return

    frames = parse_with_carball_old(replay_path, max_frames, skip_frames)
    if frames:
        yield frames
        return

    print(f"  No compatible carball API found")

def serialize_frames(all_frames, output_path):
    total = len(all_frames)
    print(f"Serializing {total} frames to {output_path}...")
    with open(output_path, "wb") as f:
        f.write(struct.pack("<q", total))
        for (ball_pos, ball_vel, ball_ang_vel, car0, car1, bs, os) in all_frames:
            f.write(ball_pos.tobytes())
            f.write(ball_vel.tobytes())
            f.write(ball_ang_vel.tobytes())
            for car_frame in (car0, car1):
                pos, rot, vel, ang_vel, boost, on_ground = car_frame
                f.write(pos.tobytes())
                f.write(rot.tobytes())
                f.write(vel.tobytes())
                f.write(ang_vel.tobytes())
                f.write(struct.pack("<f", boost))
                f.write(struct.pack("<B", on_ground))
            f.write(struct.pack("<ii", bs, os))
    size = os.path.getsize(output_path)
    expected = 8 + total * 114
    print(f"  Written {size} bytes (expected ~{expected})")

def main():
    parser = argparse.ArgumentParser(description="Parse RL replays into binary for Talos v1")
    parser.add_argument("--replay-dir", required=True, help="Directory with .replay files")
    parser.add_argument("--output", default="serialized_replays.bin", help="Output binary path")
    parser.add_argument("--max-frames", type=int, default=500000, help="Max total frames")
    parser.add_argument("--skip-frames", type=int, default=2, help="Keep every Nth frame")
    args = parser.parse_args()

    replay_files = sorted(glob.glob(os.path.join(args.replay_dir, "*.replay")))
    if not replay_files:
        print(f"No .replay files found in {args.replay_dir}")
        return

    print(f"Found {len(replay_files)} replay files")
    all_frames, total_frames = [], 0

    for replay_path in replay_files:
        if total_frames >= args.max_frames:
            print(f"Reached frame limit ({args.max_frames}), stopping")
            break
        print(f"\nProcessing: {os.path.basename(replay_path)}")
        for frames in process_replay(replay_path, max_frames=args.max_frames - total_frames, skip_frames=args.skip_frames):
            all_frames.extend(frames)
            total_frames = len(all_frames)
            print(f"  Frames: {len(frames)} (total: {total_frames})")

    if all_frames:
        serialize_frames(all_frames, args.output)
        print(f"\nDone! {total_frames} frames -> {args.output}")
    else:
        print("\nNo frames extracted!")

if __name__ == "__main__":
    main()
