# Talos v1 — Kaggle Training Plan

## Goal
Train Talos v1 (Rocket League 1v1 RL bot) to **200B steps** using **4 Kaggle Free accounts cycling**, with **GitHub** for code distribution and **checkpoint repo** for state transfer between accounts.

## Timeline
- ~4-5 months to 200B steps
- 4× Kaggle Free accounts, 30h/week GPU quota each → 120h/week total
- Each session: ~9h (Kaggle limit)

## Status (June 4, 2026)
### Done
- All source changes implemented and **compiled clean** (0 errors, 0 warnings):
  - **PhaseManager**: 4-phase gamma/entropy schedule, 512×6 network, epochs=15, LR=1e-4, miniBatch=10K, same Necto rewards across all phases
  - **main.cpp**: CLI args (--resume, --device, --games, --save-dir, --phase, --replays), SIGTERM handler (Save+exit), correct Necto reward wiring (no vel_ball_to_goal)
  - **TalosRewards.cpp**: GoalSpeedBonusReward fires only on goals; GoalDistBonusReward inverted-scoring bug fixed (uses `RS_TEAM_FROM_Y`)
  - **TalosOBS**: +3 scoreboard features (goal_diff, time_left, overtime), Reset() override
  - **TalosStateSetter**: Binary replay loader with height-weighted sampling, 70% replay / 30% procedural
- Created `Kaggle/` with: TRAINING_PLAN.md, run_talos.ipynb, package_dataset.sh, parse_replays.py
- Local git repo initialized at `Talos v1/`
- GitHub repo **vfxjamer/Talos-V1** exists (private)
- Collision meshes symlink fixed (copied from render build output)

### Pending
- **GitHub push blocked**: fine-grained PAT (`github_pat_...`) doesn't have write access to the repo — needs a classic PAT with `repo` scope, or an updated fine-grained PAT with read/write permissions
- **Notebook rewrite**: `Kaggle/run_talos.ipynb` still uses old Kaggle Dataset + Google Drive approach — needs to be rewritten for GitHub clone + checkpoint repo model
- **Checkpoint repo**: needs to be created (e.g., `vfxjamer/Talos-Checkpoints` or branch on Talos-V1)
- **Kaggle notebook deployment**: test full cycle on Kaggle Free

## Architecture

### Network
- Shared MLP: 2×512 → Policy 6×512 / Critic 6×512
- ≈3.46M params

### 4-Phase Schedule
| Phase | Steps | Gamma | Entropy | LR | Epochs | MiniBatch |
|---|---|---|---|---|---|---|
| 1 | 0–30B | 0.990 | 0.05 | 1e-4 | 15 | 10,000 |
| 2 | 30–80B | 0.993 | 0.03 | 1e-4 | 15 | 10,000 |
| 3 | 80–150B | 0.995 | 0.02 | 1e-4 | 15 | 10,000 |
| 4 | 150–200B | 0.998 | 0.01 | 1e-4 | 15 | 10,000 |

### Rewards (Necto-derived, same across all phases)
| Reward | Weight | Notes |
|---|---|---|
| goal_w | 10.0 | Sparse goal score |
| win_prob_w | 10.0 | Skellam win prob delta |
| goal_dist_w | 10.0 | Exponential distance delta |
| goal_speed_bonus_w | 2.5 | Only on goal events |
| goal_dist_bonus_w | 2.5 | Goal event reward/penalty |
| demo_w | 8.0 | Split 50/50 demo/reward penalty |
| boost_gain_w | 1.5 | sqrt-boost gain |
| boost_lose_w | 0.8 | sqrt-boost loss (below GOAL_HEIGHT) |
| touch_height_w | 3.0 | Height-scaled touch + wall proximity |
| touch_accel_w | 0.5 | Ball speed change on touch |
| flip_reset_w | 10.0 | Flip reset mechanic |
| dist_w | 0.25 | Player quality: exp(-dist/1410) |
| align_w | 0.25 | Player quality: alignment |
| ang_vel_w | 0.005 | Tiny spinning reward |
| touch_grass_w | 0.005 | Ground idle penalty |
| opponent_punish_w | **1.0** | Opponent avg subtraction (was 0.5 in Necto) |
| ~~vel_ball_to_goal_w~~ | removed | Not in Necto |
| ~~team_spirit~~ | removed | No-op in 1v1 |

### Observations
88 total features:
- 85 standard RocketSim features
- +3 scoreboard: goal_diff (clamped [-1,1]), time_left [0,1], is_overtime (binary)

### State Initialization
- **70% replay** → binary serialized frames from carball, height-weighted (aerial bias)
- **30% procedural** → kickoff, ground, goalie, aerial, wall, dribble

## Planned Session Workflow (to be implemented)
```
1. Clone Talos-V1 from GitHub (using PAT)
2. Pull latest checkpoint from checkpoints repo
3. Run binary: ./Talos --device cuda --games 20 --resume <ckpt_dir> --save-dir checkpoints
4. Every 5M steps: auto-save → git push checkpoint to checkpoints repo
5. Session end: final save → push to checkpoints repo
```

## Account Rotation
- 4 × Kaggle Free accounts
- Cycle N→N+1 when quota exhausted (30h/week)
- All share same checkpoints via GitHub repo
- By the time Account 4's 30h is spent, Account 1's quota resets

## Key Files
| File | Purpose |
|---|---|
| `src/PhaseManager.cpp` | 4-phase gamma/entropy config |
| `src/main.cpp` | CLI args, SIGTERM, Necto reward wiring |
| `src/TalosRewards.cpp` | GoalSpeedBonus (goals only), GoalDistBonus (fixed) |
| `src/TalosOBS.cpp` | +3 scoreboard features |
| `src/TalosStateSetter.cpp` | Binary replay loader, 70/30 split |
| `Kaggle/run_talos.ipynb` | Kaggle runner (NEEDS REWRITE) |
| `scripts/parse_replays.py` | Carball → binary serialization |
| `Kaggle/package_dataset.sh` | Dataset packager (may be replaced) |

## Credentials (in Kaggle/API/)
- `GITHUB-API.txt` → PAT (needs write scope)
- `Kaggle acc-1.txt` → Kaggle API token
