# Talos v1 — Kaggle Training Plan

## Goal
Train Talos v1 (Rocket League 1v1 RL bot) to **200B steps** using **4 Kaggle Free accounts cycling**, with **GitHub** for code distribution and **checkpoint repo** for state transfer between accounts.

## Timeline
- ~4-5 months to 200B steps
- 4× Kaggle Free accounts, 30h/week GPU quota each → 120h/week total
- Each session: ~9h (Kaggle limit)

## Status (June 4, 2026)

### Done
- **Source code**: All changes implemented and **compile clean** (0 errors, 0 warnings):
  - PhaseManager: 4-phase gamma/entropy, 512×6 MLP, epochs=15, LR=1e-4, miniBatch=10K
  - main.cpp: CLI args, SIGTERM handler, Necto reward wiring
  - TalosRewards.cpp: GoalSpeedBonus (goals only), GoalDistBonus (fixed scoring logic)
  - TalosOBS.cpp: +3 scoreboard features (goal_diff, time_left, overtime)
  - TalosStateSetter.cpp: Binary replay loader with height-weighted sampling, 70/30 split
- **Binary tested**: Runs locally, prints help, all CLI args functional
- **GitHub source repo**: `vfxjamer/Talos-V1` (private) — latest commit `c57d286`
- **Collision meshes**: Tracked in git (16 .cmf files, ~175KB) — removed from .gitignore
- **Kaggle notebook**: `Kaggle/run_talos.ipynb` rewritten with:
  - GitHub clone → build (cmake/g++ on Kaggle) → pull checkpoints → train → push checkpoints
- **PAT**: Updated to classic PAT (`ghp_...`) with full `repo` scope — git push works
- **Checkpoint repo**: `vfxjamer/talos-checkpoints` created (public, empty)

### Remaining
- **Replay files**: No .replay files found on this machine. User needs to provide them or download from ballchasing.com
- **Kaggle Secrets**: User must set `GITHUB_PAT` as a Kaggle Secret on the notebook
- **First Kaggle session**: Test full cycle: clone → build → train → push checkpoints

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
88 total features: 85 standard RocketSim + 3 scoreboard (goal_diff clamped [-1,1], time_left [0,1], is_overtime)

### State Initialization
- 70% replay frames (binary, height-weighted aerial bias)
- 30% procedural modes (kickoff, ground, goalie, aerial, wall, dribble)

## Session Workflow
```
1. Clone Talos-V1 from GitHub (PAT from Kaggle Secret)
2. Install cmake/g++, download LibTorch, build binary
3. Pull latest checkpoint from vfxjamer/talos-checkpoints
4. Run: ./Talos --device cuda --games 20 --resume checkpoints --save-dir checkpoints
5. Every checkpoint save → git push to talos-checkpoints
6. Session end → final push
```

## Account Rotation
- 4 × Kaggle Free accounts cycling
- Each has 30h/week GPU quota
- Checkpoints shared via `vfxjamer/talos-checkpoints` repo

## Key Repositories
| Repo | Purpose |
|---|---|
| `vfxjamer/Talos-V1` | Source code + build |
| `vfxjamer/talos-checkpoints` | Training checkpoints (shared across accounts) |

## Key Files
| File | Purpose |
|---|---|
| `src/PhaseManager.cpp` | 4-phase gamma/entropy config |
| `src/main.cpp` | CLI args, SIGTERM, Necto reward wiring |
| `src/TalosRewards.cpp` | GoalSpeedBonus (goals only), GoalDistBonus (fixed) |
| `src/TalosOBS.cpp` | +3 scoreboard features |
| `src/TalosStateSetter.cpp` | Binary replay loader, 70/30 split |
| `Kaggle/run_talos.ipynb` | Kaggle runner (clone → build → train → push) |
| `collision_meshes/soccar/*.cmf` | RocketSim collision meshes (tracked in git) |

## Credentials (Kaggle/API/ — gitignored)
- `GITHUB-API.txt` → classic PAT with repo scope
- `Kaggle acc-1.txt` → Kaggle API token
