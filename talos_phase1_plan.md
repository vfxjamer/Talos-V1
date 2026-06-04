# Talos — Phase 1 Training Design Document

> **Last updated:** Phase 1 final
> **Status:** Implementation ready

---

## 1. Project Overview

**Talos** is a self-play Rocket League bot trained with deep reinforcement learning using GigaLearnCPP and PPO. Phase 1 targets a "Nexto-lite" skill level by 30B steps: consistent scoring, basic defense (stays goal-side), and ball-on-nose dribbling.

The project builds on lessons from Talos v1 (network architecture baseline) and previous reward experiments, extending both with new reward designs and a structured multi-phase curriculum.

---

## 2. Network Architecture

### Structure

```
Input (~91 obs)
    → Shared Trunk: Linear(256) → Linear(256)
              ↙                          ↘
    Policy Head                       Critic Head
    Linear(256) × 4                  Linear(256) × 4
    → Action logits (90)             → Value scalar
```

### Depth Rationale

| Consideration | Detail |
|---|---|
| Total layers | 6 (shared ×2 + head ×4) |
| Pros | Capacity for complex policies; headroom across all 4 phases; Talos v1 proved convergence at this depth |
| Cons | Slower inference per step (caps steps/sec on CPU); early training will look flat longer than a shallower net |
| Decision | Kept intentionally — 4-phase curriculum requires capacity from the start |

**Fallback:** If Phase 1 plateaus for >5B steps with no measurable improvement, trim to
`shared 256×2 → policy 256×2` and restart from last good checkpoint.

---

## 3. Observation Space

| Feature | Dims | Normalization |
|---|---|---|
| Ball pos / vel / angVel | 9 | pos: ÷5000, vel: ÷2300, angVel: ÷3 |
| Previous action | 8 | Raw one-hot |
| Boost pad states | 34 | Active=1, cooldown=1/(1+timer) |
| Player pos / forward / up / vel / angVel | 15 | Same coefficients |
| Local angVel (body frame) | 3 | ÷3 |
| Local ball pos / vel (body frame) | 6 | pos: ÷5000, vel: ÷2300 |
| Player boost / onGround / hasFlip / isDemoed | 4 | boost÷100, rest bool |
| Opponent pos / vel / angVel / boost / onGround / hasFlip | 12 | Same coefficients |
| **Total** | **~91** | |

---

## 4. Action Space

**RLGym DefaultAction — 90 discrete actions**

Combinatorial space over: `throttle / steer / pitch / yaw / roll / jump / boost / handbrake`

No continuous action space. Discrete actions reduce training complexity in Phase 1 and are sufficient through Phase 2. Re-evaluate for Phase 3+ if a ceiling is hit.

---

## 5. Four-Phase Roadmap (High-Level)

| Phase | Name | Step Budget | Goal |
|---|---|---|---|
| 1 | Nexto-lite | 0–30B | Score consistently, basic defense, ball on nose |
| 2 | Scoring & Awareness | 30–80B | Win probability awareness, aerials, kickoff wavedash |
| 3 | Mechanical Depth | 80–150B | Flip resets, demos, chain wall wavedashes, aggression |
| 4 | Refinement | 150–200B | Strip crutches, pure win-rate optimization |

> Phases 2–4 are high-level placeholders. Each will be fully detailed before implementation begins.

---

## 6. Phase 1 — Nexto-lite (0–30B steps)

**Goal:** By 30B steps the bot can score consistently, apply basic defense (stays goal-side), and dribble the ball on its nose.

### 6.1 Hyperparameters

| Parameter | Value |
|---|---|
| Discount (γ) | 0.99 |
| Clip (ε) | 0.04 |
| Learning rate | 3e-4 |
| Batch size | 50,000 |
| Mini-batch size | 10,000 |
| Epochs per update | 2 |
| GAE lambda | 0.95 |
| Reward clip range | 10 |

### 6.2 Training Setup

| Setting | Value |
|---|---|
| Parallel games | 36 |
| Hardware | CPU |
| Tick skip | 8 |
| Action delay | 7 |
| Step budget | 30B (hard cap) |
| State resets | 50% kickoff, 50% random |
| Opponent | Self-play (same policy) |

### 6.3 Reward Table

| Reward | Weight | Class | Source |
|---|---|---|---|
| GoalReward | 10.0 | `PlayerGoalReward` | CommonRewards |
| GoalDistanceReward | 5.0 | `GoalDistanceReward` | TalosRewards |
| DefensiveReward | 4.0 | `DefensiveReward` | TalosRewards |
| GoalSpeedBonusReward | 2.0 | `GoalSpeedBonusReward` | TalosRewards |
| TouchReward | 1.5 | `TouchBallReward` | CommonRewards |
| PickupBoostReward | 1.5 | `PickupBoostReward` | CommonRewards |
| DribbleReward | 1.0 | `DribbleReward` | TalosRewards (gated) |
| VelocityBallToGoal | 1.0 | `VelocityBallToGoalReward` | CommonRewards |
| VelocityPlayerToBall | 0.6 | `VelocityPlayerToBallReward` | CommonRewards |
| BoostWastePenalty | 0.5 | `BoostWastePenalty` | TalosRewards (negative) |
| GroundIdlePenalty | 0.15 | `GroundIdlePenalty` | TalosRewards (negative) |

#### Removed from Earlier Draft

| Reward | Reason |
|---|---|
| WinProbReward | Too complex for Phase 1; requires game-state comprehension the bot doesn't have yet. Moved to Phase 2 |
| BallProximityReward | Farmable — see §6.5 for full worked example |
| WavedashReward | Dead weight at 0.2 weight; gradient too small to reinforce anything. Phase 3+ concern |
| GoalAlignmentReward | Replaced by DefensiveReward (more targeted) |

### 6.4 DribbleReward — Three-Gate Implementation

Raw DribbleReward (ball near car) can be farmed by driving laps once the ball is balanced on the roof. Three conditions must be satisfied simultaneously:

```cpp
// Per-player state tracked by carId to avoid cross-player contamination
struct PerPlayer {
    int stepsSinceTouch = 999;
    int dribbleConsecutiveTicks = 0;
};

// Condition 1: Ball is physically on the car
//   Geometry check — ball above roof threshold in car-local space
//   Velocity-delta check — ball and car are moving together
bool ballAboveRoof = relBallPos.z > ROOF_THRESHOLD;  // ~80 UU
bool velMatched = |ball.vel - car.vel| < VEL_DELTA_MAX;  // ~200 UU/s
bool ballOnCar = ballAboveRoof && velMatched;

// Condition 2: Bot touched the ball recently (< 20 steps ~1.3s)
bool touchGate = stepsSinceTouch < TOUCH_GATE_TICKS;

// Condition 3: Dribble has been sustained (>= 8 ticks ~0.5s)
bool consecutiveGate = dribbleConsecutiveTicks >= CONSECUTIVE_MIN_TICKS;

// Final gate — all three required
bool dribbleActive = ballOnCar && touchGate && consecutiveGate;
float reward = dribbleActive ? 1.0f : 0.0f;
```

**Why the velocity-delta check is the key insight:**
`relBallPos.z > ROOF_THRESHOLD` alone passes any time the ball lands on the roof by chance.
`|ball.vel - car.vel| < VEL_DELTA_MAX` is what separates *controlling the ball* from *the ball happened to land there*. A bot cannot fake velocity matching without actively carrying the ball.

**Parameter guidance:**

| Constant | Value | Notes |
|---|---|---|
| `ROOF_THRESHOLD` | 80 UU | Car roof height in Unreal units |
| `VEL_DELTA_MAX` | 200 UU/s | Tight enough to filter one-tick bounces |
| `TOUCH_GATE_TICKS` | 20 | ~1.3s at 15 ticks/sec |
| `CONSECUTIVE_MIN_TICKS` | 8 | ~0.5s; filters accidental single-frame triggers |

---

### 6.5 Why BallProximityReward Was Removed — The Farming Trap

BallProximityReward is a **continuous dense reward** paid every step. GoalReward is **sparse**. The math works against you even at low weights.

**Worked example — typical 250-step episode:**

```
GoalReward (sparse):
    1 goal × 10.0                          =  10.0 total

BallProximityReward (continuous, weight 0.8):
    250 steps × 0.8 × 0.70 avg closeness  = 140.0 total
```

The bot earns **14× more reward** from hovering near the ball than from scoring.
The optimizer finds this local optimum early and never leaves it.

**Observed failure modes once the bot discovers this:**

- **Orbiting** — circling the ball without committing to a hit
- **Corner pinning** — trapping the ball in a corner and sitting on it
- **Shot avoidance** — hard shots send the ball far away, *reducing* proximity reward, so the bot learns to tap softly or not at all
- **Defensive camping** — ball near own net satisfies proximity *and* DefensiveReward simultaneously, double-farming both

**Why the replacements don't have the same problem:**

| Removed | Replaced with | Why it can't be farmed |
|---|---|---|
| BallProximityReward | VelocityPlayerToBall | `dot(player_vel, normalize(ball_pos - player_pos))` — drops to zero once you stop *moving toward* the ball. Standing still next to the ball pays nothing |
| BallProximityReward | VelocityBallToGoal | `dot(ball_vel, normalize(opponentGoal - ball_pos))` — only pays when the ball is actively moving toward the opponent net. Requires actually hitting it |
| BallProximityReward | TouchReward | Event-based (1.5 per contact frame), not continuous. Can't be collected by proximity alone |

---

### 6.6 DefensiveReward — How It Works

```cpp
Vec ownGoal = (player.team == BLUE) ? BLUE_GOAL_BACK : ORANGE_GOAL_BACK;
Vec carToGoal = (ownGoal - player.pos).Normalized();
Vec ballToGoal = (ownGoal - state.ball.pos).Normalized();

float alignment = carToGoal.Dot(ballToGoal);
// Both direction vectors point toward same goal → car is on correct side

float carDistToGoal = (player.pos - ownGoal).Length();
float ballDistToGoal = (state.ball.pos - ownGoal).Length();
// Penalize if car is further from own goal than the ball (out of position)
float distFactor = (carDistToGoal <= ballDistToGoal) ? 1.0 : ballDistToGoal / carDistToGoal;

return max(0, alignment * distFactor);
```

The reward is highest when the car is between the ball and its own goal, facing the right direction. It drops to zero if the car is on the wrong side of the ball.

---

### 6.7 Phase 1 Exit Criteria

**Hard cap:** 30B steps regardless of performance.

**Early graduation trigger** — Phase 2 unlocks when *any one* of the following is met:

| Metric | Threshold | Eval window |
|---|---|---|
| Win rate vs Atba baseline | > 55% | 1,000 eval games |
| Average goals / game | > 1.2 | 1,000 eval games |
| Dribble sequences > 3 consecutive seconds | ≥ 20% of games | 1,000 eval games |

---

## 7. Checkpoint Strategy

| Setting | Value |
|---|---|
| Save frequency | Every 1M steps |
| Checkpoints kept | Last 8 (rolling) |
| Metrics tracked | Avg reward / goals per episode / touch rate |

**"Good checkpoint" definition:** Manual promotion. Any checkpoint that crosses the 55% win rate threshold vs Atba is tagged and retained indefinitely, outside the rolling 8-checkpoint window.

---

## 8. Implementation Files

| File | Purpose |
|---|---|
| `src/main.cpp` | Entry point, env creation, Learner setup |
| `src/TalosOBS.h/.cpp` | Observation builder (~91 dims) |
| `src/TalosRewards.h/.cpp` | Custom rewards: GoalDistance, GoalSpeedBonus, BoostWaste, GroundIdle, Defensive, Dribble |
| `src/PhaseManager.h/.cpp` | 4-phase curriculum config and transitions |
| `talos_phase1_plan.md` | This document |

**Reward class to file mapping:**

| Reward | File | Notes |
|---|---|---|
| GoalReward | `CommonRewards` (built-in) | `PlayerGoalReward` |
| GoalDistanceReward | `TalosRewards` | Custom |
| DefensiveReward | `TalosRewards` | Custom |
| GoalSpeedBonusReward | `TalosRewards` | Custom |
| TouchReward | `CommonRewards` (built-in) | `TouchBallReward` |
| PickupBoostReward | `CommonRewards` (built-in) | |
| DribbleReward | `TalosRewards` | Custom, per-player tracking |
| VelocityBallToGoal | `CommonRewards` (built-in) | `VelocityBallToGoalReward` |
| VelocityPlayerToBall | `CommonRewards` (built-in) | `VelocityPlayerToBallReward` |
| BoostWastePenalty | `TalosRewards` | Custom, negative |
| GroundIdlePenalty | `TalosRewards` | Custom, negative |

---

## 9. Build & Run

### Prerequisites

- CMake ≥ 3.8
- Visual Studio (MSVC) with C++20 support
- LibTorch installed (matching your system)
- GigaLearnCPP submodule cloned

### First-time setup

```powershell
# From the Talos project root:
git submodule update --init --recursive
# Or copy thirdparty from Talos v1 - RENDERER:
New-Item -ItemType Junction -Path "thirdparty" -Target "..\Talos v1 - RENDERER\thirdparty"
New-Item -ItemType Junction -Path "collision_meshes" -Target "..\Talos v1 - RENDERER\collision_meshes"
```

### Build

```powershell
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

### Run

```powershell
.\build\Release\Talos.exe
```

---

*End of Phase 1 design document. Phase 2 spec to follow after Phase 1 exit criteria are met.*
