# Nexto — Training Design Document

> **Last updated:** Phase 1 draft
> **Status:** Phase 1 implementation ready

---

## 1. Project Overview

**Nexto** is a self-play Rocket League bot trained with deep reinforcement learning using RLGym-CPP and PPO. The goal is a bot that progresses from basic mechanics to competitive play across four training phases.

The project builds on lessons from Talos v1 (network architecture baseline) and AetherRewards (custom reward implementations), extending both with new reward designs and a structured multi-phase curriculum.

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
| 1 | Nexto-lite | 0 – 30B | Score consistently, basic defense, ball on nose |
| 2 | Nexto-core | TBD | Win probability awareness, aerials, 50/50s |
| 3 | Nexto-plus | TBD | Wavedashes, speed mechanics, recovery |
| 4 | Nexto-full | TBD | Advanced team play, reads, rotation |

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

### 6.2 Training Setup

| Setting | Value |
|---|---|
| Parallel games | 36 |
| Hardware | CPU |
| Step budget | 30B (hard cap) |
| Expected duration | ~7–10 days |

### 6.3 Reward Table

| Reward | Weight | Purpose |
|---|---|---|
| GoalReward | 10.0 | Primary objective |
| GoalDistanceReward | 5.0 | Push ball toward net |
| DefensiveReward | 4.0 | Stay goal-side of ball |
| GoalSpeedBonusReward | 2.0 | Reward hard shots |
| TouchReward | 1.5 | Event-based contact signal |
| PickupBoostReward | 1.5 | Fuel management |
| DribbleReward | 1.0 | Controlled ball carry (gated — see §6.4) |
| VelocityBallToGoal | 1.0 | Ball moving toward opponent net |
| VelocityPlayerToBall | 0.6 | Reward approaching the ball |
| BoostWastePenalty | 0.5 | Boost efficiency |
| GroundIdlePenalty | 0.15 | Discourage stationary behavior |

#### Removed from Earlier Draft

| Reward | Reason |
|---|---|
| WinProbReward | Too complex for Phase 1; requires game-state comprehension the bot doesn't have yet. Moved to Phase 2 |
| BallProximityReward | Farmable — see §6.5 for full worked example |
| WavedashReward | Dead weight at 0.2 weight; gradient too small to reinforce anything. Phase 3+ concern |

### 6.4 DribbleReward — Three-Gate Implementation

Raw DribbleReward (ball near car) can be farmed by driving laps once the ball is balanced on the roof. Three conditions must be satisfied simultaneously:

```cpp
// Condition 1: Ball is physically on the car
//   Geometry check — ball above roof threshold
//   Velocity-delta check — ball and car are moving together
bool ball_on_car = (ball_relative_z > ROOF_THRESHOLD)
                && (abs(ball_vel - car_vel) < VEL_DELTA_MAX);

// Condition 2: Bot touched the ball recently
bool touch_gate = (steps_since_ball_touch < 20);

// Condition 3: Dribble has been sustained (not a one-tick fluke)
bool consecutive_active = (dribble_consecutive_ticks >= 8);

// Final gate — all three required
bool dribble_active = ball_on_car && touch_gate && consecutive_active;

float reward = dribble_active ? 1.0f : 0.0f;
```

**Why the velocity-delta check is the key insight:**
`ball_relative_z > ROOF_THRESHOLD` alone passes any time the ball lands on the roof by chance.
`abs(ball_vel - car_vel) < VEL_DELTA_MAX` is what separates *controlling the ball* from *the ball happened to land there*. A bot cannot fake velocity matching without actively carrying the ball.

**Parameter guidance:**

| Constant | Suggested Value | Notes |
|---|---|---|
| `ROOF_THRESHOLD` | ~60–80 UE units | Tune to car roof height in Unreal units |
| `VEL_DELTA_MAX` | ~150–200 UU/s | Tight enough to filter one-tick bounces |
| `steps_since_ball_touch` | < 20 | ~1.3s at 15 ticks/sec |
| `dribble_consecutive_ticks` | >= 8 | ~0.5s; filters accidental single-frame triggers |

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

### 6.6 Phase 1 Exit Criteria

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

| Reward | Class file | Header |
|---|---|---|
| GoalReward | `RLGymCPP/Rewards/CommonRewards` | Built-in |
| GoalDistanceReward | `TalosRewards.cpp` | `TalosRewards.h` |
| DefensiveReward | `AetherRewards.cpp` | `AetherRewards.h` |
| TouchReward | `RLGymCPP/Rewards/CommonRewards` | Built-in |
| VelocityBallToGoal | `AetherRewards.cpp` | `AetherRewards.h` |
| DribbleReward | `AetherRewards.cpp` | `AetherRewards.h` |
| VelocityPlayerToBall | `AetherRewards.cpp` | `AetherRewards.h` |
| PickupBoostReward | `RLGymCPP/Rewards/CommonRewards` | Built-in |
| GoalSpeedBonusReward | `TalosRewards.cpp` | `TalosRewards.h` |
| BoostWastePenalty | `TalosRewards.cpp` | `TalosRewards.h` |
| GroundIdlePenalty | `TalosRewards.cpp` | `TalosRewards.h` |

---

*End of Phase 1 design document. Phase 2 spec to follow after Phase 1 exit criteria are met.*
