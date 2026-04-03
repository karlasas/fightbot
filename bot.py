"""RoboKova Self-Learning Bot — Gets Smarter Every Match

This bot uses reinforcement learning to improve over time:
- Has ~15 tunable strategy "knobs" (weights)
- Tracks performance each match (score, survival, damage)
- After each match, adjusts weights toward what worked
- Starts with solid defaults, then optimizes through experience

The more matches it plays, the smarter it gets.

Run: uvicorn bot:app --host 0.0.0.0 --port 5001 --reload
"""

from __future__ import annotations

import json
import math
import os
import random
import time
from typing import Any

import httpx
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="RoboKova Learning Bot")

BOT_ID = os.environ.get("BOT_ID", "learner")
BOT_COLOR = os.environ.get("BOT_COLOR", "#00d2ff")
ARENA_URL = os.environ.get("ARENA_URL", "")


class MoveResponse(BaseModel):
    action: str
    emoji: str = ""
    mood: str = ""


# =====================================================================
# LEARNABLE STRATEGY WEIGHTS — these evolve over matches
# =====================================================================

# Defaults are tuned to be decent out of the box
strategy_weights = {
    # Combat
    "aggression": 0.45,            # 0=pacifist, 1=berserker
    "flee_health_pct": 0.28,       # flee when HP below this % of 100
    "defend_preference": 0.35,     # 0=never defend, 1=defend often
    "kill_chase_value": 0.70,      # how eagerly to chase low-HP enemies
    "helpless_enemy_bonus": 0.80,  # bonus for attacking energy-starved enemies
    "wait_out_defense": 0.75,      # tendency to wait when enemy is defending

    # Economy
    "resource_priority": 0.80,     # how much to value resources
    "energy_pack_priority": 0.85,  # priority for energy pack power-ups
    "damage_boost_priority": 0.90, # priority for damage boost power-ups
    "shield_priority": 0.75,       # priority for shield power-ups
    "speed_priority": 0.50,        # priority for speed boost
    "vision_priority": 0.25,       # priority for vision boost
    "min_energy_reserve": 3.0,     # keep at least this much energy

    # Survival
    "survival_value": 0.65,        # 0=yolo, 1=hide-and-survive
    "zone_prep_timing": 0.58,      # start moving to center at this % of match
    "center_pull": 0.40,           # how strongly to prefer center positions
}

# Clamp ranges for each weight
WEIGHT_BOUNDS = {
    "aggression": (0.05, 0.95),
    "flee_health_pct": (0.10, 0.50),
    "defend_preference": (0.10, 0.80),
    "kill_chase_value": (0.20, 0.95),
    "helpless_enemy_bonus": (0.30, 0.95),
    "wait_out_defense": (0.30, 0.95),
    "resource_priority": (0.30, 0.95),
    "energy_pack_priority": (0.40, 0.95),
    "damage_boost_priority": (0.50, 0.99),
    "shield_priority": (0.30, 0.95),
    "speed_priority": (0.15, 0.80),
    "vision_priority": (0.05, 0.60),
    "min_energy_reserve": (1.0, 8.0),
    "survival_value": (0.20, 0.90),
    "zone_prep_timing": (0.40, 0.70),
    "center_pull": (0.15, 0.75),
}


# =====================================================================
# LEARNING ENGINE
# =====================================================================

LEARNING_RATE = 0.08
EXPLORATION_NOISE = 0.03
MIN_MATCHES_BEFORE_LEARNING = 3

match_history: list[dict] = []  # stores results of completed matches
current_match: dict = {}        # tracks the ongoing match

# Stats display
total_matches = 0
total_wins = 0


def start_new_match(match_id: str, weights_snapshot: dict):
    """Initialize tracking for a new match."""
    global current_match
    current_match = {
        "match_id": match_id,
        "weights_used": dict(weights_snapshot),
        "start_time": time.time(),
        "turns_played": 0,
        "total_score_gained": 0,
        "total_damage_dealt": 0,
        "total_damage_taken": 0,
        "resources_collected": 0,
        "kills": 0,
        "last_health": 100,
        "last_score": 0,
        "last_energy": 20,
        "survived": True,
        "final_score": 0,
        "max_turns": 100,
        "num_bots": 2,
    }


def track_turn(state: dict):
    """Track changes between turns for reward calculation."""
    me = state["self"]
    hp = me["health"]
    score = me["score"]

    score_gained = max(0, score - current_match.get("last_score", 0))
    damage_taken = max(0, current_match.get("last_health", 100) - hp)

    current_match["turns_played"] += 1
    current_match["total_score_gained"] += score_gained
    current_match["total_damage_taken"] += damage_taken
    current_match["last_health"] = hp
    current_match["last_score"] = score
    current_match["last_energy"] = me["energy"]
    current_match["final_score"] = score
    current_match["max_turns"] = state["max_turns"]
    current_match["num_bots"] = state.get("num_bots", 2)

    # Detect kills from score jumps (kill = +30 score spike)
    if score_gained >= 30:
        current_match["kills"] += score_gained // 30

    # Detect resource collection from score jumps
    if score_gained >= 10 and score_gained < 30:
        current_match["resources_collected"] += 1


def finish_match():
    """Evaluate completed match and learn from it."""
    global total_matches, total_wins

    if not current_match.get("match_id"):
        return

    total_matches += 1

    # Calculate match reward
    reward = calculate_reward(current_match)
    current_match["reward"] = reward

    match_history.append(dict(current_match))

    # Learn from history
    if len(match_history) >= MIN_MATCHES_BEFORE_LEARNING:
        learn_from_history()

    # Track wins (heuristic: survived with high score)
    if current_match.get("survived", False):
        total_wins += 1


def calculate_reward(match: dict) -> float:
    """Calculate a single reward score for the match."""
    reward = 0.0

    # Core: final score is the main signal
    reward += match["final_score"] * 1.0

    # Survival bonus (big — placement bonus is huge)
    turns_pct = match["turns_played"] / max(1, match["max_turns"])
    reward += turns_pct * 50  # surviving longer = good

    # Full survival bonus
    if match.get("survived", False) and match["last_health"] > 0:
        reward += 40

    # Efficiency: score per damage taken
    if match["total_damage_taken"] > 0:
        efficiency = match["total_score_gained"] / match["total_damage_taken"]
        reward += min(efficiency * 10, 30)  # cap it
    elif match["total_score_gained"] > 0:
        reward += 30  # took no damage but scored — perfect

    # Kill bonus
    reward += match["kills"] * 15

    # Penalty for dying early
    if match["last_health"] <= 0 or (not match.get("survived", True)):
        early_death_penalty = (1 - turns_pct) * 60
        reward -= early_death_penalty

    # Scale by number of bots (harder matches = more reward)
    bot_multiplier = 1.0 + (match.get("num_bots", 2) - 2) * 0.1
    reward *= bot_multiplier

    return reward


def learn_from_history():
    """Adjust strategy weights based on match results."""
    global strategy_weights

    if len(match_history) < MIN_MATCHES_BEFORE_LEARNING:
        return

    # Compare recent matches to running average
    recent = match_history[-1]
    avg_reward = sum(m["reward"] for m in match_history) / len(match_history)
    recent_reward = recent["reward"]

    # How much better/worse than average
    advantage = (recent_reward - avg_reward) / max(abs(avg_reward), 1.0)
    # Clamp advantage to prevent wild swings
    advantage = max(-1.0, min(1.0, advantage))

    # Get the weights used in the recent match
    used_weights = recent.get("weights_used", {})

    for key in strategy_weights:
        if key not in used_weights:
            continue

        current = strategy_weights[key]
        used = used_weights[key]

        # Direction: if match was good, move toward used value
        # If match was bad, move away from used value
        diff = used - current
        nudge = LEARNING_RATE * advantage * (1 + abs(diff))

        # Apply update
        new_val = current + nudge

        # Add exploration noise (decreases over time)
        noise_scale = EXPLORATION_NOISE * max(0.3, 1.0 - len(match_history) / 100)
        new_val += random.gauss(0, noise_scale)

        # Clamp to valid range
        lo, hi = WEIGHT_BOUNDS.get(key, (0.0, 1.0))
        strategy_weights[key] = max(lo, min(hi, new_val))


def get_active_weights() -> dict:
    """Get current weights with slight exploration noise for this match."""
    w = dict(strategy_weights)
    # Small per-match exploration
    if len(match_history) < 30:
        for key in w:
            lo, hi = WEIGHT_BOUNDS.get(key, (0.0, 1.0))
            noise = random.gauss(0, EXPLORATION_NOISE * 0.5)
            w[key] = max(lo, min(hi, w[key] + noise))
    return w


# =====================================================================
# GAME UTILITIES (same solid foundation as before)
# =====================================================================

def manhattan(x1: int, y1: int, x2: int, y2: int) -> int:
    return abs(x1 - x2) + abs(y1 - y2)


def chebyshev(x1: int, y1: int, x2: int, y2: int) -> int:
    return max(abs(x1 - x2), abs(y1 - y2))


def center_of(arena_size: int) -> tuple[float, float]:
    c = (arena_size - 1) / 2
    return (c, c)


def dist_from_center(x: int, y: int, arena_size: int) -> float:
    cx, cy = center_of(arena_size)
    return max(abs(x - cx), abs(y - cy))


def is_in_safe_zone(x: int, y: int, arena_size: int, safe_radius: int) -> bool:
    cx, cy = center_of(arena_size)
    return abs(x - cx) <= safe_radius and abs(y - cy) <= safe_radius


def apply_move(x: int, y: int, action: str) -> tuple[int, int]:
    if action == "MOVE_UP":    return (x, y - 1)
    if action == "MOVE_DOWN":  return (x, y + 1)
    if action == "MOVE_LEFT":  return (x - 1, y)
    if action == "MOVE_RIGHT": return (x + 1, y)
    return (x, y)


def is_valid(x: int, y: int, arena_size: int, walls: set) -> bool:
    return 0 <= x < arena_size and 0 <= y < arena_size and (x, y) not in walls


ALL_MOVES = ["MOVE_UP", "MOVE_DOWN", "MOVE_LEFT", "MOVE_RIGHT"]


def get_walls(tiles: list[dict]) -> set[tuple[int, int]]:
    return {(t["x"], t["y"]) for t in tiles if t.get("type") == "wall"}


def smart_move(my_x: int, my_y: int, tx: int, ty: int,
               arena_size: int, walls: set, safe_radius: int,
               enemies: list[dict], avoid_enemies: bool,
               positions: list, center_w: float) -> str | None:
    """Unified smart movement considering all factors."""
    best_action = None
    best_score = -9999

    for action in ALL_MOVES:
        nx, ny = apply_move(my_x, my_y, action)
        if not is_valid(nx, ny, arena_size, walls):
            continue

        s = 0.0
        # Progress toward target
        old_d = manhattan(my_x, my_y, tx, ty)
        new_d = manhattan(nx, ny, tx, ty)
        s += (old_d - new_d) * 8

        # Safe zone
        if is_in_safe_zone(nx, ny, arena_size, safe_radius):
            s += 8
        else:
            s -= 18

        # Enemy avoidance
        if avoid_enemies:
            for e in enemies:
                d = chebyshev(nx, ny, e["x"], e["y"])
                if d <= 1: s -= 25
                elif d <= 2: s -= 8

        # Anti-oscillation
        if (nx, ny) in positions[-3:]: s -= 18
        elif (nx, ny) in positions[-6:]: s -= 8

        # Center pull (learnable)
        s -= dist_from_center(nx, ny, arena_size) * center_w

        if s > best_score:
            best_score = s
            best_action = action

    return best_action


def flee_move(my_x: int, my_y: int, enemies: list[dict],
              arena_size: int, walls: set, safe_radius: int,
              positions: list) -> str | None:
    """Best escape direction."""
    best_action = None
    best_score = -9999

    for action in ALL_MOVES:
        nx, ny = apply_move(my_x, my_y, action)
        if not is_valid(nx, ny, arena_size, walls):
            continue

        s = 0.0
        for e in enemies:
            old_d = chebyshev(my_x, my_y, e["x"], e["y"])
            new_d = chebyshev(nx, ny, e["x"], e["y"])
            s += (new_d - old_d) * 15

        if is_in_safe_zone(nx, ny, arena_size, safe_radius):
            s += 10
        else:
            s -= 22

        if (nx, ny) in positions[-4:]:
            s -= 12

        s -= dist_from_center(nx, ny, arena_size) * 0.5

        if s > best_score:
            best_score = s
            best_action = action

    return best_action


# =====================================================================
# COMBAT INTELLIGENCE (uses learned weights)
# =====================================================================

def my_damage(me: dict) -> int:
    stacks = me.get("damage_boost_stacks", 0)
    return 15 * (1 + stacks)


def effective_dmg(me: dict, enemy: dict) -> int:
    d = my_damage(me)
    if enemy.get("is_defending"):
        d //= 2
    return d


def can_one_shot(me: dict, enemy: dict) -> bool:
    return effective_dmg(me, enemy) >= enemy["health"]


def should_fight(me: dict, enemy: dict, adjacent_count: int,
                 phase: str, w: dict) -> bool:
    """Learned combat decision."""
    my_hp = me["health"]
    my_nrg = me["energy"]
    en_hp = enemy["health"]
    en_nrg = enemy.get("energy", 10)
    my_dmg = effective_dmg(me, enemy)

    # ALWAYS finish kills
    if en_hp <= my_dmg:
        return True

    # Don't attack defending enemies without boost
    if enemy.get("is_defending") and me.get("damage_boost_stacks", 0) == 0:
        if w["wait_out_defense"] > 0.5:
            return False

    # Enemy can't fight back — learned bonus
    if en_nrg < 3:
        return random.random() < w["helpless_enemy_bonus"]

    # Base aggression check
    fight_score = w["aggression"]

    # Modify by situation
    if my_hp > en_hp: fight_score += 0.15
    if my_nrg > en_nrg: fight_score += 0.10
    if me.get("damage_boost_stacks", 0) >= 1: fight_score += 0.20
    if me.get("shield_charges", 0) >= 1: fight_score += 0.15
    if adjacent_count >= 2: fight_score -= 0.30
    if my_hp < 40: fight_score -= 0.25
    if my_nrg < 6: fight_score -= 0.20
    if phase == "late": fight_score += 0.10 * (1 - w["survival_value"])

    return fight_score > 0.50


def should_flee(me: dict, adjacent: list[dict], w: dict) -> bool:
    hp_pct = me["health"] / 100
    if hp_pct <= w["flee_health_pct"]:
        return True
    if len(adjacent) >= 2 and hp_pct < w["flee_health_pct"] + 0.20:
        return True
    if me["health"] < 45 and me["energy"] < w["min_energy_reserve"] + 1:
        return True
    return False


# =====================================================================
# TARGET SCORING (uses learned weights)
# =====================================================================

def score_target(tile: dict, my_x: int, my_y: int, energy: int,
                 health: int, me: dict, phase: str,
                 arena_size: int, safe_radius: int, w: dict) -> float:
    tx, ty = tile["x"], tile["y"]
    dist = manhattan(my_x, my_y, tx, ty)

    if tile.get("has_resource"):
        if energy < dist + 2:
            return -1
    elif tile.get("power_up"):
        if energy < max(1, dist):
            return -1
    else:
        return -1

    # Don't chase things outside safe zone in late game
    if not is_in_safe_zone(tx, ty, arena_size, safe_radius):
        if phase == "late":
            return -1

    score = 0.0

    if tile.get("power_up"):
        pu = tile["power_up"]
        if pu == "damage_boost":
            score = 20 + w["damage_boost_priority"] * 30
        elif pu == "shield":
            score = 15 + w["shield_priority"] * 25
            if health < 50: score += 12
        elif pu == "energy_pack":
            score = 15 + w["energy_pack_priority"] * 25
            if energy < 10: score += 18
        elif pu == "speed_boost":
            score = 10 + w["speed_priority"] * 20
        elif pu == "vision_boost":
            score = 5 + w["vision_priority"] * 15
    elif tile.get("has_resource"):
        score = 10 + w["resource_priority"] * 20
        if energy < 10: score += 8
        if phase == "early": score += 5

    score -= dist * 2.5

    if is_in_safe_zone(tx, ty, arena_size, safe_radius):
        score += 4

    return score


# =====================================================================
# MAIN DECISION ENGINE
# =====================================================================

# Per-match tracking
positions: list[tuple[int, int]] = []
prev_action: str = ""
consecutive_waits: int = 0
active_weights: dict = {}


def choose_action(state: dict[str, Any]) -> MoveResponse:
    global positions, prev_action, consecutive_waits, active_weights

    me = state["self"]
    my_x, my_y = me["x"], me["y"]
    energy = me["energy"]
    health = me["health"]
    turn = state["turn"]
    max_turns = state["max_turns"]
    arena_size = state["arena_size"]
    safe_radius = state["safe_zone_radius"]
    num_bots = state.get("num_bots", 2)
    enemies = state.get("enemies", [])
    tiles = state.get("visible_tiles", [])
    dmg_stacks = me.get("damage_boost_stacks", 0)
    shields = me.get("shield_charges", 0)

    # Use active learned weights
    w = active_weights

    phase_pct = turn / max_turns
    if phase_pct < 0.30: phase = "early"
    elif phase_pct < 0.65: phase = "mid"
    else: phase = "late"

    walls = get_walls(tiles)
    in_safe = is_in_safe_zone(my_x, my_y, arena_size, safe_radius)

    positions.append((my_x, my_y))
    if len(positions) > 12:
        positions.pop(0)

    adjacent = [e for e in enemies if chebyshev(my_x, my_y, e["x"], e["y"]) <= 1]
    nearby = [e for e in enemies if 1 < chebyshev(my_x, my_y, e["x"], e["y"]) <= 3]

    on_resource = any(t.get("has_resource") and t["x"] == my_x and t["y"] == my_y for t in tiles)

    resources = [t for t in tiles if t.get("has_resource") and not (t["x"] == my_x and t["y"] == my_y)]
    power_ups = [t for t in tiles if t.get("power_up") and not (t["x"] == my_x and t["y"] == my_y)]
    all_targets = resources + power_ups

    my_dmg = my_damage(me)
    center_w = w.get("center_pull", 0.4)
    min_energy = w.get("min_energy_reserve", 3)

    def respond(action: str, emoji: str, mood: str) -> MoveResponse:
        global prev_action, consecutive_waits
        prev_action = action
        if action == "WAIT":
            consecutive_waits += 1
        else:
            consecutive_waits = 0
        return MoveResponse(action=action, emoji=emoji, mood=mood)

    # ============================================================
    # 1. ALWAYS COLLECT on resource (best action in the game!)
    # ============================================================
    if on_resource and energy >= 2:
        return respond("COLLECT", "💰", "farming")

    # ============================================================
    # 2. DANGER ZONE — escape or pre-position
    # ============================================================
    if not in_safe and phase_pct >= w.get("zone_prep_timing", 0.55):
        cx, cy = int(center_of(arena_size)[0]), int(center_of(arena_size)[1])
        action = smart_move(my_x, my_y, cx, cy, arena_size, walls,
                            safe_radius, enemies, True, positions, center_w)
        if action and energy >= 1:
            return respond(action, "🏃", "zone escape")

    # ============================================================
    # 3. FLEE when in danger
    # ============================================================
    if adjacent and should_flee(me, adjacent, w):
        # Parting shot if can one-shot
        killable = [e for e in adjacent if can_one_shot(me, e)]
        if killable and energy >= 3:
            return respond("ATTACK", "💀", "parting kill")

        # Defend or flee based on learned preference
        if health <= 20 and energy >= 2 and w["defend_preference"] > 0.5:
            return respond("DEFEND", "🛡️", "emergency shield")

        action = flee_move(my_x, my_y, enemies, arena_size, walls, safe_radius, positions)
        if action and energy >= 1:
            return respond(action, "💨", "retreat")

    # ============================================================
    # 4. ALWAYS take free kills
    # ============================================================
    if adjacent and energy >= 3:
        killable = [e for e in adjacent if can_one_shot(me, e)]
        if killable:
            return respond("ATTACK", "💀", "executing")

    # ============================================================
    # 5. HIT & RUN — if just attacked, flee
    # ============================================================
    if prev_action == "ATTACK" and adjacent and energy >= 1:
        action = flee_move(my_x, my_y, enemies, arena_size, walls, safe_radius, positions)
        if action:
            return respond(action, "💨", "hit and run")

    # ============================================================
    # 6. SMART COMBAT — fight with learned aggression
    # ============================================================
    if adjacent and energy >= 4:
        for enemy in adjacent:
            if should_fight(me, enemy, len(adjacent), phase, w):
                # Wait out defense if learned to do so
                if enemy.get("is_defending") and w["wait_out_defense"] > 0.5 and dmg_stacks == 0:
                    return respond("WAIT", "⏳", "waiting out defense")

                return respond("ATTACK", "⚔️", "calculated strike")

        # Not worth fighting — disengage
        if energy >= 1:
            action = flee_move(my_x, my_y, enemies, arena_size, walls, safe_radius, positions)
            if action:
                return respond(action, "💨", "disengage")

    # ============================================================
    # 7. CHASE TARGETS — scored by learned priorities
    # ============================================================
    if all_targets and energy >= 2:
        scored = []
        for t in all_targets:
            s = score_target(t, my_x, my_y, energy, health, me, phase,
                             arena_size, safe_radius, w)
            if s > 0:
                scored.append((s, t))
        scored.sort(key=lambda x: x[0], reverse=True)

        if scored:
            best = scored[0][1]
            avoid = health < 50 and w["survival_value"] > 0.4
            action = smart_move(my_x, my_y, best["x"], best["y"],
                                arena_size, walls, safe_radius,
                                enemies, avoid, positions, center_w)
            if action:
                label = best.get("power_up", "resource")
                return respond(action, "🎯", f"hunting {label}")

    # ============================================================
    # 8. HUNT WOUNDED (based on learned chase value)
    # ============================================================
    if nearby and energy >= 6 and health > 50:
        huntable = [e for e in nearby
                    if e["health"] < health
                    and e["health"] <= my_dmg * 3]
        if huntable and random.random() < w["kill_chase_value"]:
            target = min(huntable, key=lambda e: e["health"])
            action = smart_move(my_x, my_y, target["x"], target["y"],
                                arena_size, walls, safe_radius,
                                enemies, False, positions, center_w)
            if action:
                return respond(action, "🐺", "hunting enemy")

    # ============================================================
    # 9. LATE GAME CENTER
    # ============================================================
    if phase == "late":
        cd = dist_from_center(my_x, my_y, arena_size)
        if cd > max(1, safe_radius * 0.4) and energy >= 1:
            cx, cy = int(center_of(arena_size)[0]), int(center_of(arena_size)[1])
            action = smart_move(my_x, my_y, cx, cy, arena_size, walls,
                                safe_radius, enemies, True, positions, center_w)
            if action:
                return respond(action, "🏠", "centering")

    # ============================================================
    # 10. ENERGY MANAGEMENT
    # ============================================================
    if energy < min_energy:
        if consecutive_waits >= 4:
            # Break wait loop
            for action in ALL_MOVES:
                nx, ny = apply_move(my_x, my_y, action)
                if is_valid(nx, ny, arena_size, walls) and (nx, ny) not in positions[-3:]:
                    return respond(action, "🔍", "breaking loop")
        return respond("WAIT", "🔋", "recharging")

    # ============================================================
    # 11. EXPLORE
    # ============================================================
    best_action = None
    best_score = -9999
    random.shuffle(ALL_MOVES)

    for action in ALL_MOVES:
        nx, ny = apply_move(my_x, my_y, action)
        if not is_valid(nx, ny, arena_size, walls):
            continue
        s = 0.0
        if (nx, ny) in positions[-3:]: s -= 20
        elif (nx, ny) in positions[-6:]: s -= 10
        if is_in_safe_zone(nx, ny, arena_size, safe_radius): s += 8
        else: s -= 12
        s -= dist_from_center(nx, ny, arena_size) * center_w
        for e in enemies:
            if chebyshev(nx, ny, e["x"], e["y"]) <= 2: s -= 8
        s += random.random() * 5
        if s > best_score:
            best_score = s
            best_action = action

    if best_action and energy >= 1:
        return respond(best_action, "🔍", "exploring")

    return respond("WAIT", "😴", "idle")


# =====================================================================
# HTTP ENDPOINTS
# =====================================================================

@app.post("/move")
async def move(state: dict[str, Any]) -> MoveResponse:
    global positions, prev_action, consecutive_waits, active_weights

    match_id = state["match_id"]

    # Detect new match → finish old one, start fresh
    if match_id != current_match.get("match_id"):
        if current_match.get("match_id"):
            finish_match()

        # Reset per-match state
        positions = []
        prev_action = ""
        consecutive_waits = 0

        # Get fresh weights (with exploration for this match)
        active_weights = get_active_weights()
        start_new_match(match_id, active_weights)

    # Track this turn
    track_turn(state)

    # Make decision
    response = choose_action(state)

    # Optional: send status to arena UI
    if ARENA_URL:
        try:
            matches_info = f"M{total_matches}"
            if total_matches > 0:
                win_rate = int(total_wins / total_matches * 100)
                matches_info += f" W{win_rate}%"

            async with httpx.AsyncClient() as client:
                await client.post(
                    f"{ARENA_URL}/arena/bot-update",
                    json={
                        "bot_id": state["self"]["bot_id"],
                        "status": f"{response.mood.upper()} [{matches_info}]",
                        "message": f"T{state['turn']}: {response.action} | {matches_info}",
                        "color": BOT_COLOR,
                    },
                    timeout=0.3,
                )
        except Exception:
            pass

    return response


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "bot_id": BOT_ID}


@app.get("/stats")
async def stats() -> dict:
    """Check learning progress — visit /stats to see how the bot is evolving."""
    recent_rewards = [m["reward"] for m in match_history[-10:]] if match_history else []
    return {
        "total_matches": total_matches,
        "total_wins": total_wins,
        "win_rate": f"{total_wins / max(1, total_matches) * 100:.1f}%",
        "matches_in_memory": len(match_history),
        "current_weights": {k: round(v, 3) for k, v in strategy_weights.items()},
        "recent_avg_reward": round(sum(recent_rewards) / max(1, len(recent_rewards)), 1),
        "learning_rate": LEARNING_RATE,
    }
