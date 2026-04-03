"""RoboKova Self-Learning Bot v2 — Fixed & Improved

FIXES from v1:
- Death detection: tracks if we died (arena stops calling us before max_turns)
- Win tracking: properly distinguishes win / survived / died
- Resource collection: ALWAYS collects when standing on resource, no exceptions
- Better reward: dying early = big penalty, surviving = big bonus
- Power-up handling: routes THROUGH power-up tiles (auto-collected on walk)

Run: uvicorn bot:app --host 0.0.0.0 --port 5001 --reload
"""

from __future__ import annotations

import os
import random
import time
from typing import Any

import httpx
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="RoboKova Learning Bot v2")

BOT_ID = os.environ.get("BOT_ID", "learner-v2")
BOT_COLOR = os.environ.get("BOT_COLOR", "#00d2ff")
ARENA_URL = os.environ.get("ARENA_URL", "")


class MoveResponse(BaseModel):
    action: str
    emoji: str = ""
    mood: str = ""


# =====================================================================
# LEARNABLE WEIGHTS — these evolve with every match
# =====================================================================

strategy_weights = {
    "aggression": 0.45,
    "flee_health_pct": 0.28,
    "defend_preference": 0.35,
    "kill_chase_value": 0.70,
    "helpless_enemy_bonus": 0.80,
    "wait_out_defense": 0.75,
    "resource_priority": 0.80,
    "energy_pack_priority": 0.85,
    "damage_boost_priority": 0.90,
    "shield_priority": 0.75,
    "speed_priority": 0.50,
    "vision_priority": 0.25,
    "min_energy_reserve": 3.0,
    "survival_value": 0.65,
    "zone_prep_timing": 0.58,
    "center_pull": 0.40,
}

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

LEARNING_RATE = 0.08
EXPLORATION_NOISE = 0.03
MIN_MATCHES = 3


# =====================================================================
# MATCH TRACKING — properly detects wins, losses, deaths
# =====================================================================

match_history: list[dict] = []
current_match: dict = {}
total_matches = 0
total_wins = 0
total_deaths = 0

# Per-match volatile state
positions: list[tuple[int, int]] = []
prev_action: str = ""
consecutive_waits: int = 0
active_weights: dict = {}


def start_new_match(match_id: str, weights: dict):
    global current_match
    current_match = {
        "match_id": match_id,
        "weights_used": dict(weights),
        "turns_played": 0,
        "max_turns": 100,
        "num_bots": 2,
        "last_health": 100,
        "last_score": 0,
        "peak_score": 0,
        "total_damage_taken": 0,
        "total_score_gained": 0,
        "kills": 0,
        "resources_collected": 0,
    }


def track_turn(state: dict):
    """Record what happened this turn."""
    me = state["self"]
    hp = me["health"]
    score = me["score"]
    prev_hp = current_match.get("last_health", 100)
    prev_score = current_match.get("last_score", 0)

    damage_taken = max(0, prev_hp - hp)
    score_gained = max(0, score - prev_score)

    current_match["turns_played"] += 1
    current_match["max_turns"] = state["max_turns"]
    current_match["num_bots"] = state.get("num_bots", 2)
    current_match["last_health"] = hp
    current_match["last_score"] = score
    current_match["peak_score"] = max(current_match.get("peak_score", 0), score)
    current_match["total_damage_taken"] += damage_taken
    current_match["total_score_gained"] += score_gained

    if score_gained >= 30:
        current_match["kills"] += 1
    if score_gained >= 10 and score_gained < 30:
        current_match["resources_collected"] += 1


def finish_match():
    """Evaluate the completed match and learn."""
    global total_matches, total_wins, total_deaths

    if not current_match.get("match_id"):
        return

    total_matches += 1

    turns = current_match["turns_played"]
    max_t = current_match["max_turns"]
    last_hp = current_match["last_health"]

    # === DETECT OUTCOME ===
    # If we played all turns (or close to it), we survived
    # If arena stopped calling us early, we died
    survived_to_end = turns >= max_t * 0.90
    probably_died = turns < max_t * 0.80 and last_hp < 30

    if survived_to_end:
        outcome = "survived"
        total_wins += 1  # survived = at least didn't die
    elif probably_died:
        outcome = "died"
        total_deaths += 1
    else:
        # Ambiguous — might have died near the end or match ended with few turns
        outcome = "unclear"
        if last_hp <= 15:
            total_deaths += 1
            outcome = "died_late"

    current_match["outcome"] = outcome

    # === CALCULATE REWARD ===
    reward = 0.0

    # Score earned is the primary signal
    reward += current_match["last_score"] * 1.0

    # Survival bonus
    survival_pct = turns / max(1, max_t)
    reward += survival_pct * 40

    if outcome == "survived":
        reward += 50  # big bonus for surviving
    elif outcome == "died":
        reward -= 40 * (1 - survival_pct)  # dying early = big penalty
    elif outcome == "died_late":
        reward -= 15

    # Efficiency bonus: high score with low damage taken
    if current_match["total_damage_taken"] > 0:
        eff = current_match["total_score_gained"] / current_match["total_damage_taken"]
        reward += min(eff * 8, 25)
    elif current_match["total_score_gained"] > 0:
        reward += 25

    # Kill bonus
    reward += current_match["kills"] * 12

    # Scale by bot count (harder = more reward)
    bots = current_match.get("num_bots", 2)
    reward *= 1.0 + (bots - 2) * 0.08

    current_match["reward"] = reward
    match_history.append(dict(current_match))

    # === LEARN ===
    if len(match_history) >= MIN_MATCHES:
        learn()


def learn():
    """Adjust weights based on recent match vs average."""
    global strategy_weights

    recent = match_history[-1]
    avg_reward = sum(m["reward"] for m in match_history) / len(match_history)
    advantage = (recent["reward"] - avg_reward) / max(abs(avg_reward), 1.0)
    advantage = max(-1.0, min(1.0, advantage))

    used = recent.get("weights_used", {})

    for key in strategy_weights:
        if key not in used:
            continue

        current = strategy_weights[key]
        diff = used[key] - current
        nudge = LEARNING_RATE * advantage * (1 + abs(diff))

        new_val = current + nudge

        # Exploration noise (decreases over time)
        noise_scale = EXPLORATION_NOISE * max(0.25, 1.0 - len(match_history) / 120)
        new_val += random.gauss(0, noise_scale)

        lo, hi = WEIGHT_BOUNDS.get(key, (0.0, 1.0))
        strategy_weights[key] = max(lo, min(hi, new_val))


def get_active_weights() -> dict:
    w = dict(strategy_weights)
    if len(match_history) < 40:
        for key in w:
            lo, hi = WEIGHT_BOUNDS.get(key, (0.0, 1.0))
            w[key] = max(lo, min(hi, w[key] + random.gauss(0, EXPLORATION_NOISE * 0.4)))
    return w


# =====================================================================
# GAME UTILITIES
# =====================================================================

def manhattan(x1, y1, x2, y2):
    return abs(x1 - x2) + abs(y1 - y2)


def chebyshev(x1, y1, x2, y2):
    return max(abs(x1 - x2), abs(y1 - y2))


def center_of(sz):
    c = (sz - 1) / 2
    return (c, c)


def dist_center(x, y, sz):
    cx, cy = center_of(sz)
    return max(abs(x - cx), abs(y - cy))


def in_safe(x, y, sz, sr):
    cx, cy = center_of(sz)
    return abs(x - cx) <= sr and abs(y - cy) <= sr


def apply(x, y, a):
    if a == "MOVE_UP":    return (x, y - 1)
    if a == "MOVE_DOWN":  return (x, y + 1)
    if a == "MOVE_LEFT":  return (x - 1, y)
    if a == "MOVE_RIGHT": return (x + 1, y)
    return (x, y)


def valid(x, y, sz, walls):
    return 0 <= x < sz and 0 <= y < sz and (x, y) not in walls


MOVES = ["MOVE_UP", "MOVE_DOWN", "MOVE_LEFT", "MOVE_RIGHT"]


def walls_from(tiles):
    return {(t["x"], t["y"]) for t in tiles if t.get("type") == "wall"}


# Smart pathfinding
def best_move_to(mx, my, tx, ty, sz, walls, sr, enemies, avoid_en, pos, cw):
    best_a = None
    best_s = -9999
    for a in MOVES:
        nx, ny = apply(mx, my, a)
        if not valid(nx, ny, sz, walls):
            continue
        s = 0.0
        # Closer to target
        s += (manhattan(mx, my, tx, ty) - manhattan(nx, ny, tx, ty)) * 8
        # Safe zone
        if in_safe(nx, ny, sz, sr): s += 8
        else: s -= 20
        # Enemy avoidance
        if avoid_en:
            for e in enemies:
                d = chebyshev(nx, ny, e["x"], e["y"])
                if d <= 1: s -= 28
                elif d <= 2: s -= 10
        # Anti-oscillation
        if (nx, ny) in pos[-3:]: s -= 20
        elif (nx, ny) in pos[-6:]: s -= 8
        # Center pull
        s -= dist_center(nx, ny, sz) * cw
        if s > best_s:
            best_s = s
            best_a = a
    return best_a


def flee(mx, my, enemies, sz, walls, sr, pos):
    best_a = None
    best_s = -9999
    for a in MOVES:
        nx, ny = apply(mx, my, a)
        if not valid(nx, ny, sz, walls):
            continue
        s = 0.0
        for e in enemies:
            s += (chebyshev(nx, ny, e["x"], e["y"]) - chebyshev(mx, my, e["x"], e["y"])) * 15
        if in_safe(nx, ny, sz, sr): s += 10
        else: s -= 24
        if (nx, ny) in pos[-4:]: s -= 12
        s -= dist_center(nx, ny, sz) * 0.5
        if s > best_s:
            best_s = s
            best_a = a
    return best_a


# =====================================================================
# COMBAT INTELLIGENCE
# =====================================================================

def my_dmg(me):
    return 15 * (1 + me.get("damage_boost_stacks", 0))


def eff_dmg(me, en):
    d = my_dmg(me)
    if en.get("is_defending"): d //= 2
    return d


def can_kill(me, en):
    return eff_dmg(me, en) >= en["health"]


def should_fight(me, en, adj_count, phase, w):
    d = eff_dmg(me, en)
    # Always take free kills
    if en["health"] <= d: return True
    # Don't waste energy on defending enemies without boost
    if en.get("is_defending") and me.get("damage_boost_stacks", 0) == 0:
        return False
    # Enemy can't attack — exploit
    if en.get("energy", 10) < 3:
        return random.random() < w["helpless_enemy_bonus"]
    # Aggression score
    score = w["aggression"]
    if me["health"] > en["health"]: score += 0.15
    if me["energy"] > en.get("energy", 10): score += 0.10
    if me.get("damage_boost_stacks", 0) >= 1: score += 0.20
    if me.get("shield_charges", 0) >= 1: score += 0.15
    if adj_count >= 2: score -= 0.30
    if me["health"] < 40: score -= 0.25
    if me["energy"] < 6: score -= 0.20
    if phase == "late": score += 0.10 * (1 - w["survival_value"])
    return score > 0.50


def should_flee(me, adjacent, w):
    hp_pct = me["health"] / 100
    if hp_pct <= w["flee_health_pct"]: return True
    if len(adjacent) >= 2 and hp_pct < w["flee_health_pct"] + 0.20: return True
    if me["health"] < 45 and me["energy"] < w["min_energy_reserve"] + 1: return True
    return False


# =====================================================================
# TARGET SCORING
# =====================================================================

def score_target(tile, mx, my, energy, health, me, phase, sz, sr, w):
    tx, ty = tile["x"], tile["y"]
    dist = manhattan(mx, my, tx, ty)

    is_resource = tile.get("has_resource", False)
    power_up = tile.get("power_up")

    if is_resource:
        cost = dist + 2  # move + collect
        net_energy = cost - 3  # collect returns 3
        if energy < cost: return -1
    elif power_up:
        cost = max(1, dist)  # just walk there
        if energy < cost: return -1
    else:
        return -1

    if not in_safe(tx, ty, sz, sr) and phase == "late":
        return -1

    s = 0.0

    if power_up:
        if power_up == "damage_boost":
            s = 20 + w["damage_boost_priority"] * 30
        elif power_up == "shield":
            s = 15 + w["shield_priority"] * 25
            if health < 50: s += 12
        elif power_up == "energy_pack":
            s = 15 + w["energy_pack_priority"] * 25
            if energy < 10: s += 18
        elif power_up == "speed_boost":
            s = 10 + w["speed_priority"] * 20
        elif power_up == "vision_boost":
            s = 5 + w["vision_priority"] * 15
    elif is_resource:
        s = 10 + w["resource_priority"] * 20
        if energy < 10: s += 8
        if phase == "early": s += 5

    s -= dist * 2.5
    if in_safe(tx, ty, sz, sr): s += 4

    return s


# =====================================================================
# MAIN DECISION ENGINE
# =====================================================================

def choose_action(state: dict[str, Any]) -> MoveResponse:
    global prev_action, consecutive_waits

    me = state["self"]
    mx, my = me["x"], me["y"]
    energy = me["energy"]
    health = me["health"]
    turn = state["turn"]
    max_turns = state["max_turns"]
    sz = state["arena_size"]
    sr = state["safe_zone_radius"]
    enemies = state.get("enemies", [])
    tiles = state.get("visible_tiles", [])

    w = active_weights
    phase_pct = turn / max_turns
    phase = "early" if phase_pct < 0.30 else ("mid" if phase_pct < 0.65 else "late")
    walls = walls_from(tiles)
    is_safe = in_safe(mx, my, sz, sr)
    cw = w.get("center_pull", 0.4)

    positions.append((mx, my))
    if len(positions) > 12: positions.pop(0)

    adjacent = [e for e in enemies if chebyshev(mx, my, e["x"], e["y"]) <= 1]
    nearby = [e for e in enemies if 1 < chebyshev(mx, my, e["x"], e["y"]) <= 3]

    # Find what's on our tile and what's nearby
    on_resource = any(t.get("has_resource") and t["x"] == mx and t["y"] == my for t in tiles)

    other_resources = [t for t in tiles
                       if t.get("has_resource") and not (t["x"] == mx and t["y"] == my)]
    other_powerups = [t for t in tiles
                      if t.get("power_up") and not (t["x"] == mx and t["y"] == my)]
    all_targets = other_resources + other_powerups

    def respond(action, emoji, mood):
        global prev_action, consecutive_waits
        prev_action = action
        consecutive_waits = consecutive_waits + 1 if action == "WAIT" else 0
        return MoveResponse(action=action, emoji=emoji, mood=mood)

    # ==============================================================
    # 1. ALWAYS COLLECT RESOURCE UNDER US — #1 priority, no exceptions!
    #    Costs 2 energy, gives +3 back = free +1 energy AND +10 score
    #    This is the best action in the entire game.
    # ==============================================================
    if on_resource and energy >= 2:
        return respond("COLLECT", "💰", "farming")

    # ==============================================================
    # 2. DANGER ZONE — get to safety
    # ==============================================================
    if not is_safe and phase_pct >= w.get("zone_prep_timing", 0.58):
        cx = int(center_of(sz)[0])
        cy = int(center_of(sz)[1])
        a = best_move_to(mx, my, cx, cy, sz, walls, sr, enemies, True, positions, cw)
        if a and energy >= 1:
            return respond(a, "🏃", "zone escape")

    # ==============================================================
    # 3. FLEE — when we should run
    # ==============================================================
    if adjacent and should_flee(me, adjacent, w):
        # Parting kill?
        kills = [e for e in adjacent if can_kill(me, e)]
        if kills and energy >= 3:
            return respond("ATTACK", "💀", "parting kill")
        # Defend if extremely low
        if health <= 20 and energy >= 2 and w["defend_preference"] > 0.5:
            return respond("DEFEND", "🛡️", "emergency")
        # Run
        a = flee(mx, my, enemies, sz, walls, sr, positions)
        if a and energy >= 1:
            return respond(a, "💨", "retreat")

    # ==============================================================
    # 4. FREE KILLS — always finish enemies off
    # ==============================================================
    if adjacent and energy >= 3:
        kills = [e for e in adjacent if can_kill(me, e)]
        if kills:
            return respond("ATTACK", "💀", "executing")

    # ==============================================================
    # 5. HIT & RUN — just attacked? now escape
    # ==============================================================
    if prev_action == "ATTACK" and adjacent and energy >= 1:
        a = flee(mx, my, enemies, sz, walls, sr, positions)
        if a:
            return respond(a, "💨", "hit & run")

    # ==============================================================
    # 6. SMART COMBAT — learned aggression
    # ==============================================================
    if adjacent and energy >= 4:
        fought = False
        for en in adjacent:
            if should_fight(me, en, len(adjacent), phase, w):
                # Wait out defense?
                if en.get("is_defending") and w["wait_out_defense"] > 0.5:
                    if me.get("damage_boost_stacks", 0) == 0:
                        return respond("WAIT", "⏳", "wait defense")
                return respond("ATTACK", "⚔️", "strike")

        # Not worth fighting — leave
        if not fought and energy >= 1:
            a = flee(mx, my, enemies, sz, walls, sr, positions)
            if a:
                return respond(a, "💨", "disengage")

    # ==============================================================
    # 7. CHASE TARGETS — go after best resource/power-up
    #    NOTE: power-ups auto-collect when walking onto them!
    #    So we just need to MOVE to the power-up tile.
    # ==============================================================
    if all_targets and energy >= 2:
        scored = []
        for t in all_targets:
            s = score_target(t, mx, my, energy, health, me, phase, sz, sr, w)
            if s > 0:
                scored.append((s, t))
        scored.sort(key=lambda x: x[0], reverse=True)

        if scored:
            best = scored[0][1]
            avoid = health < 50 and w["survival_value"] > 0.4
            a = best_move_to(mx, my, best["x"], best["y"],
                             sz, walls, sr, enemies, avoid, positions, cw)
            if a:
                label = best.get("power_up", "resource")
                return respond(a, "🎯", f"get {label}")

    # ==============================================================
    # 8. HUNT WOUNDED ENEMIES
    # ==============================================================
    if nearby and energy >= 6 and health > 50:
        md = my_dmg(me)
        huntable = [e for e in nearby if e["health"] < health and e["health"] <= md * 3]
        if huntable and random.random() < w["kill_chase_value"]:
            target = min(huntable, key=lambda e: e["health"])
            a = best_move_to(mx, my, target["x"], target["y"],
                             sz, walls, sr, enemies, False, positions, cw)
            if a:
                return respond(a, "🐺", "hunting")

    # ==============================================================
    # 9. LATE GAME — center up
    # ==============================================================
    if phase == "late":
        cd = dist_center(mx, my, sz)
        if cd > max(1, sr * 0.4) and energy >= 1:
            cx = int(center_of(sz)[0])
            cy = int(center_of(sz)[1])
            a = best_move_to(mx, my, cx, cy, sz, walls, sr, enemies, True, positions, cw)
            if a:
                return respond(a, "🏠", "center")

    # ==============================================================
    # 10. ENERGY — recharge or break loops
    # ==============================================================
    if energy < w.get("min_energy_reserve", 3):
        if consecutive_waits >= 4:
            for a in MOVES:
                nx, ny = apply(mx, my, a)
                if valid(nx, ny, sz, walls) and (nx, ny) not in positions[-3:]:
                    return respond(a, "🔍", "break loop")
        return respond("WAIT", "🔋", "recharge")

    # ==============================================================
    # 11. EXPLORE
    # ==============================================================
    best_a = None
    best_s = -9999
    shuffled = list(MOVES)
    random.shuffle(shuffled)

    for a in shuffled:
        nx, ny = apply(mx, my, a)
        if not valid(nx, ny, sz, walls): continue
        s = 0.0
        if (nx, ny) in positions[-3:]: s -= 22
        elif (nx, ny) in positions[-6:]: s -= 10
        if in_safe(nx, ny, sz, sr): s += 8
        else: s -= 12
        s -= dist_center(nx, ny, sz) * cw
        for e in enemies:
            if chebyshev(nx, ny, e["x"], e["y"]) <= 2: s -= 8
        s += random.random() * 5
        if s > best_s:
            best_s = s
            best_a = a

    if best_a and energy >= 1:
        return respond(best_a, "🔍", "scout")

    return respond("WAIT", "😴", "idle")


# =====================================================================
# HTTP ENDPOINTS
# =====================================================================

@app.post("/move")
async def move(state: dict[str, Any]) -> MoveResponse:
    global positions, prev_action, consecutive_waits, active_weights

    match_id = state["match_id"]

    # New match? Finish old one, start fresh
    if match_id != current_match.get("match_id"):
        if current_match.get("match_id"):
            finish_match()

        positions = []
        prev_action = ""
        consecutive_waits = 0
        active_weights = get_active_weights()
        start_new_match(match_id, active_weights)

    track_turn(state)
    response = choose_action(state)

    if ARENA_URL:
        try:
            async with httpx.AsyncClient() as client:
                await client.post(
                    f"{ARENA_URL}/arena/bot-update",
                    json={
                        "bot_id": state["self"]["bot_id"],
                        "status": response.mood.upper(),
                        "message": f"T{state['turn']}: {response.action}",
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
    """Visit /stats to see learning progress."""
    recent = match_history[-10:] if match_history else []
    recent_rewards = [m["reward"] for m in recent]
    outcomes = [m.get("outcome", "?") for m in recent]

    return {
        "total_matches": total_matches,
        "survived": total_wins,
        "died": total_deaths,
        "survival_rate": f"{total_wins / max(1, total_matches) * 100:.0f}%",
        "matches_in_memory": len(match_history),
        "recent_outcomes": outcomes,
        "recent_avg_reward": round(sum(recent_rewards) / max(1, len(recent_rewards)), 1) if recent_rewards else 0,
        "weights": {k: round(v, 3) for k, v in strategy_weights.items()},
    }
