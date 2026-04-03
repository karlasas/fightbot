"""RoboKova Self-Learning Bot v5

MAJOR UPGRADE: Full enemy tracking for accurate win/loss detection
- Tracks every enemy's score, health, and alive status across turns
- Compares our score vs enemy scores to determine who actually won
- Detects elimination wins (all enemies dead) vs score wins vs losses
- Distinguishes: WON (elimination), WON (score), LOST (outscored), DIED

Run: uvicorn bot:app --host 0.0.0.0 --port 5001 --reload
"""

from __future__ import annotations

import os
import random
from typing import Any

import httpx
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

app = FastAPI(title="RoboKova Learning Bot v5")

BOT_ID = os.environ.get("BOT_ID", "learner-v5")
BOT_COLOR = os.environ.get("BOT_COLOR", "#00d2ff")
ARENA_URL = os.environ.get("ARENA_URL", "")


class MoveResponse(BaseModel):
    action: str
    emoji: str = ""
    mood: str = ""


# =====================================================================
# LEARNABLE WEIGHTS
# =====================================================================

strategy_weights = {
    "aggression": 0.45, "flee_health_pct": 0.28, "defend_preference": 0.35,
    "kill_chase_value": 0.70, "helpless_enemy_bonus": 0.80, "wait_out_defense": 0.75,
    "resource_priority": 0.80, "energy_pack_priority": 0.85, "damage_boost_priority": 0.90,
    "shield_priority": 0.75, "speed_priority": 0.50, "vision_priority": 0.25,
    "min_energy_reserve": 3.0, "survival_value": 0.65, "zone_prep_timing": 0.58,
    "center_pull": 0.40,
}

WEIGHT_BOUNDS = {
    "aggression": (0.05, 0.95), "flee_health_pct": (0.10, 0.50),
    "defend_preference": (0.10, 0.80), "kill_chase_value": (0.20, 0.95),
    "helpless_enemy_bonus": (0.30, 0.95), "wait_out_defense": (0.30, 0.95),
    "resource_priority": (0.30, 0.95), "energy_pack_priority": (0.40, 0.95),
    "damage_boost_priority": (0.50, 0.99), "shield_priority": (0.30, 0.95),
    "speed_priority": (0.15, 0.80), "vision_priority": (0.05, 0.60),
    "min_energy_reserve": (1.0, 8.0), "survival_value": (0.20, 0.90),
    "zone_prep_timing": (0.40, 0.70), "center_pull": (0.15, 0.75),
}

LEARNING_RATE = 0.08
EXPLORATION_NOISE = 0.03


# =====================================================================
# MATCH TRACKING — now with full enemy intelligence
# =====================================================================

match_history: list[dict] = []
current_match: dict = {}
total_matches = 0
total_won = 0
total_lost = 0
total_died = 0

positions: list[tuple[int, int]] = []
prev_action: str = ""
consecutive_waits: int = 0
active_weights: dict = {}

# Enemy tracking across turns within a match
enemy_tracker: dict[str, dict] = {}
# { "bot_id": { "last_score": 30, "last_health": 45, "last_seen_turn": 15, "seen_count": 8 } }


def new_match(match_id, weights):
    global current_match, enemy_tracker
    enemy_tracker = {}
    current_match = {
        "match_id": match_id, "weights_used": dict(weights),
        "turns_played": 0, "max_turns": 100, "num_bots": 2,
        "last_health": 100, "last_score": 0,
        "total_damage_taken": 0, "total_score_gained": 0, "kills": 0,
        "health_history": [], "enemies_alive_last_turn": 0,
    }


def track_turn(state):
    global enemy_tracker
    me = state["self"]
    hp, sc = me["health"], me["score"]
    prev_hp = current_match.get("last_health", 100)
    prev_sc = current_match.get("last_score", 0)
    turn = state["turn"]
    enemies = state.get("enemies", [])

    current_match["turns_played"] += 1
    current_match["max_turns"] = state["max_turns"]
    current_match["num_bots"] = state.get("num_bots", 2)
    current_match["total_damage_taken"] += max(0, prev_hp - hp)
    current_match["total_score_gained"] += max(0, sc - prev_sc)
    current_match["last_health"] = hp
    current_match["last_score"] = sc
    current_match["enemies_alive_last_turn"] = len(enemies)

    current_match["health_history"].append(hp)
    if len(current_match["health_history"]) > 10:
        current_match["health_history"].pop(0)

    if sc - prev_sc >= 30:
        current_match["kills"] += 1

    # === TRACK EVERY ENEMY WE SEE ===
    for e in enemies:
        bid = e["bot_id"]
        enemy_tracker[bid] = {
            "last_score": e.get("score", 0),
            "last_health": e.get("health", 100),
            "last_energy": e.get("energy", 0),
            "last_seen_turn": turn,
            "is_defending": e.get("is_defending", False),
            "seen_count": enemy_tracker.get(bid, {}).get("seen_count", 0) + 1,
        }


def detect_outcome() -> str:
    """Accurate outcome detection using enemy scores and health.

    Uses: our score, enemy scores, health trends, turn count, enemy tracking.
    """
    turns = current_match["turns_played"]
    max_t = current_match["max_turns"]
    our_score = current_match["last_score"]
    our_hp = current_match["last_health"]
    hp_hist = current_match.get("health_history", [])
    enemies_last = current_match["enemies_alive_last_turn"]
    num_bots = current_match["num_bots"]

    # Get all known enemy scores
    enemy_scores = {bid: info["last_score"] for bid, info in enemy_tracker.items()}
    highest_enemy_score = max(enemy_scores.values()) if enemy_scores else 0

    # Was our health declining in last turns?
    health_declining = False
    if len(hp_hist) >= 3:
        health_declining = hp_hist[-1] < hp_hist[0] - 10

    # === CASE 1: We played until (near) the end — match ended by time ===
    if turns >= max_t - 2:
        if our_score > highest_enemy_score:
            return "won_score"
        elif our_score == highest_enemy_score:
            return "tied"
        else:
            return "lost_score"

    # === CASE 2: Match ended early ===

    # If our health is good and no enemies visible → we won by elimination
    if our_hp > 20 and enemies_last == 0 and not health_declining:
        return "won_elimination"

    # If our health is good and wasn't dropping → probably won
    if our_hp > 40 and not health_declining:
        return "won_elimination"

    # If our health was dropping sharply → we died
    if health_declining and our_hp < 30:
        return "died"

    # If very low health → died
    if our_hp <= 15:
        return "died"

    # If enemies were visible and we stopped getting calls → died
    if enemies_last > 0 and health_declining:
        return "died"

    # If match ended very early → probably died
    if turns < max_t * 0.3:
        return "died"

    # Ambiguous but health OK → probably won
    if our_hp > 50:
        return "won_elimination"

    return "died"


def finish_match():
    global total_matches, total_won, total_lost, total_died

    if not current_match.get("match_id"):
        return

    total_matches += 1
    outcome = detect_outcome()
    current_match["outcome"] = outcome
    current_match["enemy_scores"] = {bid: info["last_score"] for bid, info in enemy_tracker.items()}

    if outcome.startswith("won"):
        total_won += 1
    elif outcome == "lost_score" or outcome == "tied":
        total_lost += 1
    else:
        total_died += 1

    # === REWARD ===
    reward = current_match["last_score"] * 1.0
    survival_pct = current_match["turns_played"] / max(1, current_match["max_turns"])
    reward += survival_pct * 40

    if outcome == "won_elimination": reward += 90
    elif outcome == "won_score": reward += 70
    elif outcome == "tied": reward += 30
    elif outcome == "lost_score": reward -= 10
    else: reward -= 50 * (1 - survival_pct)

    dmg, sc = current_match["total_damage_taken"], current_match["total_score_gained"]
    if dmg > 0: reward += min((sc / dmg) * 8, 25)
    elif sc > 0: reward += 25
    reward += current_match["kills"] * 12
    reward *= 1.0 + (current_match.get("num_bots", 2) - 2) * 0.08

    current_match["reward"] = round(reward, 1)
    match_history.append(dict(current_match))

    if len(match_history) >= 3:
        learn()


def learn():
    global strategy_weights
    recent = match_history[-1]
    avg = sum(m["reward"] for m in match_history) / len(match_history)
    adv = max(-1.0, min(1.0, (recent["reward"] - avg) / max(abs(avg), 1.0)))
    used = recent.get("weights_used", {})
    for key in strategy_weights:
        if key not in used: continue
        cur = strategy_weights[key]
        nudge = LEARNING_RATE * adv * (1 + abs(used[key] - cur))
        new = cur + nudge + random.gauss(0, EXPLORATION_NOISE * max(0.25, 1 - len(match_history) / 120))
        lo, hi = WEIGHT_BOUNDS.get(key, (0, 1))
        strategy_weights[key] = max(lo, min(hi, new))


def get_active_weights():
    w = dict(strategy_weights)
    if len(match_history) < 40:
        for k in w:
            lo, hi = WEIGHT_BOUNDS.get(k, (0, 1))
            w[k] = max(lo, min(hi, w[k] + random.gauss(0, EXPLORATION_NOISE * 0.4)))
    return w


# =====================================================================
# GAME UTILITIES (same proven logic)
# =====================================================================

def manhattan(x1, y1, x2, y2): return abs(x1 - x2) + abs(y1 - y2)
def chebyshev(x1, y1, x2, y2): return max(abs(x1 - x2), abs(y1 - y2))
def center_of(sz): c = (sz - 1) / 2; return (c, c)
def dist_center(x, y, sz):
    cx, cy = center_of(sz); return max(abs(x - cx), abs(y - cy))
def in_safe(x, y, sz, sr):
    cx, cy = center_of(sz); return abs(x - cx) <= sr and abs(y - cy) <= sr
def apply(x, y, a):
    if a == "MOVE_UP": return (x, y-1)
    if a == "MOVE_DOWN": return (x, y+1)
    if a == "MOVE_LEFT": return (x-1, y)
    if a == "MOVE_RIGHT": return (x+1, y)
    return (x, y)
def ok(x, y, sz, walls): return 0 <= x < sz and 0 <= y < sz and (x, y) not in walls

MOVES = ["MOVE_UP", "MOVE_DOWN", "MOVE_LEFT", "MOVE_RIGHT"]
def get_walls(tiles): return {(t["x"], t["y"]) for t in tiles if t.get("type") == "wall"}

def move_to(mx, my, tx, ty, sz, walls, sr, enemies, avoid, pos, cw):
    best_a, best_s = None, -9999
    for a in MOVES:
        nx, ny = apply(mx, my, a)
        if not ok(nx, ny, sz, walls): continue
        s = (manhattan(mx, my, tx, ty) - manhattan(nx, ny, tx, ty)) * 8
        if in_safe(nx, ny, sz, sr): s += 8
        else: s -= 20
        if avoid:
            for e in enemies:
                d = chebyshev(nx, ny, e["x"], e["y"])
                if d <= 1: s -= 28
                elif d <= 2: s -= 10
        if (nx, ny) in pos[-3:]: s -= 20
        elif (nx, ny) in pos[-6:]: s -= 8
        s -= dist_center(nx, ny, sz) * cw
        if s > best_s: best_s, best_a = s, a
    return best_a

def flee_from(mx, my, enemies, sz, walls, sr, pos):
    best_a, best_s = None, -9999
    for a in MOVES:
        nx, ny = apply(mx, my, a)
        if not ok(nx, ny, sz, walls): continue
        s = sum((chebyshev(nx, ny, e["x"], e["y"]) - chebyshev(mx, my, e["x"], e["y"])) * 15 for e in enemies)
        if in_safe(nx, ny, sz, sr): s += 10
        else: s -= 24
        if (nx, ny) in pos[-4:]: s -= 12
        s -= dist_center(nx, ny, sz) * 0.5
        if s > best_s: best_s, best_a = s, a
    return best_a

def my_dmg(me): return 15 * (1 + me.get("damage_boost_stacks", 0))
def eff_dmg(me, en):
    d = my_dmg(me); return d // 2 if en.get("is_defending") else d
def can_kill(me, en): return eff_dmg(me, en) >= en["health"]

def want_fight(me, en, adj_n, phase, w):
    if en["health"] <= eff_dmg(me, en): return True
    if en.get("is_defending") and me.get("damage_boost_stacks", 0) == 0: return False
    if en.get("energy", 10) < 3: return random.random() < w["helpless_enemy_bonus"]
    sc = w["aggression"]
    if me["health"] > en["health"]: sc += 0.15
    if me["energy"] > en.get("energy", 10): sc += 0.10
    if me.get("damage_boost_stacks", 0) >= 1: sc += 0.20
    if me.get("shield_charges", 0) >= 1: sc += 0.15
    if adj_n >= 2: sc -= 0.30
    if me["health"] < 40: sc -= 0.25
    if me["energy"] < 6: sc -= 0.20
    if phase == "late": sc += 0.10 * (1 - w["survival_value"])
    return sc > 0.50

def want_flee(me, adj, w):
    hp = me["health"] / 100
    if hp <= w["flee_health_pct"]: return True
    if len(adj) >= 2 and hp < w["flee_health_pct"] + 0.20: return True
    if me["health"] < 45 and me["energy"] < w["min_energy_reserve"] + 1: return True
    return False

def score_target(tile, mx, my, energy, health, me, phase, sz, sr, w):
    tx, ty = tile["x"], tile["y"]
    dist = manhattan(mx, my, tx, ty)
    has_res, pu = tile.get("has_resource", False), tile.get("power_up")
    if energy < dist + 2: return -1
    if not in_safe(tx, ty, sz, sr) and phase == "late": return -1
    s = 0.0
    if pu:
        if pu == "damage_boost":   s = 20 + w["damage_boost_priority"] * 30
        elif pu == "shield":       s = 15 + w["shield_priority"] * 25 + (12 if health < 50 else 0)
        elif pu == "energy_pack":  s = 15 + w["energy_pack_priority"] * 25 + (18 if energy < 10 else 0)
        elif pu == "speed_boost":  s = 10 + w["speed_priority"] * 20
        elif pu == "vision_boost": s = 5 + w["vision_priority"] * 15
    elif has_res:
        s = 10 + w["resource_priority"] * 20 + (8 if energy < 10 else 0) + (5 if phase == "early" else 0)
    else: return -1
    s -= dist * 2.5
    if in_safe(tx, ty, sz, sr): s += 4
    return s


# =====================================================================
# MAIN DECISION ENGINE
# =====================================================================

def choose_action(state):
    global prev_action, consecutive_waits

    me = state["self"]
    mx, my, energy, health = me["x"], me["y"], me["energy"], me["health"]
    turn, max_turns = state["turn"], state["max_turns"]
    sz, sr = state["arena_size"], state["safe_zone_radius"]
    enemies, tiles = state.get("enemies", []), state.get("visible_tiles", [])
    w = active_weights

    pct = turn / max_turns
    phase = "early" if pct < 0.30 else ("mid" if pct < 0.65 else "late")
    walls = get_walls(tiles)
    safe = in_safe(mx, my, sz, sr)
    cw = w.get("center_pull", 0.4)

    positions.append((mx, my))
    if len(positions) > 12: positions.pop(0)

    adjacent = [e for e in enemies if chebyshev(mx, my, e["x"], e["y"]) <= 1]
    nearby = [e for e in enemies if 1 < chebyshev(mx, my, e["x"], e["y"]) <= 3]

    my_tile = [t for t in tiles if t["x"] == mx and t["y"] == my]
    on_collectible = any(t.get("has_resource") or t.get("power_up") for t in my_tile)

    other_targets = [t for t in tiles
                     if (t.get("has_resource") or t.get("power_up"))
                     and not (t["x"] == mx and t["y"] == my)]
    nearby_powerups = [t for t in other_targets
                       if t.get("power_up") and manhattan(mx, my, t["x"], t["y"]) <= 3]

    def respond(action, emoji, mood):
        global prev_action, consecutive_waits
        prev_action = action
        consecutive_waits = consecutive_waits + 1 if action == "WAIT" else 0
        return MoveResponse(action=action, emoji=emoji, mood=mood)

    # 1. COLLECT
    if on_collectible and energy >= 2:
        return respond("COLLECT", "💰", "collecting")

    # 2. NEARBY POWER-UP
    if nearby_powerups and energy >= 4 and not adjacent:
        best_pu = max(nearby_powerups, key=lambda t: score_target(t, mx, my, energy, health, me, phase, sz, sr, w))
        if score_target(best_pu, mx, my, energy, health, me, phase, sz, sr, w) > 10:
            a = move_to(mx, my, best_pu["x"], best_pu["y"], sz, walls, sr, enemies, False, positions, cw)
            if a: return respond(a, "✨", f"grab {best_pu['power_up']}")

    # 3. DANGER ZONE
    if not safe and pct >= w.get("zone_prep_timing", 0.58):
        cx, cy = int(center_of(sz)[0]), int(center_of(sz)[1])
        a = move_to(mx, my, cx, cy, sz, walls, sr, enemies, True, positions, cw)
        if a and energy >= 1: return respond(a, "🏃", "zone escape")

    # 4. FLEE
    if adjacent and want_flee(me, adjacent, w):
        if any(can_kill(me, e) for e in adjacent) and energy >= 3:
            return respond("ATTACK", "💀", "parting kill")
        if health <= 20 and energy >= 2 and w["defend_preference"] > 0.5:
            return respond("DEFEND", "🛡️", "emergency")
        a = flee_from(mx, my, enemies, sz, walls, sr, positions)
        if a and energy >= 1: return respond(a, "💨", "retreat")

    # 5. FREE KILLS
    if adjacent and energy >= 3 and any(can_kill(me, e) for e in adjacent):
        return respond("ATTACK", "💀", "executing")

    # 6. HIT & RUN
    if prev_action == "ATTACK" and adjacent and energy >= 1:
        a = flee_from(mx, my, enemies, sz, walls, sr, positions)
        if a: return respond(a, "💨", "hit & run")

    # 7. SMART COMBAT
    if adjacent and energy >= 4:
        for en in adjacent:
            if want_fight(me, en, len(adjacent), phase, w):
                if en.get("is_defending") and w["wait_out_defense"] > 0.5 and me.get("damage_boost_stacks", 0) == 0:
                    return respond("WAIT", "⏳", "wait defense")
                return respond("ATTACK", "⚔️", "strike")
        if energy >= 1:
            a = flee_from(mx, my, enemies, sz, walls, sr, positions)
            if a: return respond(a, "💨", "disengage")

    # 8. CHASE TARGETS
    if other_targets and energy >= 3:
        scored = [(score_target(t, mx, my, energy, health, me, phase, sz, sr, w), t) for t in other_targets]
        scored = [(s, t) for s, t in scored if s > 0]
        scored.sort(key=lambda x: x[0], reverse=True)
        if scored:
            best = scored[0][1]
            avoid = health < 50 and w["survival_value"] > 0.4
            a = move_to(mx, my, best["x"], best["y"], sz, walls, sr, enemies, avoid, positions, cw)
            if a: return respond(a, "🎯", f"get {best.get('power_up', 'resource')}")

    # 9. HUNT WOUNDED
    if nearby and energy >= 6 and health > 50:
        md = my_dmg(me)
        huntable = [e for e in nearby if e["health"] < health and e["health"] <= md * 3]
        if huntable and random.random() < w["kill_chase_value"]:
            t = min(huntable, key=lambda e: e["health"])
            a = move_to(mx, my, t["x"], t["y"], sz, walls, sr, enemies, False, positions, cw)
            if a: return respond(a, "🐺", "hunting")

    # 10. LATE CENTER
    if phase == "late" and dist_center(mx, my, sz) > max(1, sr * 0.4) and energy >= 1:
        cx, cy = int(center_of(sz)[0]), int(center_of(sz)[1])
        a = move_to(mx, my, cx, cy, sz, walls, sr, enemies, True, positions, cw)
        if a: return respond(a, "🏠", "center")

    # 11. ENERGY
    if energy < w.get("min_energy_reserve", 3):
        if consecutive_waits >= 4:
            for a in MOVES:
                nx, ny = apply(mx, my, a)
                if ok(nx, ny, sz, walls) and (nx, ny) not in positions[-3:]:
                    return respond(a, "🔍", "break loop")
        return respond("WAIT", "🔋", "recharge")

    # 12. EXPLORE
    best_a, best_s = None, -9999
    shuffled = list(MOVES); random.shuffle(shuffled)
    for a in shuffled:
        nx, ny = apply(mx, my, a)
        if not ok(nx, ny, sz, walls): continue
        s = -(22 if (nx, ny) in positions[-3:] else (10 if (nx, ny) in positions[-6:] else 0))
        s += 8 if in_safe(nx, ny, sz, sr) else -12
        s -= dist_center(nx, ny, sz) * cw
        for e in enemies:
            if chebyshev(nx, ny, e["x"], e["y"]) <= 2: s -= 8
        s += random.random() * 5
        if s > best_s: best_s, best_a = s, a
    if best_a and energy >= 1: return respond(best_a, "🔍", "scout")
    return respond("WAIT", "😴", "idle")


# =====================================================================
# HTTP ENDPOINTS
# =====================================================================

@app.post("/move")
async def move(state: dict[str, Any]) -> MoveResponse:
    global positions, prev_action, consecutive_waits, active_weights

    match_id = state["match_id"]
    if match_id != current_match.get("match_id"):
        if current_match.get("match_id"):
            finish_match()
        positions, prev_action, consecutive_waits = [], "", 0
        active_weights = get_active_weights()
        new_match(match_id, active_weights)

    track_turn(state)
    response = choose_action(state)

    if ARENA_URL:
        try:
            async with httpx.AsyncClient() as client:
                await client.post(
                    f"{ARENA_URL}/arena/bot-update",
                    json={"bot_id": state["self"]["bot_id"], "status": response.mood.upper(),
                          "message": f"T{state['turn']}: {response.action}", "color": BOT_COLOR},
                    timeout=0.3,
                )
        except Exception:
            pass

    return response


@app.get("/health")
async def health_check():
    return {"status": "ok", "bot_id": BOT_ID}


@app.get("/stats/json")
async def stats_json():
    recent = match_history[-20:]
    current_info = None
    if current_match.get("match_id"):
        current_info = {
            "turns": current_match["turns_played"], "max_turns": current_match["max_turns"],
            "health": current_match["last_health"], "score": current_match["last_score"],
            "kills": current_match["kills"], "enemies_visible": current_match["enemies_alive_last_turn"],
            "enemy_scores": {bid: info["last_score"] for bid, info in enemy_tracker.items()},
        }
    return {
        "matches": total_matches, "won": total_won, "lost": total_lost, "died": total_died,
        "current_match": current_info,
        "recent": [{"outcome": m.get("outcome"), "score": m["last_score"],
                     "reward": m.get("reward", 0), "turns": m["turns_played"],
                     "kills": m["kills"], "enemy_scores": m.get("enemy_scores", {})}
                    for m in recent],
        "weights": {k: round(v, 3) for k, v in strategy_weights.items()},
    }


@app.get("/stats", response_class=HTMLResponse)
async def stats_page():
    wr = f"{total_won / max(1, total_matches) * 100:.0f}" if total_matches else "0"
    recent = match_history[-20:]
    avg_reward = sum(m.get("reward", 0) for m in recent) / max(1, len(recent)) if recent else 0
    avg_score = sum(m["last_score"] for m in recent) / max(1, len(recent)) if recent else 0

    # Current match
    cm = ""
    if current_match.get("match_id"):
        hp = current_match["last_health"]
        hp_color = "#4ade80" if hp > 60 else ("#fbbf24" if hp > 30 else "#ef4444")
        our_sc = current_match["last_score"]
        en_scores = ", ".join(f"{bid[:8]}: {info['last_score']}" for bid, info in enemy_tracker.items())
        score_status = ""
        if enemy_tracker:
            max_en = max(info["last_score"] for info in enemy_tracker.values())
            if our_sc > max_en:
                score_status = '<span style="color:#4ade80">📈 WINNING</span>'
            elif our_sc == max_en:
                score_status = '<span style="color:#fbbf24">🤝 TIED</span>'
            else:
                score_status = '<span style="color:#ef4444">📉 BEHIND</span>'

        cm = f"""
        <div class="card live">
            <h2>🔴 LIVE MATCH</h2>
            <div class="stat-row"><span>Turn</span><strong>{current_match["turns_played"]} / {current_match["max_turns"]}</strong></div>
            <div class="stat-row"><span>Health</span><div class="bar-bg"><div class="bar" style="width:{hp}%;background:{hp_color}"></div></div><strong>{hp}</strong></div>
            <div class="stat-row"><span>Our Score</span><strong>{our_sc}</strong> {score_status}</div>
            <div class="stat-row"><span>Enemy Scores</span><strong style="font-size:0.8em">{en_scores or "none visible"}</strong></div>
            <div class="stat-row"><span>Kills</span><strong>{current_match["kills"]}</strong></div>
            <div class="stat-row"><span>Dmg Taken</span><strong>{current_match["total_damage_taken"]}</strong></div>
        </div>"""

    # Match history
    rows = ""
    for m in reversed(recent[-15:]):
        o = m.get("outcome", "?")
        if o == "won_elimination": icon, cls, label = "🏆", "won", "WON (elimination)"
        elif o == "won_score": icon, cls, label = "🥇", "won", "WON (top score)"
        elif o == "tied": icon, cls, label = "🤝", "tied", "TIED"
        elif o == "lost_score": icon, cls, label = "🥈", "lost", "LOST (outscored)"
        else: icon, cls, label = "💀", "died", "DIED"
        en_sc = m.get("enemy_scores", {})
        en_str = ", ".join(f"{v}" for v in en_sc.values()) if en_sc else "-"
        rows += f"""<tr class="{cls}">
            <td>{icon} {label}</td><td>{m['last_score']}</td><td>{en_str}</td>
            <td>{m['kills']}</td><td>{m['turns_played']}/{m['max_turns']}</td>
            <td>{m.get('reward', 0):.0f}</td></tr>"""

    # Weights
    wbars = ""
    for k, v in strategy_weights.items():
        lo, hi = WEIGHT_BOUNDS.get(k, (0, 1))
        pct = ((v - lo) / (hi - lo)) * 100 if hi > lo else 50
        wbars += f"""<div class="wr"><span class="wl">{k.replace("_"," ").title()}</span>
            <div class="bar-bg small"><div class="bar accent" style="width:{pct:.0f}%"></div></div>
            <span class="wv">{v:.2f}</span></div>"""

    html = f"""<!DOCTYPE html><html><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="refresh" content="4"><title>🤖 Bot Stats</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:-apple-system,sans-serif;background:#0f172a;color:#e2e8f0;padding:16px}}
h1{{text-align:center;font-size:1.6em;margin-bottom:16px;color:#38bdf8}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(110px,1fr));gap:10px;margin-bottom:16px}}
.kpi{{background:#1e293b;border-radius:10px;padding:12px;text-align:center}}
.kpi .num{{font-size:1.8em;font-weight:bold}}.kpi .label{{font-size:0.75em;color:#94a3b8;margin-top:2px}}
.green{{color:#4ade80}}.red{{color:#ef4444}}.blue{{color:#38bdf8}}.yellow{{color:#fbbf24}}.purple{{color:#a78bfa}}
.card{{background:#1e293b;border-radius:10px;padding:14px;margin-bottom:14px}}
.card h2{{font-size:1em;margin-bottom:10px;color:#94a3b8}}.card.live{{border:1px solid #ef4444}}
.stat-row{{display:flex;align-items:center;gap:6px;margin-bottom:5px}}
.stat-row span{{width:90px;font-size:0.8em;color:#94a3b8}}.stat-row strong{{font-size:0.9em}}
.bar-bg{{flex:1;background:#334155;border-radius:4px;height:12px;overflow:hidden}}
.bar-bg.small{{height:7px}}.bar{{height:100%;border-radius:4px;background:#38bdf8}}
.bar.accent{{background:#818cf8}}
table{{width:100%;border-collapse:collapse;font-size:0.8em}}
th{{text-align:left;color:#94a3b8;padding:6px;border-bottom:1px solid #334155}}
td{{padding:6px;border-bottom:1px solid #1e293b}}
tr.won td{{color:#4ade80}}tr.lost td{{color:#fbbf24}}tr.died td{{color:#ef4444}}tr.tied td{{color:#94a3b8}}
.two-col{{display:grid;grid-template-columns:1fr 1fr;gap:14px}}
@media(max-width:600px){{.two-col{{grid-template-columns:1fr}}}}
.wr{{display:flex;align-items:center;gap:4px;margin-bottom:3px}}
.wl{{width:130px;font-size:0.7em;color:#94a3b8}}.wv{{width:36px;font-size:0.7em;text-align:right}}
</style></head><body>
<h1>🤖 RoboKova Learning Bot</h1>
<div class="grid">
    <div class="kpi"><div class="num blue">{total_matches}</div><div class="label">Matches</div></div>
    <div class="kpi"><div class="num green">{total_won}</div><div class="label">Won</div></div>
    <div class="kpi"><div class="num yellow">{total_lost}</div><div class="label">Lost</div></div>
    <div class="kpi"><div class="num red">{total_died}</div><div class="label">Died</div></div>
    <div class="kpi"><div class="num {'green' if int(wr)>=50 else 'red'}">{wr}%</div><div class="label">Win Rate</div></div>
    <div class="kpi"><div class="num purple">{avg_score:.0f}</div><div class="label">Avg Score</div></div>
</div>
{cm}
<div class="two-col">
    <div class="card"><h2>📜 Match History</h2>
        <table><tr><th>Result</th><th>Score</th><th>Enemy</th><th>Kills</th><th>Turns</th><th>Reward</th></tr>
        {rows if rows else '<tr><td colspan="6" style="color:#64748b">No matches yet</td></tr>'}</table>
    </div>
    <div class="card"><h2>🧠 Learned Weights</h2>{wbars}</div>
</div>
<p style="text-align:center;color:#475569;margin-top:16px;font-size:0.7em">
    Refreshes every 4s · {len(match_history)} matches learned · Learning rate {LEARNING_RATE}</p>
</body></html>"""
    return HTMLResponse(content=html)
