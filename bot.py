"""RoboKova Ultimate Bot — Exploits Every Mechanic

KEY INSIGHTS THIS BOT USES:
- COLLECT is the best action: costs 2 energy, returns 3 = NET +1 energy AND +10 score
- Power-ups auto-collect on walk — route through them for free buffs
- Damage boost stacking is OP: 2 stacks = 45 dmg/hit, farm boosts BEFORE fighting
- Enemy last_action tells you what they'll likely do next
- Enemy energy < 3 means they CANNOT attack you — free hits
- Defending enemy = halved damage, WAIT a turn then hit
- Farming resources > fighting for score (unless you can get kills)
- Placement bonus is huge in big lobbies — survival matters most

Run: uvicorn bot:app --host 0.0.0.0 --port 5001 --reload
"""

from __future__ import annotations

import os
import random
from typing import Any

import httpx
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="RoboKova Ultimate Bot")

BOT_ID = os.environ.get("BOT_ID", "ultimate")
BOT_COLOR = os.environ.get("BOT_COLOR", "#00d2ff")
ARENA_URL = os.environ.get("ARENA_URL", "")

# Per-match state tracking
match_state: dict[str, Any] = {
    "last_action": "",
    "positions": [],          # anti-oscillation
    "last_match": "",         # reset state between matches
    "consecutive_waits": 0,   # avoid getting stuck waiting
}


class MoveResponse(BaseModel):
    action: str
    emoji: str = ""
    mood: str = ""


# ===================================================================
# UTILITY FUNCTIONS
# ===================================================================

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


def is_valid_pos(x: int, y: int, arena_size: int, walls: set) -> bool:
    return 0 <= x < arena_size and 0 <= y < arena_size and (x, y) not in walls


ALL_MOVES = ["MOVE_UP", "MOVE_DOWN", "MOVE_LEFT", "MOVE_RIGHT"]


def get_walls(tiles: list[dict]) -> set[tuple[int, int]]:
    return {(t["x"], t["y"]) for t in tiles if t.get("type") == "wall"}


def get_enemy_positions(enemies: list[dict]) -> set[tuple[int, int]]:
    return {(e["x"], e["y"]) for e in enemies}


# ===================================================================
# SMART MOVEMENT — avoids walls, enemies, oscillation, danger zone
# ===================================================================

def score_move(nx: int, ny: int, target_x: int, target_y: int,
               arena_size: int, safe_radius: int, enemies: list[dict],
               avoid_enemies: bool, positions: list) -> float:
    """Score a potential move position. Higher = better."""
    score = 0.0

    # Distance to target (closer = better)
    old_dist = manhattan(target_x, target_y, target_x, target_y)  # irrelevant
    new_dist = manhattan(nx, ny, target_x, target_y)
    score -= new_dist * 3

    # Stay in safe zone
    if is_in_safe_zone(nx, ny, arena_size, safe_radius):
        score += 10
    else:
        score -= 20

    # Avoid enemies if requested
    if avoid_enemies:
        for e in enemies:
            d = chebyshev(nx, ny, e["x"], e["y"])
            if d <= 1:
                score -= 30  # very bad — adjacent
            elif d <= 2:
                score -= 10

    # Anti-oscillation
    if (nx, ny) in positions[-4:]:
        score -= 15

    # Slight center preference
    score -= dist_from_center(nx, ny, arena_size) * 0.5

    return score


def best_move_toward(my_x: int, my_y: int, tx: int, ty: int,
                     arena_size: int, walls: set, safe_radius: int,
                     enemies: list[dict], avoid_enemies: bool = False) -> str | None:
    """Find best move toward target considering all factors."""
    best_action = None
    best_score = -9999

    for action in ALL_MOVES:
        nx, ny = apply_move(my_x, my_y, action)
        if not is_valid_pos(nx, ny, arena_size, walls):
            continue

        score = score_move(nx, ny, tx, ty, arena_size, safe_radius,
                           enemies, avoid_enemies, match_state["positions"])

        # Bonus for actually getting closer to target
        old_dist = manhattan(my_x, my_y, tx, ty)
        new_dist = manhattan(nx, ny, tx, ty)
        score += (old_dist - new_dist) * 5

        if score > best_score:
            best_score = score
            best_action = action

    return best_action


def best_flee_move(my_x: int, my_y: int, enemies: list[dict],
                   arena_size: int, walls: set, safe_radius: int) -> str | None:
    """Find best escape direction — maximize distance from all enemies."""
    best_action = None
    best_score = -9999

    for action in ALL_MOVES:
        nx, ny = apply_move(my_x, my_y, action)
        if not is_valid_pos(nx, ny, arena_size, walls):
            continue

        score = 0.0
        for e in enemies:
            old_d = chebyshev(my_x, my_y, e["x"], e["y"])
            new_d = chebyshev(nx, ny, e["x"], e["y"])
            score += (new_d - old_d) * 15

        if is_in_safe_zone(nx, ny, arena_size, safe_radius):
            score += 10
        else:
            score -= 25

        if (nx, ny) in match_state["positions"][-4:]:
            score -= 12

        score -= dist_from_center(nx, ny, arena_size)

        if score > best_score:
            best_score = score
            best_action = action

    return best_action


def move_to_center(my_x: int, my_y: int, arena_size: int, walls: set,
                   safe_radius: int, enemies: list[dict]) -> str | None:
    cx, cy = int(center_of(arena_size)[0]), int(center_of(arena_size)[1])
    return best_move_toward(my_x, my_y, cx, cy, arena_size, walls,
                            safe_radius, enemies, avoid_enemies=True)


# ===================================================================
# GAME PHASE
# ===================================================================

def get_phase(turn: int, max_turns: int) -> str:
    pct = turn / max_turns
    if pct < 0.30:
        return "early"
    elif pct < 0.65:
        return "mid"
    return "late"


def zone_is_shrinking(turn: int, max_turns: int) -> bool:
    return turn >= max_turns * 0.7


def zone_about_to_shrink(turn: int, max_turns: int) -> bool:
    """Returns True when we should start moving to center."""
    return turn >= max_turns * 0.55


# ===================================================================
# COMBAT INTELLIGENCE
# ===================================================================

def my_attack_damage(me: dict) -> int:
    """Calculate our actual attack damage with buffs."""
    stacks = me.get("damage_boost_stacks", 0)
    return 15 * (1 + stacks)


def effective_damage_to(me: dict, enemy: dict) -> int:
    """Damage we'd actually deal to this specific enemy."""
    dmg = my_attack_damage(me)
    if enemy.get("is_defending"):
        dmg = dmg // 2
    return dmg


def hits_to_kill(me: dict, enemy: dict) -> int:
    """How many attacks to eliminate this enemy."""
    dmg = effective_damage_to(me, enemy)
    if dmg <= 0:
        return 999
    return max(1, -(-enemy["health"] // dmg))  # ceiling division


def energy_to_kill(me: dict, enemy: dict) -> int:
    """Energy needed to kill this enemy (attacks only)."""
    return hits_to_kill(me, enemy) * 3


def can_one_shot(me: dict, enemy: dict) -> bool:
    return effective_damage_to(me, enemy) >= enemy["health"]


def enemy_can_attack(enemy: dict) -> bool:
    return enemy.get("energy", 0) >= 3


def enemy_is_helpless(enemy: dict) -> bool:
    """Enemy can't attack AND can't flee effectively."""
    return enemy.get("energy", 0) < 1


def enemy_just_defended(enemy: dict) -> bool:
    return enemy.get("last_action") == "DEFEND"


def should_attack_enemy(me: dict, enemy: dict, adjacent_count: int, phase: str) -> bool:
    """Smart combat decision — should we attack this enemy?"""
    my_hp = me["health"]
    my_energy = me["energy"]
    my_dmg = effective_damage_to(me, enemy)
    en_hp = enemy["health"]

    # ALWAYS finish off one-shottable enemies
    if en_hp <= my_dmg:
        return True

    # DON'T attack defending enemies without damage boost (waste of energy)
    if enemy.get("is_defending") and me.get("damage_boost_stacks", 0) == 0:
        return False

    # Enemy can't attack — free hits!
    if not enemy_can_attack(enemy):
        return True

    # We have significant damage boost — press the advantage
    if me.get("damage_boost_stacks", 0) >= 2:
        return True

    # We have shields — can absorb hits
    if me.get("shield_charges", 0) >= 1 and my_energy >= 6:
        return True

    # Outnumbered — don't engage
    if adjacent_count >= 2:
        return False

    # We're healthier and have more energy — favorable fight
    if my_hp > en_hp and my_energy > enemy.get("energy", 0) and my_energy >= 6:
        return True

    # Late game and we need points — be more aggressive
    if phase == "late" and my_hp > 40 and my_energy >= 6:
        return True

    # Default: don't fight
    return False


def should_flee(me: dict, adjacent: list[dict], phase: str) -> bool:
    """Should we run away?"""
    hp = me["health"]
    energy = me["energy"]

    # Critical health — always flee
    if hp <= 15:
        return True

    # Low health and no shields
    if hp <= 30 and me.get("shield_charges", 0) == 0:
        return True

    # Outnumbered with moderate health
    if len(adjacent) >= 2 and hp < 60:
        return True

    # Low health AND low energy — can't fight back
    if hp < 45 and energy < 4:
        return True

    # All adjacent enemies have higher HP and can attack
    if all(e["health"] > hp and enemy_can_attack(e) for e in adjacent):
        if me.get("damage_boost_stacks", 0) == 0:
            return True

    return False


# ===================================================================
# TARGET SCORING — what's worth going after
# ===================================================================

def score_collectible(tile: dict, my_x: int, my_y: int, energy: int,
                      health: int, me: dict, phase: str,
                      arena_size: int, safe_radius: int) -> float:
    """Score a tile with resource or power-up. Higher = go get it."""
    tx, ty = tile["x"], tile["y"]
    dist = manhattan(my_x, my_y, tx, ty)

    # Energy budget: can we afford the round trip?
    if tile.get("has_resource"):
        # Need: dist moves (1 each) + 1 COLLECT (2 energy) = dist + 2
        # But COLLECT returns 3, so net cost = dist + 2 - 3 = dist - 1
        if energy < dist + 2:
            return -1
    elif tile.get("power_up"):
        # Power-ups auto-collect on walk, just need to move there
        if energy < dist:
            return -1
    else:
        return -1

    # Don't go outside safe zone for anything in late game
    if not is_in_safe_zone(tx, ty, arena_size, safe_radius):
        if zone_about_to_shrink(0, 1):  # checked elsewhere; penalize here
            return -1

    score = 0.0

    if tile.get("power_up"):
        pu = tile["power_up"]
        if pu == "damage_boost":
            score = 40  # ALWAYS worth it — stacking is OP
            score += me.get("damage_boost_stacks", 0) * 5  # more stacks = even better
        elif pu == "shield":
            score = 35
            if health < 50:
                score += 15
        elif pu == "energy_pack":
            score = 30
            if energy < 10:
                score += 20  # critical when low
        elif pu == "speed_boost":
            score = 22
        elif pu == "vision_boost":
            score = 12
    elif tile.get("has_resource"):
        # Resources are incredible: +10 score, net +1 energy
        score = 20
        if energy < 10:
            score += 10  # energy refund matters more when low
        if phase == "early":
            score += 5  # farming is king early

    # Closer targets are better (distance penalty)
    score -= dist * 3

    # Prefer targets in safe zone
    if is_in_safe_zone(tx, ty, arena_size, safe_radius):
        score += 5

    return score


# ===================================================================
# EXPLORATION — smart wandering when nothing else to do
# ===================================================================

def explore_move(my_x: int, my_y: int, arena_size: int, walls: set,
                 safe_radius: int, enemies: list[dict]) -> str | None:
    """Wander intelligently — avoid oscillation, prefer safe zone, vary direction."""
    candidates = []

    for action in ALL_MOVES:
        nx, ny = apply_move(my_x, my_y, action)
        if not is_valid_pos(nx, ny, arena_size, walls):
            continue

        score = 0.0
        # Avoid recent positions heavily
        positions = match_state["positions"]
        if (nx, ny) in positions[-3:]:
            score -= 20
        if (nx, ny) in positions[-6:]:
            score -= 10

        # Stay in safe zone
        if is_in_safe_zone(nx, ny, arena_size, safe_radius):
            score += 8
        else:
            score -= 15

        # Slight center pull
        score -= dist_from_center(nx, ny, arena_size) * 0.5

        # Avoid enemies while exploring
        for e in enemies:
            if chebyshev(nx, ny, e["x"], e["y"]) <= 2:
                score -= 10

        # Randomness to avoid patterns
        score += random.random() * 5

        candidates.append((score, action))

    if candidates:
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]
    return None


# ===================================================================
# MAIN DECISION ENGINE
# ===================================================================

def choose_action(state: dict[str, Any]) -> MoveResponse:
    global match_state

    # Reset state for new match
    if state["match_id"] != match_state.get("last_match"):
        match_state = {
            "last_action": "",
            "positions": [],
            "last_match": state["match_id"],
            "consecutive_waits": 0,
        }

    me = state["self"]
    my_x, my_y = me["x"], me["y"]
    energy = me["energy"]
    health = me["health"]
    score = me["score"]
    turn = state["turn"]
    max_turns = state["max_turns"]
    arena_size = state["arena_size"]
    safe_radius = state["safe_zone_radius"]
    num_bots = state.get("num_bots", 2)
    enemies = state.get("enemies", [])
    tiles = state.get("visible_tiles", [])
    dmg_stacks = me.get("damage_boost_stacks", 0)
    shields = me.get("shield_charges", 0)
    speed = me.get("speed_boost_turns", 0)

    phase = get_phase(turn, max_turns)
    walls = get_walls(tiles)
    enemy_pos = get_enemy_positions(enemies)
    in_safe = is_in_safe_zone(my_x, my_y, arena_size, safe_radius)

    # Track positions
    match_state["positions"].append((my_x, my_y))
    if len(match_state["positions"]) > 12:
        match_state["positions"].pop(0)

    prev_action = match_state["last_action"]

    # Categorize enemies by distance
    adjacent = [e for e in enemies if chebyshev(my_x, my_y, e["x"], e["y"]) <= 1]
    nearby = [e for e in enemies if 1 < chebyshev(my_x, my_y, e["x"], e["y"]) <= 3]
    all_close = adjacent + nearby

    # Find collectibles
    resources = [t for t in tiles if t.get("has_resource") and not (t["x"] == my_x and t["y"] == my_y)]
    power_ups = [t for t in tiles if t.get("power_up") and not (t["x"] == my_x and t["y"] == my_y)]
    on_resource = any(
        t.get("has_resource") and t["x"] == my_x and t["y"] == my_y for t in tiles
    )
    on_powerup = any(
        t.get("power_up") and t["x"] == my_x and t["y"] == my_y for t in tiles
    )
    all_targets = resources + power_ups

    my_dmg = my_attack_damage(me)

    def respond(action: str, emoji: str, mood: str) -> MoveResponse:
        match_state["last_action"] = action
        if action == "WAIT":
            match_state["consecutive_waits"] += 1
        else:
            match_state["consecutive_waits"] = 0
        return MoveResponse(action=action, emoji=emoji, mood=mood)

    # ==============================================================
    # 1. ALWAYS COLLECT if standing on resource (it's FREE energy + score!)
    #    This is the single best action in the game: net +1 energy, +10 score
    # ==============================================================
    if on_resource and energy >= 2:
        return respond("COLLECT", "💰", "farming")

    # ==============================================================
    # 2. DANGER ZONE — get to safety immediately
    # ==============================================================
    if not in_safe:
        if zone_is_shrinking(turn, max_turns):
            action = move_to_center(my_x, my_y, arena_size, walls, safe_radius, enemies)
            if action and energy >= 1:
                return respond(action, "🏃", "escaping zone")
        elif zone_about_to_shrink(turn, max_turns):
            # Start moving toward center preemptively
            center_dist = dist_from_center(my_x, my_y, arena_size)
            if center_dist > safe_radius + 2:
                action = move_to_center(my_x, my_y, arena_size, walls, safe_radius, enemies)
                if action and energy >= 1:
                    return respond(action, "🏃", "pre-positioning")

    # ==============================================================
    # 3. EMERGENCY — critical health, run or defend
    # ==============================================================
    if adjacent and should_flee(me, adjacent, phase):
        # Can we one-shot someone on the way out? Free 30 pts!
        killable = [e for e in adjacent if can_one_shot(me, e)]
        if killable and energy >= 3:
            return respond("ATTACK", "💀", "parting gift")

        # Shield up if very low
        if health <= 20 and energy >= 2 and shields == 0:
            return respond("DEFEND", "🛡️", "last stand")

        # RUN
        action = best_flee_move(my_x, my_y, enemies, arena_size, walls, safe_radius)
        if action and energy >= 1:
            return respond(action, "💨", "retreating")

    # ==============================================================
    # 4. FINISH OFF — always take free kills (+30 score!)
    # ==============================================================
    if adjacent and energy >= 3:
        killable = [e for e in adjacent if can_one_shot(me, e)]
        if killable:
            return respond("ATTACK", "💀", "executing")

    # ==============================================================
    # 5. SMART COMBAT — attack only with clear advantage
    # ==============================================================
    if adjacent and energy >= 4:
        for enemy in adjacent:
            if should_attack_enemy(me, enemy, len(adjacent), phase):
                # If we just attacked, flee instead (hit & run)
                if prev_action == "ATTACK" and not can_one_shot(me, enemy):
                    action = best_flee_move(my_x, my_y, enemies, arena_size, walls, safe_radius)
                    if action:
                        return respond(action, "💨", "hit and run")

                # If enemy is defending RIGHT NOW, wait a turn
                if enemy.get("is_defending") and dmg_stacks == 0:
                    return respond("WAIT", "⏳", "waiting out defense")

                return respond("ATTACK", "⚔️", "calculated strike")

    # If adjacent enemies but we chose not to fight — disengage
    if adjacent and energy >= 1:
        if not any(should_attack_enemy(me, e, len(adjacent), phase) for e in adjacent):
            action = best_flee_move(my_x, my_y, enemies, arena_size, walls, safe_radius)
            if action:
                return respond(action, "💨", "disengaging")

    # ==============================================================
    # 6. CHASE BEST TARGET — score and route to best collectible
    # ==============================================================
    if all_targets and energy >= 2:
        scored = []
        for t in all_targets:
            s = score_collectible(t, my_x, my_y, energy, health, me, phase,
                                  arena_size, safe_radius)
            if s > 0:
                scored.append((s, t))
        scored.sort(key=lambda x: x[0], reverse=True)

        if scored:
            best = scored[0][1]
            avoid = health < 50  # avoid enemies when hurt
            action = best_move_toward(my_x, my_y, best["x"], best["y"],
                                      arena_size, walls, safe_radius,
                                      enemies, avoid_enemies=avoid)
            if action:
                label = best.get("power_up", "resource")
                return respond(action, "🎯", f"hunting {label}")

    # ==============================================================
    # 7. HUNT WOUNDED ENEMIES — stalk and finish off
    # ==============================================================
    if phase in ("mid", "late") and energy >= 6 and health > 50:
        # Only hunt if we have damage advantage
        huntable = [e for e in nearby
                    if e["health"] < health
                    and e["health"] <= my_dmg * 3
                    and (not enemy_can_attack(e) or dmg_stacks >= 1)]
        if huntable:
            target = min(huntable, key=lambda e: e["health"])
            action = best_move_toward(my_x, my_y, target["x"], target["y"],
                                      arena_size, walls, safe_radius, enemies)
            if action:
                return respond(action, "🐺", "stalking")

    # ==============================================================
    # 8. LATE GAME POSITIONING — stay near center, stay alive
    # ==============================================================
    if phase == "late":
        center_dist = dist_from_center(my_x, my_y, arena_size)
        if center_dist > max(1, safe_radius * 0.4) and energy >= 1:
            action = move_to_center(my_x, my_y, arena_size, walls, safe_radius, enemies)
            if action:
                return respond(action, "🏠", "centering")

    # ==============================================================
    # 9. ENERGY MANAGEMENT — rest when needed, but not too long
    # ==============================================================
    if energy < 3:
        # Don't get stuck in WAIT loops — if waited 3+ turns, try to move
        if match_state["consecutive_waits"] >= 3:
            action = explore_move(my_x, my_y, arena_size, walls, safe_radius, enemies)
            if action and energy >= 1:
                return respond(action, "🔍", "breaking wait loop")
        return respond("WAIT", "🔋", "recharging")

    # ==============================================================
    # 10. EXPLORE — wander intelligently to find resources
    # ==============================================================
    action = explore_move(my_x, my_y, arena_size, walls, safe_radius, enemies)
    if action and energy >= 1:
        return respond(action, "🔍", "scouting")

    return respond("WAIT", "😴", "nothing to do")


# ===================================================================
# HTTP ENDPOINTS
# ===================================================================

@app.post("/move")
async def move(state: dict[str, Any]) -> MoveResponse:
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
