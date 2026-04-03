"""RoboKova Battle Bot — Hit & Run Strategy

Punches hard, then disappears. Never stays in a prolonged fight.
Prioritizes power-ups (shields, damage boosts) to make each hit devastating.
Only commits to a kill when the enemy is one hit from death.

Run: uvicorn bot:app --host 0.0.0.0 --port 5001 --reload
"""

from __future__ import annotations

import os
import random
from typing import Any

import httpx
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="RoboKova Hit & Run Bot")

BOT_ID = os.environ.get("BOT_ID", "hitrun-bot")
BOT_COLOR = os.environ.get("BOT_COLOR", "#ff2e63")
ARENA_URL = os.environ.get("ARENA_URL", "")

# State tracking across turns
last_action: str = ""
last_positions: list[tuple[int, int]] = []
turns_adjacent_to_enemy: int = 0


class MoveResponse(BaseModel):
    action: str
    emoji: str = ""
    mood: str = ""


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def manhattan(x1: int, y1: int, x2: int, y2: int) -> int:
    return abs(x1 - x2) + abs(y1 - y2)


def chebyshev(x1: int, y1: int, x2: int, y2: int) -> int:
    return max(abs(x1 - x2), abs(y1 - y2))


def dist_from_center(x: int, y: int, arena_size: int) -> float:
    center = (arena_size - 1) / 2
    return max(abs(x - center), abs(y - center))


def is_in_safe_zone(x: int, y: int, arena_size: int, safe_radius: int) -> bool:
    center = (arena_size - 1) / 2
    return abs(x - center) <= safe_radius and abs(y - center) <= safe_radius


def get_wall_set(tiles: list[dict]) -> set[tuple[int, int]]:
    return {(t["x"], t["y"]) for t in tiles if t.get("type") == "wall"}


def get_enemy_positions(enemies: list[dict]) -> set[tuple[int, int]]:
    return {(e["x"], e["y"]) for e in enemies}


def apply_move(x: int, y: int, action: str) -> tuple[int, int]:
    if action == "MOVE_UP":    return (x, y - 1)
    if action == "MOVE_DOWN":  return (x, y + 1)
    if action == "MOVE_LEFT":  return (x - 1, y)
    if action == "MOVE_RIGHT": return (x + 1, y)
    return (x, y)


def is_valid(x: int, y: int, arena_size: int, walls: set) -> bool:
    return 0 <= x < arena_size and 0 <= y < arena_size and (x, y) not in walls


def move_toward(my_x: int, my_y: int, tx: int, ty: int,
                walls: set, arena_size: int, enemy_pos: set | None = None) -> str | None:
    dx = tx - my_x
    dy = ty - my_y
    blocked = walls | (enemy_pos or set())

    candidates = []
    if abs(dx) >= abs(dy):
        if dx > 0:
            candidates = ["MOVE_RIGHT", "MOVE_DOWN" if dy > 0 else "MOVE_UP",
                          "MOVE_UP" if dy > 0 else "MOVE_DOWN", "MOVE_LEFT"]
        elif dx < 0:
            candidates = ["MOVE_LEFT", "MOVE_DOWN" if dy > 0 else "MOVE_UP",
                          "MOVE_UP" if dy > 0 else "MOVE_DOWN", "MOVE_RIGHT"]
        else:
            candidates = ["MOVE_DOWN" if dy > 0 else "MOVE_UP",
                          "MOVE_RIGHT", "MOVE_LEFT"]
    else:
        if dy > 0:
            candidates = ["MOVE_DOWN", "MOVE_RIGHT" if dx > 0 else "MOVE_LEFT",
                          "MOVE_LEFT" if dx > 0 else "MOVE_RIGHT", "MOVE_UP"]
        elif dy < 0:
            candidates = ["MOVE_UP", "MOVE_RIGHT" if dx > 0 else "MOVE_LEFT",
                          "MOVE_LEFT" if dx > 0 else "MOVE_RIGHT", "MOVE_DOWN"]
        else:
            candidates = ["MOVE_RIGHT", "MOVE_LEFT", "MOVE_UP", "MOVE_DOWN"]

    for action in candidates:
        nx, ny = apply_move(my_x, my_y, action)
        if is_valid(nx, ny, arena_size, blocked):
            return action
    return None


def flee_from_enemies(my_x: int, my_y: int, enemies: list[dict],
                      walls: set, arena_size: int, safe_radius: int) -> str | None:
    all_moves = ["MOVE_UP", "MOVE_DOWN", "MOVE_LEFT", "MOVE_RIGHT"]
    center = (arena_size - 1) / 2

    best_action = None
    best_score = -999

    for action in all_moves:
        nx, ny = apply_move(my_x, my_y, action)
        if not is_valid(nx, ny, arena_size, walls):
            continue

        score = 0
        for e in enemies:
            old_dist = chebyshev(my_x, my_y, e["x"], e["y"])
            new_dist = chebyshev(nx, ny, e["x"], e["y"])
            score += (new_dist - old_dist) * 10

        if is_in_safe_zone(nx, ny, arena_size, safe_radius):
            score += 5
        else:
            score -= 10

        center_dist = dist_from_center(nx, ny, arena_size)
        score -= center_dist

        if (nx, ny) in last_positions[-4:]:
            score -= 8

        if score > best_score:
            best_score = score
            best_action = action

    return best_action


def move_toward_center(my_x: int, my_y: int, arena_size: int, walls: set) -> str | None:
    cx = int((arena_size - 1) / 2)
    cy = int((arena_size - 1) / 2)
    return move_toward(my_x, my_y, cx, cy, walls, arena_size)


# ---------------------------------------------------------------------------
# Game phase
# ---------------------------------------------------------------------------

def get_phase(turn: int, max_turns: int) -> str:
    pct = turn / max_turns
    if pct < 0.35:
        return "early"
    elif pct < 0.70:
        return "mid"
    return "late"


# ---------------------------------------------------------------------------
# Target scoring — power-ups are king for hit & run
# ---------------------------------------------------------------------------

POWER_UP_VALUE = {
    "damage_boost": 30,
    "shield": 28,
    "energy_pack": 25,
    "speed_boost": 20,
    "vision_boost": 8,
}


def score_target(tile: dict, my_x: int, my_y: int, energy: int,
                 health: int, phase: str) -> float:
    dist = manhattan(my_x, my_y, tile["x"], tile["y"])
    energy_needed = dist + 2

    if energy < energy_needed:
        return -1

    if tile.get("power_up"):
        base = POWER_UP_VALUE.get(tile["power_up"], 10)
        if tile["power_up"] == "energy_pack" and energy < 10:
            base += 15
        if tile["power_up"] == "shield" and health < 50:
            base += 10
        return base - dist * 2

    if tile.get("has_resource"):
        base = 14
        if energy < 8:
            base += 8
        if phase == "early":
            base += 4
        return base - dist * 2

    return -1


# ---------------------------------------------------------------------------
# Combat helpers
# ---------------------------------------------------------------------------

def calc_my_damage(me: dict) -> int:
    stacks = me.get("damage_boost_stacks", 0)
    return 15 * (1 + stacks)


def can_one_shot(me: dict, enemy: dict) -> bool:
    dmg = calc_my_damage(me)
    if enemy.get("is_defending"):
        dmg //= 2
    return enemy["health"] <= dmg


def is_safe_to_attack(me: dict, enemies: list[dict], adjacent: list[dict]) -> bool:
    if len(adjacent) >= 2 and me["health"] < 60:
        return False
    if me["health"] <= 20:
        return False
    return True


# ---------------------------------------------------------------------------
# Main strategy — HIT & RUN
# ---------------------------------------------------------------------------

def choose_action(state: dict[str, Any]) -> MoveResponse:
    global last_action, last_positions, turns_adjacent_to_enemy

    me = state["self"]
    my_x, my_y = me["x"], me["y"]
    energy = me["energy"]
    health = me["health"]
    turn = state["turn"]
    max_turns = state["max_turns"]
    arena_size = state["arena_size"]
    safe_radius = state["safe_zone_radius"]
    enemies = state.get("enemies", [])
    tiles = state.get("visible_tiles", [])
    shields = me.get("shield_charges", 0)

    phase = get_phase(turn, max_turns)
    walls = get_wall_set(tiles)
    enemy_pos = get_enemy_positions(enemies)
    in_safe = is_in_safe_zone(my_x, my_y, arena_size, safe_radius)

    last_positions.append((my_x, my_y))
    if len(last_positions) > 8:
        last_positions.pop(0)

    adjacent = [e for e in enemies if chebyshev(my_x, my_y, e["x"], e["y"]) <= 1]
    nearby = [e for e in enemies if chebyshev(my_x, my_y, e["x"], e["y"]) <= 3]

    if adjacent:
        turns_adjacent_to_enemy += 1
    else:
        turns_adjacent_to_enemy = 0

    resources = [t for t in tiles if t.get("has_resource")]
    power_ups = [t for t in tiles if t.get("power_up")]
    collectibles = resources + power_ups

    on_resource = any(
        (t.get("has_resource") or t.get("power_up"))
        and t["x"] == my_x and t["y"] == my_y
        for t in tiles
    )

    my_dmg = calc_my_damage(me)

    # === 1: ESCAPE DANGER ZONE ===
    if not in_safe and phase in ("mid", "late"):
        if on_resource and energy >= 2:
            last_action = "COLLECT"
            return MoveResponse(action="COLLECT", emoji="💎", mood="grab and go")
        action = move_toward_center(my_x, my_y, arena_size, walls)
        if action and energy >= 1:
            last_action = action
            return MoveResponse(action=action, emoji="🏃", mood="escaping zone")

    # === 2: JUST ATTACKED? RUN! (core hit & run) ===
    if last_action == "ATTACK" and adjacent and energy >= 1:
        killable = [e for e in adjacent if can_one_shot(me, e)]
        if killable and energy >= 3:
            last_action = "ATTACK"
            return MoveResponse(action="ATTACK", emoji="💀", mood="finishing blow")
        action = flee_from_enemies(my_x, my_y, enemies, walls, arena_size, safe_radius)
        if action:
            last_action = action
            return MoveResponse(action=action, emoji="💨", mood="hit and run!")

    # === 3: EMERGENCY — very low HP ===
    if adjacent and health <= 25:
        if energy >= 2:
            last_action = "DEFEND"
            return MoveResponse(action="DEFEND", emoji="🛡️", mood="emergency shield")
        action = flee_from_enemies(my_x, my_y, enemies, walls, arena_size, safe_radius)
        if action and energy >= 1:
            last_action = action
            return MoveResponse(action=action, emoji="💨", mood="running for life")

    # === 4: FINISH OFF weak enemy ===
    if adjacent and energy >= 3:
        killable = [e for e in adjacent if can_one_shot(me, e)]
        if killable:
            last_action = "ATTACK"
            return MoveResponse(action="ATTACK", emoji="💀", mood="executing")

    # === 5: STUCK IN FIGHT TOO LONG — disengage ===
    if adjacent and turns_adjacent_to_enemy >= 2 and energy >= 1:
        action = flee_from_enemies(my_x, my_y, enemies, walls, arena_size, safe_radius)
        if action:
            last_action = action
            return MoveResponse(action=action, emoji="💨", mood="disengaging")

    # === 6: LAND A HIT (only if safe to escape after) ===
    if adjacent and energy >= 4 and is_safe_to_attack(me, enemies, adjacent):
        weakest = min(adjacent, key=lambda e: e["health"])
        if not (weakest.get("is_defending") and me.get("damage_boost_stacks", 0) == 0):
            last_action = "ATTACK"
            emoji = "🔪" if weakest["health"] < 40 else "👊"
            return MoveResponse(action="ATTACK", emoji=emoji, mood="striking")

    # === 7: COLLECT if standing on something ===
    if on_resource and energy >= 2:
        last_action = "COLLECT"
        return MoveResponse(action="COLLECT", emoji="💎", mood="collecting")

    # === 8: CHASE best target ===
    if collectibles and energy >= 3:
        scored = [(score_target(t, my_x, my_y, energy, health, phase), t)
                  for t in collectibles]
        scored = [(s, t) for s, t in scored if s > 0]
        scored.sort(key=lambda x: x[0], reverse=True)

        if scored:
            best = scored[0][1]
            action = move_toward(my_x, my_y, best["x"], best["y"],
                                 walls, arena_size,
                                 enemy_pos if health < 60 else None)
            if action:
                label = best.get("power_up", "resource")
                last_action = action
                return MoveResponse(action=action, emoji="🎯", mood=f"hunting {label}")

    # === 9: STALK wounded enemies ===
    if phase in ("mid", "late") and energy >= 6 and health > 40:
        huntable = [e for e in nearby if e["health"] <= my_dmg * 2 and e["health"] < health]
        if huntable:
            target = min(huntable, key=lambda e: e["health"])
            action = move_toward(my_x, my_y, target["x"], target["y"],
                                 walls, arena_size)
            if action:
                last_action = action
                return MoveResponse(action=action, emoji="🐺", mood="stalking prey")

    # === 10: LATE GAME — stay near center ===
    if phase == "late" and energy >= 1:
        center_dist = dist_from_center(my_x, my_y, arena_size)
        if center_dist > safe_radius * 0.5:
            action = move_toward_center(my_x, my_y, arena_size, walls)
            if action:
                last_action = action
                return MoveResponse(action=action, emoji="🏠", mood="centering")

    # === 11: REST if low energy ===
    if energy < 4:
        last_action = "WAIT"
        return MoveResponse(action="WAIT", emoji="😴", mood="recharging")

    # === 12: EXPLORE smartly ===
    moves = ["MOVE_UP", "MOVE_DOWN", "MOVE_LEFT", "MOVE_RIGHT"]
    random.shuffle(moves)
    center = (arena_size - 1) / 2
    best_move = None
    best_score = -999

    for action in moves:
        nx, ny = apply_move(my_x, my_y, action)
        if not is_valid(nx, ny, arena_size, walls):
            continue
        score = 0
        score -= dist_from_center(nx, ny, arena_size)
        if (nx, ny) in last_positions[-4:]:
            score -= 15
        if is_in_safe_zone(nx, ny, arena_size, safe_radius):
            score += 5
        score += random.random() * 3
        if score > best_score:
            best_score = score
            best_move = action

    if best_move:
        last_action = best_move
        return MoveResponse(action=best_move, emoji="🔍", mood="scouting")

    last_action = "WAIT"
    return MoveResponse(action="WAIT", emoji="😴", mood="waiting")


# ---------------------------------------------------------------------------
# HTTP endpoints
# ---------------------------------------------------------------------------

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
                        "message": f"Turn {state['turn']}: {response.action}",
                        "color": BOT_COLOR,
                    },
                    timeout=0.5,
                )
        except Exception:
            pass

    return response


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "bot_id": BOT_ID}
