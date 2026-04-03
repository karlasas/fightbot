"""RoboKova Battle Bot — Competitive Balanced Strategy

A smart bot that adapts to game phase, manages energy carefully,
fights when it has the advantage, and survives to the end.

Run: uvicorn bot:app --host 0.0.0.0 --port 5001 --reload
"""

from __future__ import annotations

import os
import random
from typing import Any

import httpx
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="RoboKova Competitor")

BOT_ID = os.environ.get("BOT_ID", "competitor")
BOT_COLOR = os.environ.get("BOT_COLOR", "#e94560")
ARENA_URL = os.environ.get("ARENA_URL", "")

# Track last positions to avoid oscillation
last_positions: list[tuple[int, int]] = []


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


def move_toward(my_x: int, my_y: int, tx: int, ty: int,
                walls: set[tuple[int, int]], arena_size: int) -> str | None:
    """Move toward target, avoiding walls. Returns None if stuck."""
    dx = tx - my_x
    dy = ty - my_y

    # Build candidate moves sorted by preference
    candidates = []
    if abs(dx) >= abs(dy):
        if dx > 0:
            candidates = ["MOVE_RIGHT", "MOVE_DOWN" if dy > 0 else "MOVE_UP",
                          "MOVE_UP" if dy > 0 else "MOVE_DOWN", "MOVE_LEFT"]
        else:
            candidates = ["MOVE_LEFT", "MOVE_DOWN" if dy > 0 else "MOVE_UP",
                          "MOVE_UP" if dy > 0 else "MOVE_DOWN", "MOVE_RIGHT"]
    else:
        if dy > 0:
            candidates = ["MOVE_DOWN", "MOVE_RIGHT" if dx > 0 else "MOVE_LEFT",
                          "MOVE_LEFT" if dx > 0 else "MOVE_RIGHT", "MOVE_UP"]
        else:
            candidates = ["MOVE_UP", "MOVE_RIGHT" if dx > 0 else "MOVE_LEFT",
                          "MOVE_LEFT" if dx > 0 else "MOVE_RIGHT", "MOVE_DOWN"]

    for action in candidates:
        nx, ny = apply_move(my_x, my_y, action)
        if 0 <= nx < arena_size and 0 <= ny < arena_size and (nx, ny) not in walls:
            return action
    return None


def move_away_from(my_x: int, my_y: int, tx: int, ty: int,
                   walls: set[tuple[int, int]], arena_size: int) -> str | None:
    """Move away from target, avoiding walls."""
    dx = my_x - tx  # reversed direction
    dy = my_y - ty
    if dx == 0 and dy == 0:
        dx, dy = 1, 0  # arbitrary

    candidates = []
    if abs(dx) >= abs(dy):
        primary = "MOVE_RIGHT" if dx > 0 else "MOVE_LEFT"
        secondary = "MOVE_DOWN" if dy > 0 else "MOVE_UP"
        candidates = [primary, secondary]
    else:
        primary = "MOVE_DOWN" if dy > 0 else "MOVE_UP"
        secondary = "MOVE_RIGHT" if dx > 0 else "MOVE_LEFT"
        candidates = [primary, secondary]

    # Add perpendicular options
    all_moves = ["MOVE_UP", "MOVE_DOWN", "MOVE_LEFT", "MOVE_RIGHT"]
    for m in all_moves:
        if m not in candidates:
            candidates.append(m)

    for action in candidates:
        nx, ny = apply_move(my_x, my_y, action)
        if 0 <= nx < arena_size and 0 <= ny < arena_size and (nx, ny) not in walls:
            return action
    return None


def apply_move(x: int, y: int, action: str) -> tuple[int, int]:
    if action == "MOVE_UP":    return (x, y - 1)
    if action == "MOVE_DOWN":  return (x, y + 1)
    if action == "MOVE_LEFT":  return (x - 1, y)
    if action == "MOVE_RIGHT": return (x + 1, y)
    return (x, y)


def move_toward_center(my_x: int, my_y: int, arena_size: int,
                       walls: set[tuple[int, int]]) -> str | None:
    cx = int((arena_size - 1) / 2)
    cy = int((arena_size - 1) / 2)
    return move_toward(my_x, my_y, cx, cy, walls, arena_size)


# ---------------------------------------------------------------------------
# Game phase detection
# ---------------------------------------------------------------------------

def get_phase(turn: int, max_turns: int) -> str:
    pct = turn / max_turns
    if pct < 0.35:
        return "early"
    elif pct < 0.70:
        return "mid"
    return "late"


# ---------------------------------------------------------------------------
# Target scoring — decides what's worth going after
# ---------------------------------------------------------------------------

POWER_UP_PRIORITY = {
    "energy_pack": 25,
    "damage_boost": 22,
    "shield": 20,
    "speed_boost": 15,
    "vision_boost": 10,
}


def score_target(tile: dict, my_x: int, my_y: int, energy: int,
                 phase: str) -> float:
    """Score a resource or power-up tile. Higher = more desirable."""
    dist = manhattan(my_x, my_y, tile["x"], tile["y"])
    energy_cost = dist * 1  # movement cost

    # Can't afford the trip + collection
    if energy < energy_cost + 2:
        return -1

    if tile.get("power_up"):
        base = POWER_UP_PRIORITY.get(tile["power_up"], 10)
        # Energy packs are worth more when low on energy
        if tile["power_up"] == "energy_pack" and energy < 8:
            base += 15
        return base - dist * 1.5

    if tile.get("has_resource"):
        base = 12
        # Resources are worth more in early game (energy building)
        if phase == "early":
            base += 5
        return base - dist * 1.5

    return -1


# ---------------------------------------------------------------------------
# Combat decision
# ---------------------------------------------------------------------------

def should_fight(me: dict, enemy: dict, phase: str) -> bool:
    """Decide if we should engage this enemy."""
    my_hp = me["health"]
    my_energy = me["energy"]
    my_stacks = me.get("damage_boost_stacks", 0)
    my_shields = me.get("shield_charges", 0)

    en_hp = enemy["health"]
    en_energy = enemy["energy"]
    en_defending = enemy.get("is_defending", False)

    # Base attack damage considering our boosts
    my_dmg = 15 * (1 + my_stacks)
    if en_defending:
        my_dmg //= 2

    # How many hits to kill them vs. them killing us
    hits_to_kill = max(1, (en_hp + my_dmg - 1) // my_dmg)
    # Assume enemy does 15 base damage to us
    enemy_dmg = 15
    hits_they_need = max(1, (my_hp + enemy_dmg - 1) // enemy_dmg)
    if my_shields > 0:
        hits_they_need += my_shields

    # Energy check — can we sustain the fight?
    attacks_affordable = my_energy // 3

    # Fight if we can likely win
    if hits_to_kill <= attacks_affordable and hits_to_kill < hits_they_need:
        return True

    # Always finish off low-HP enemies
    if en_hp <= my_dmg and my_energy >= 3:
        return True

    # Late game — be more aggressive if we need points
    if phase == "late" and my_hp > 40 and my_energy >= 6:
        return True

    # Enemy is out of energy — free hits
    if en_energy < 3 and my_energy >= 3:
        return True

    return False


def should_flee(me: dict, enemies: list[dict]) -> bool:
    """Decide if we should run away."""
    my_hp = me["health"]
    my_energy = me["energy"]

    if my_hp <= 20:
        return True

    # Multiple enemies nearby — dangerous
    if len(enemies) >= 2 and my_hp < 50:
        return True

    # Low health AND low energy — can't fight
    if my_hp < 40 and my_energy < 4:
        return True

    return False


# ---------------------------------------------------------------------------
# Main strategy
# ---------------------------------------------------------------------------

def choose_action(state: dict[str, Any]) -> MoveResponse:
    global last_positions

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

    phase = get_phase(turn, max_turns)
    walls = get_wall_set(tiles)
    in_safe = is_in_safe_zone(my_x, my_y, arena_size, safe_radius)

    # Track positions for anti-oscillation
    last_positions.append((my_x, my_y))
    if len(last_positions) > 6:
        last_positions.pop(0)

    # Find adjacent enemies (Chebyshev distance 1)
    adjacent = [e for e in enemies if chebyshev(my_x, my_y, e["x"], e["y"]) <= 1]
    nearby = [e for e in enemies if chebyshev(my_x, my_y, e["x"], e["y"]) <= 3]

    # Gather collectible targets
    resources = [t for t in tiles if t.get("has_resource")]
    power_ups = [t for t in tiles if t.get("power_up")]
    collectibles = resources + power_ups

    # Standing on a resource or power-up?
    on_resource = any(
        (t.get("has_resource") or t.get("power_up"))
        and t["x"] == my_x and t["y"] == my_y
        for t in tiles
    )

    # === PRIORITY 1: DANGER ZONE — get to safety! ===
    if not in_safe and phase in ("mid", "late"):
        # Collect if standing on something first
        if on_resource and energy >= 2:
            return MoveResponse(action="COLLECT", emoji="💎", mood="grabbing before running")

        action = move_toward_center(my_x, my_y, arena_size, walls)
        if action and energy >= 1:
            return MoveResponse(action=action, emoji="🏃", mood="fleeing zone")

    # === PRIORITY 2: FLEE if in danger ===
    if adjacent and should_flee(me, adjacent):
        # Defend if very low HP and being attacked
        if health <= 25 and energy >= 2:
            return MoveResponse(action="DEFEND", emoji="🛡️", mood="desperate defense")

        # Run from nearest enemy
        closest = min(adjacent, key=lambda e: chebyshev(my_x, my_y, e["x"], e["y"]))
        action = move_away_from(my_x, my_y, closest["x"], closest["y"], walls, arena_size)
        if action and energy >= 1:
            return MoveResponse(action=action, emoji="💨", mood="retreating")

    # === PRIORITY 3: FINISH OFF weak adjacent enemies ===
    if adjacent and energy >= 3:
        weak = [e for e in adjacent if e["health"] <= 15 * (1 + me.get("damage_boost_stacks", 0))]
        if weak:
            return MoveResponse(action="ATTACK", emoji="💀", mood="finishing off")

    # === PRIORITY 4: COLLECT if standing on something ===
    if on_resource and energy >= 2:
        return MoveResponse(action="COLLECT", emoji="💎", mood="collecting")

    # === PRIORITY 5: FIGHT adjacent enemies (if smart) ===
    if adjacent and energy >= 3:
        for enemy in adjacent:
            if should_fight(me, enemy, phase):
                return MoveResponse(action="ATTACK", emoji="⚔️", mood="fighting")

        # If enemy is adjacent but fight isn't great, defend
        if energy >= 2 and health < 60:
            return MoveResponse(action="DEFEND", emoji="🛡️", mood="guarding")

    # === PRIORITY 6: GO AFTER best target ===
    if collectibles and energy >= 3:
        scored = [(score_target(t, my_x, my_y, energy, phase), t) for t in collectibles]
        scored = [(s, t) for s, t in scored if s > 0]
        scored.sort(key=lambda x: x[0], reverse=True)

        if scored:
            best = scored[0][1]
            action = move_toward(my_x, my_y, best["x"], best["y"], walls, arena_size)
            if action:
                label = best.get("power_up", "resource")
                return MoveResponse(action=action, emoji="🎯", mood=f"hunting {label}")

    # === PRIORITY 7: HUNT weak enemies (mid/late game) ===
    if phase in ("mid", "late") and nearby and energy >= 5 and health > 50:
        weak_enemies = [e for e in nearby if e["health"] < health]
        if weak_enemies:
            target = min(weak_enemies, key=lambda e: e["health"])
            action = move_toward(my_x, my_y, target["x"], target["y"], walls, arena_size)
            if action:
                return MoveResponse(action=action, emoji="🔪", mood="hunting enemy")

    # === PRIORITY 8: MOVE TOWARD CENTER in late game ===
    if phase == "late" and energy >= 1:
        center_dist = dist_from_center(my_x, my_y, arena_size)
        if center_dist > safe_radius * 0.6:
            action = move_toward_center(my_x, my_y, arena_size, walls)
            if action:
                return MoveResponse(action=action, emoji="🏠", mood="centering")

    # === PRIORITY 9: WAIT if low energy ===
    if energy < 3:
        return MoveResponse(action="WAIT", emoji="😴", mood="recharging")

    # === PRIORITY 10: EXPLORE — move randomly but avoid oscillation ===
    moves = ["MOVE_UP", "MOVE_DOWN", "MOVE_LEFT", "MOVE_RIGHT"]
    random.shuffle(moves)

    for action in moves:
        nx, ny = apply_move(my_x, my_y, action)
        if 0 <= nx < arena_size and 0 <= ny < arena_size and (nx, ny) not in walls:
            # Avoid going back to recent positions
            if (nx, ny) not in last_positions[-3:]:
                return MoveResponse(action=action, emoji="🔍", mood="exploring")

    # Fallback — any valid move
    for action in moves:
        nx, ny = apply_move(my_x, my_y, action)
        if 0 <= nx < arena_size and 0 <= ny < arena_size and (nx, ny) not in walls:
            return MoveResponse(action=action, emoji="🤔", mood="wandering")

    return MoveResponse(action="WAIT", emoji="😴", mood="stuck")


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
