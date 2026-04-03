"""RoboKova Q-Learning Bot v10 — Learns From Every Move

ARCHITECTURE:
- Q-Table: maps (situation → action → expected reward) learned from every turn
- 10 features describe each situation → ~7000 possible states
- 9 abstract actions (ATTACK, COLLECT, FLEE, CHASE_ENEMY, etc.)
- Every turn: observe reward, update Q-value for previous action
- Opponent modeling: tracks each enemy's behavior patterns
- Falls back to rules when Q-table lacks data for a state
- Saves Q-table + opponent data to disk periodically

Run: uvicorn bot:app --host 0.0.0.0 --port 5001 --reload
"""

from __future__ import annotations
import json, os, random, time
from collections import deque, defaultdict
from pathlib import Path
from typing import Any
import httpx
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

app = FastAPI(title="RoboKova Q-Learning v10")
BOT_ID = os.environ.get("BOT_ID", "q-learner")
BOT_COLOR = os.environ.get("BOT_COLOR", "#e94560")
ARENA_URL = os.environ.get("ARENA_URL", "")
BRAIN_FILE = Path("brain_v10.json")
SAVE_INTERVAL = 20  # save every N turns across all matches
STALE_TIMEOUT = 60

class MoveResponse(BaseModel):
    action: str
    emoji: str = ""
    mood: str = ""

# =====================================================================
# Q-LEARNING SYSTEM
# =====================================================================

# Abstract actions (translated to concrete moves later)
ACTIONS = ["ATTACK", "COLLECT", "DEFEND", "WAIT", "CHASE_ENEMY",
           "CHASE_TARGET", "FLEE", "GO_CENTER", "EXPLORE"]
NUM_ACTIONS = len(ACTIONS)

# Q-Learning parameters
ALPHA = 0.12       # learning rate
GAMMA = 0.90       # discount factor (value future rewards)
EPSILON_START = 0.25  # initial exploration rate
EPSILON_MIN = 0.05    # minimum exploration
EPSILON_DECAY = 0.9995  # per-turn decay
MIN_VISITS = 3      # fall back to rules below this

# Q-table: state_key → [q_values for each action]
# Visit count: state_key → [visit_count for each action]
q_table: dict[str, list[float]] = {}
visit_count: dict[str, list[int]] = {}
epsilon = EPSILON_START
total_turns_learned = 0

# Opponent patterns: bot_id → behavior stats
opponent_data: dict[str, dict] = {}

def save_brain():
    try:
        data = {
            "q_table": q_table, "visit_count": visit_count,
            "epsilon": epsilon, "total_turns": total_turns_learned,
            "opponents": opponent_data,
        }
        BRAIN_FILE.write_text(json.dumps(data))
    except Exception as e:
        print(f"Save failed: {e}")

def load_brain():
    global q_table, visit_count, epsilon, total_turns_learned, opponent_data
    if BRAIN_FILE.exists():
        try:
            data = json.loads(BRAIN_FILE.read_text())
            q_table = data.get("q_table", {})
            visit_count = data.get("visit_count", {})
            epsilon = data.get("epsilon", EPSILON_START)
            total_turns_learned = data.get("total_turns", 0)
            opponent_data = data.get("opponents", {})
            print(f"Loaded brain: {len(q_table)} states, {total_turns_learned} turns learned")
        except: pass

load_brain()

# =====================================================================
# STATE DISCRETIZATION — compress game state into a compact key
# =====================================================================

def discretize_state(me: dict, enemies: list, tiles: list,
                     turn: int, mt: int, sz: int, sr: int) -> str:
    """Convert full game state into a compact string key with 10 features."""
    mx, my = me["x"], me["y"]
    hp, energy = me["health"], me["energy"]
    pct = turn / mt

    # 1. Health: 0=low(<35), 1=mid(35-70), 2=high(>70)
    h = 0 if hp < 35 else (1 if hp <= 70 else 2)

    # 2. Energy: 0=low(<5), 1=mid(5-12), 2=high(>12)
    e = 0 if energy < 5 else (1 if energy <= 12 else 2)

    # 3. Nearest enemy distance: 0=adjacent, 1=near(2-3), 2=far/none
    adj = [en for en in enemies if chebyshev(mx, my, en["x"], en["y"]) <= 1]
    near = [en for en in enemies if 1 < chebyshev(mx, my, en["x"], en["y"]) <= 3]
    d = 0 if adj else (1 if near else 2)

    # 4. Enemy killable (adjacent): 0=no/none, 1=yes
    my_d = 15 * (1 + me.get("damage_boost_stacks", 0))
    k = 1 if any(en["health"] <= my_d * 2 for en in adj) else 0

    # 5. On collectible: 0=no, 1=yes
    g = 1 if any((t.get("has_resource") or t.get("power_up"))
                  and t["x"] == mx and t["y"] == my for t in tiles) else 0

    # 6. Target nearby (resource/powerup within 3): 0=no, 1=yes
    n = 1 if any((t.get("has_resource") or t.get("power_up"))
                  and manhattan(mx, my, t["x"], t["y"]) <= 3
                  and not (t["x"] == mx and t["y"] == my) for t in tiles) else 0

    # 7. Zone status: 0=safe+future_safe, 1=safe_but_shrinking, 2=in_danger
    cx, cy = center_of(sz)
    in_s = max(abs(mx - cx), abs(my - cy)) <= sr
    taking = max(abs(mx - cx), abs(my - cy)) > sr
    z = 2 if taking else (1 if not future_safe_check(mx, my, turn, mt, sz) else 0)

    # 8. Phase: 0=early, 1=mid, 2=late
    p = 0 if pct < 0.30 else (1 if pct < 0.65 else 2)

    # 9. Have damage boost: 0=no, 1=yes
    b = 1 if me.get("damage_boost_stacks", 0) > 0 else 0

    # 10. Score status: 0=losing/tied, 1=winning
    my_score = me.get("score", 0)
    max_enemy_score = max((en.get("score", 0) for en in enemies), default=0)
    s = 1 if my_score > max_enemy_score else 0

    return f"{h}{e}{d}{k}{g}{n}{z}{p}{b}{s}"


def get_q_values(state_key: str) -> list[float]:
    if state_key not in q_table:
        q_table[state_key] = [0.0] * NUM_ACTIONS
        visit_count[state_key] = [0] * NUM_ACTIONS
    return q_table[state_key]


def get_visits(state_key: str) -> list[int]:
    if state_key not in visit_count:
        visit_count[state_key] = [0] * NUM_ACTIONS
    return visit_count[state_key]


def select_action_q(state_key: str) -> int:
    """ε-greedy action selection from Q-table."""
    qv = get_q_values(state_key)
    visits = get_visits(state_key)

    # Not enough data? Return -1 to signal fallback to rules
    total_visits = sum(visits)
    if total_visits < MIN_VISITS * NUM_ACTIONS:
        return -1  # use rules

    # Explore randomly with probability ε
    if random.random() < epsilon:
        return random.randint(0, NUM_ACTIONS - 1)

    # Pick best action (with tie-breaking)
    max_q = max(qv)
    best = [i for i, q in enumerate(qv) if abs(q - max_q) < 0.01]
    return random.choice(best)


def update_q(prev_state: str, action_idx: int, reward: float, new_state: str):
    """Q-learning update: Q(s,a) += α * (r + γ*max(Q(s',·)) - Q(s,a))"""
    global total_turns_learned
    qv = get_q_values(prev_state)
    new_qv = get_q_values(new_state)
    best_next = max(new_qv)

    old_q = qv[action_idx]
    qv[action_idx] = old_q + ALPHA * (reward + GAMMA * best_next - old_q)

    vc = get_visits(prev_state)
    vc[action_idx] += 1
    total_turns_learned += 1


def decay_epsilon():
    global epsilon
    epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)


# =====================================================================
# OPPONENT MODELING
# =====================================================================

def update_opponent(bot_id: str, enemy: dict):
    """Track an enemy's behavior patterns."""
    if bot_id not in opponent_data:
        opponent_data[bot_id] = {
            "seen": 0, "attacks": 0, "defends": 0, "collects": 0,
            "moves": 0, "waits": 0, "flees_low_hp": 0, "low_hp_seen": 0,
        }
    od = opponent_data[bot_id]
    od["seen"] += 1

    la = enemy.get("last_action", "")
    if la == "ATTACK": od["attacks"] += 1
    elif la == "DEFEND": od["defends"] += 1
    elif la == "COLLECT": od["collects"] += 1
    elif la == "WAIT": od["waits"] += 1
    elif la and la.startswith("MOVE"): od["moves"] += 1

    if enemy.get("health", 100) < 30:
        od["low_hp_seen"] += 1
        if la and la.startswith("MOVE"): od["flees_low_hp"] += 1


def predict_enemy(bot_id: str) -> dict:
    """Predict what an enemy is likely to do."""
    od = opponent_data.get(bot_id, {})
    s = max(od.get("seen", 0), 1)
    return {
        "attack_rate": od.get("attacks", 0) / s,
        "defend_rate": od.get("defends", 0) / s,
        "flee_rate": od.get("flees_low_hp", 0) / max(od.get("low_hp_seen", 0), 1),
        "passive": (od.get("waits", 0) + od.get("collects", 0)) / s,
    }


# =====================================================================
# GAME UTILITIES
# =====================================================================
def manhattan(x1,y1,x2,y2): return abs(x1-x2)+abs(y1-y2)
def chebyshev(x1,y1,x2,y2): return max(abs(x1-x2),abs(y1-y2))
def center_of(sz): c=(sz-1)/2; return(c,c)
def dist_center(x,y,sz): cx,cy=center_of(sz); return max(abs(x-cx),abs(y-cy))
def in_safe(x,y,sz,sr): cx,cy=center_of(sz); return abs(x-cx)<=sr and abs(y-cy)<=sr
def apply(x,y,a):
    if a=="MOVE_UP":return(x,y-1)
    if a=="MOVE_DOWN":return(x,y+1)
    if a=="MOVE_LEFT":return(x-1,y)
    if a=="MOVE_RIGHT":return(x+1,y)
    return(x,y)
def okp(x,y,sz,w): return 0<=x<sz and 0<=y<sz and (x,y) not in w
MOVS=["MOVE_UP","MOVE_DOWN","MOVE_LEFT","MOVE_RIGHT"]
DIRS=[(0,-1,"MOVE_UP"),(0,1,"MOVE_DOWN"),(-1,0,"MOVE_LEFT"),(1,0,"MOVE_RIGHT")]
def get_walls(tiles): return {(t["x"],t["y"]) for t in tiles if t.get("type")=="wall"}
def escape_routes(x,y,sz,w): return sum(1 for dx,dy,_ in DIRS if 0<=x+dx<sz and 0<=y+dy<sz and (x+dx,y+dy) not in w)
def is_trap(x,y,sz,w): return escape_routes(x,y,sz,w)<=1

def predict_sr(turn,mt,sz):
    ss=int(mt*0.7); ir=sz//2
    return max(0,ir-max(0,(turn-ss)//5)) if turn>=ss else ir
def future_safe_check(x,y,turn,mt,sz,ahead=15):
    return dist_center(x,y,sz)<=predict_sr(turn+ahead,mt,sz)

def bfs_to_safe(mx,my,sz,walls,sr,blocked=None):
    cx,cy=center_of(sz)
    if max(abs(mx-cx),abs(my-cy))<=sr: return(0,None)
    avoid=walls|(blocked or set()); visited={(mx,my)}; q=deque()
    for dx,dy,a in DIRS:
        nx,ny=mx+dx,my+dy
        if 0<=nx<sz and 0<=ny<sz and (nx,ny) not in avoid:
            if max(abs(nx-cx),abs(ny-cy))<=sr: return(1,a)
            visited.add((nx,ny)); q.append((nx,ny,1,a))
    if not q and blocked:
        for dx,dy,a in DIRS:
            nx,ny=mx+dx,my+dy
            if 0<=nx<sz and 0<=ny<sz and (nx,ny) not in walls:
                if max(abs(nx-cx),abs(ny-cy))<=sr: return(1,a)
                visited.add((nx,ny)); q.append((nx,ny,1,a))
    while q:
        x,y,d,f=q.popleft()
        if d>sz*2: break
        for dx,dy,_ in DIRS:
            nx,ny=x+dx,y+dy
            if(nx,ny) not in visited and 0<=nx<sz and 0<=ny<sz and(nx,ny) not in walls:
                visited.add((nx,ny))
                if max(abs(nx-cx),abs(ny-cy))<=sr: return(d+1,f)
                q.append((nx,ny,d+1,f))
    return(999,None)

def move_toward(mx,my,tx,ty,sz,walls,sr,pos,avoid_traps=True):
    best_a,best_s=None,-9999
    pp=pos[-2] if len(pos)>=2 else None
    for a in MOVS:
        nx,ny=apply(mx,my,a)
        if not okp(nx,ny,sz,walls): continue
        s=(manhattan(mx,my,tx,ty)-manhattan(nx,ny,tx,ty))*8
        if in_safe(nx,ny,sz,sr): s+=6
        else: s-=15
        if pp and(nx,ny)==pp: s-=10
        if avoid_traps and is_trap(nx,ny,sz,walls): s-=12
        if s>best_s: best_s,best_a=s,a
    return best_a

def flee_from(mx,my,enemies,sz,walls,sr,pos):
    best_a,best_s=None,-9999
    pp=pos[-2] if len(pos)>=2 else None
    for a in MOVS:
        nx,ny=apply(mx,my,a)
        if not okp(nx,ny,sz,walls): continue
        s=sum((chebyshev(nx,ny,e["x"],e["y"])-chebyshev(mx,my,e["x"],e["y"]))*15 for e in enemies)
        if in_safe(nx,ny,sz,sr): s+=10
        else: s-=20
        if pp and(nx,ny)==pp: s-=8
        if is_trap(nx,ny,sz,walls): s-=12
        if s>best_s: best_s,best_a=s,a
    return best_a

def explore_move(mx,my,sz,walls,sr,enemies,pos,turn,mt):
    best_a,best_s=None,-9999
    pp=pos[-2] if len(pos)>=2 else None
    safe_now=in_safe(mx,my,sz,sr)
    shuffled=list(MOVS); random.shuffle(shuffled)
    for a in shuffled:
        nx,ny=apply(mx,my,a)
        if not okp(nx,ny,sz,walls): continue
        if safe_now and not in_safe(nx,ny,sz,sr): continue
        s=0.0
        if pp and(nx,ny)==pp: s-=12
        if in_safe(nx,ny,sz,sr): s+=8
        else: s-=15
        if future_safe_check(nx,ny,turn,mt,sz): s+=5
        if is_trap(nx,ny,sz,walls): s-=10
        s-=dist_center(nx,ny,sz)*0.4
        s+=random.random()*4
        if s>best_s: best_s,best_a=s,a
    return best_a


# =====================================================================
# ACTION TRANSLATION — abstract action → concrete game action
# =====================================================================

def translate_action(action_idx: int, me: dict, enemies: list, tiles: list,
                     state: dict, ms) -> MoveResponse:
    """Convert abstract Q-action into a concrete game action."""
    mx, my = me["x"], me["y"]
    energy, health = me["energy"], me["health"]
    sz, sr = state["arena_size"], state["safe_zone_radius"]
    turn, mt = state["turn"], state["max_turns"]
    walls = get_walls(tiles)
    epos = {(e["x"], e["y"]) for e in enemies}
    adj = [e for e in enemies if chebyshev(mx, my, e["x"], e["y"]) <= 1]
    action_name = ACTIONS[action_idx]

    if action_name == "ATTACK":
        if adj and energy >= 3:
            return MoveResponse(action="ATTACK", emoji="⚔️", mood="Q:attack")
        # Can't attack, move toward nearest enemy instead
        if enemies and energy >= 1:
            t = min(enemies, key=lambda e: manhattan(mx, my, e["x"], e["y"]))
            a = move_toward(mx, my, t["x"], t["y"], sz, walls, sr, ms.positions)
            if a: return MoveResponse(action=a, emoji="🐺", mood="Q:chase_to_attack")
        return MoveResponse(action="WAIT", emoji="⏳", mood="Q:no_target")

    elif action_name == "COLLECT":
        on = any((t.get("has_resource") or t.get("power_up")) and t["x"] == mx and t["y"] == my for t in tiles)
        if on and energy >= 2:
            return MoveResponse(action="COLLECT", emoji="💰", mood="Q:collect")
        return MoveResponse(action="WAIT", emoji="⏳", mood="Q:nothing_here")

    elif action_name == "DEFEND":
        if energy >= 2:
            return MoveResponse(action="DEFEND", emoji="🛡️", mood="Q:defend")
        return MoveResponse(action="WAIT", emoji="⏳", mood="Q:no_energy")

    elif action_name == "WAIT":
        return MoveResponse(action="WAIT", emoji="🔋", mood="Q:wait")

    elif action_name == "CHASE_ENEMY":
        if enemies and energy >= 1:
            # Chase weakest visible enemy
            t = min(enemies, key=lambda e: e["health"])
            a = move_toward(mx, my, t["x"], t["y"], sz, walls, sr, ms.positions)
            if a: return MoveResponse(action=a, emoji="🐺", mood="Q:chase_enemy")
        return MoveResponse(action="WAIT", emoji="🔍", mood="Q:no_enemies")

    elif action_name == "CHASE_TARGET":
        targets = [t for t in tiles if (t.get("has_resource") or t.get("power_up"))
                   and not (t["x"] == mx and t["y"] == my)]
        if targets and energy >= 1:
            # Pick nearest
            t = min(targets, key=lambda t: manhattan(mx, my, t["x"], t["y"]))
            a = move_toward(mx, my, t["x"], t["y"], sz, walls, sr, ms.positions)
            if a: return MoveResponse(action=a, emoji="🎯", mood="Q:chase_target")
        return MoveResponse(action="WAIT", emoji="🔍", mood="Q:no_targets")

    elif action_name == "FLEE":
        if enemies and energy >= 1:
            a = flee_from(mx, my, enemies, sz, walls, sr, ms.positions)
            if a: return MoveResponse(action=a, emoji="💨", mood="Q:flee")
        return MoveResponse(action="WAIT", emoji="😴", mood="Q:safe")

    elif action_name == "GO_CENTER":
        cx, cy = int(center_of(sz)[0]), int(center_of(sz)[1])
        if energy >= 1:
            a = move_toward(mx, my, cx, cy, sz, walls, sr, ms.positions)
            if a: return MoveResponse(action=a, emoji="🏠", mood="Q:center")
        return MoveResponse(action="WAIT", emoji="🏠", mood="Q:at_center")

    else:  # EXPLORE
        if energy >= 1:
            a = explore_move(mx, my, sz, walls, sr, enemies, ms.positions, turn, mt)
            if a: return MoveResponse(action=a, emoji="🔍", mood="Q:explore")
        return MoveResponse(action="WAIT", emoji="😴", mood="Q:idle")


# =====================================================================
# RULE-BASED FALLBACK (used when Q-table lacks data)
# =====================================================================

def rule_based_action(me, enemies, tiles, state, ms) -> tuple[int, MoveResponse]:
    """Returns (action_index, MoveResponse) using hardcoded rules."""
    mx, my, energy, health = me["x"], me["y"], me["energy"], me["health"]
    sz, sr = state["arena_size"], state["safe_zone_radius"]
    turn, mt = state["turn"], state["max_turns"]
    walls = get_walls(tiles)
    epos = {(e["x"], e["y"]) for e in enemies}

    adj = [e for e in enemies if chebyshev(mx, my, e["x"], e["y"]) <= 1]
    on_collect = any((t.get("has_resource") or t.get("power_up")) and t["x"] == mx and t["y"] == my for t in tiles)

    # Zone emergency
    taking_dmg = dist_center(mx, my, sz) > sr
    if taking_dmg:
        if health <= 15 and energy >= 2:
            return (2, MoveResponse(action="DEFEND", emoji="🛡️", mood="R:zone_shield"))
        _, esc = bfs_to_safe(mx, my, sz, walls, sr, epos)
        if not esc: _, esc = bfs_to_safe(mx, my, sz, walls, sr, None)
        if esc and energy >= 1:
            return (7, MoveResponse(action=esc, emoji="🚨", mood="R:zone_escape"))

    # Zone warning
    if not in_safe(mx, my, sz, sr) or not future_safe_check(mx, my, turn, mt, sz):
        if on_collect and energy >= 2:
            return (1, MoveResponse(action="COLLECT", emoji="💰", mood="R:grab+run"))
        _, esc = bfs_to_safe(mx, my, sz, walls, sr, epos)
        if esc and energy >= 1:
            return (7, MoveResponse(action=esc, emoji="⚠️", mood="R:pre_escape"))

    # Collect
    if on_collect and energy >= 2:
        return (1, MoveResponse(action="COLLECT", emoji="💰", mood="R:collect"))

    # Free kill
    my_d = 15 * (1 + me.get("damage_boost_stacks", 0))
    if adj and energy >= 3:
        for en in adj:
            d = my_d // 2 if en.get("is_defending") else my_d
            if en["health"] <= d:
                return (0, MoveResponse(action="ATTACK", emoji="💀", mood="R:execute"))

    # Flee if low
    if adj and health < 30:
        a = flee_from(mx, my, enemies, sz, walls, sr, ms.positions)
        if a and energy >= 1:
            return (6, MoveResponse(action=a, emoji="💨", mood="R:flee"))

    # Attack if advantage
    if adj and energy >= 4 and health > 40:
        weakest = min(adj, key=lambda e: e["health"])
        if me["health"] > weakest["health"] or me.get("damage_boost_stacks", 0) > 0:
            return (0, MoveResponse(action="ATTACK", emoji="⚔️", mood="R:attack"))

    # Chase targets
    targets = [t for t in tiles if (t.get("has_resource") or t.get("power_up"))
               and not (t["x"] == mx and t["y"] == my)
               and (in_safe(t["x"], t["y"], sz, sr) or turn / mt < 0.3)]
    if targets and energy >= 3:
        t = min(targets, key=lambda t: manhattan(mx, my, t["x"], t["y"]))
        a = move_toward(mx, my, t["x"], t["y"], sz, walls, sr, ms.positions)
        if a: return (5, MoveResponse(action=a, emoji="🎯", mood="R:chase"))

    # Hunt nearby enemies
    near = [e for e in enemies if 1 < chebyshev(mx, my, e["x"], e["y"]) <= 3]
    if near and energy >= 5 and health > 50:
        t = min(near, key=lambda e: e["health"])
        a = move_toward(mx, my, t["x"], t["y"], sz, walls, sr, ms.positions)
        if a: return (4, MoveResponse(action=a, emoji="🐺", mood="R:hunt"))

    # Energy
    if energy < 3:
        return (3, MoveResponse(action="WAIT", emoji="🔋", mood="R:recharge"))

    # Explore
    a = explore_move(mx, my, sz, walls, sr, enemies, ms.positions, turn, mt)
    if a and energy >= 1:
        return (8, MoveResponse(action=a, emoji="🔍", mood="R:explore"))

    return (3, MoveResponse(action="WAIT", emoji="😴", mood="R:idle"))


# =====================================================================
# MATCH STATE (per concurrent match)
# =====================================================================

class MatchState:
    def __init__(self, mid):
        self.match_id = mid
        self.last_update = time.time()
        self.positions: list[tuple[int, int]] = []
        self.prev_state_key: str = ""
        self.prev_action_idx: int = -1
        self.prev_health: int = 100
        self.prev_score: int = 0
        self.data = {"match_id": mid, "turns_played": 0, "max_turns": 100,
                     "num_bots": 2, "last_health": 100, "last_score": 0,
                     "total_damage_taken": 0, "total_score_gained": 0, "kills": 0,
                     "health_history": [], "enemies_alive_last_turn": 0,
                     "q_actions": 0, "rule_actions": 0}
        self.enemy_tracker: dict[str, dict] = {}

active_matches: dict[str, MatchState] = {}
match_history: list[dict] = []
total_matches = total_won = total_lost = total_died = 0
turns_since_save = 0

def cleanup_stale():
    now = time.time()
    stale = [m for m, ms in active_matches.items() if now - ms.last_update > STALE_TIMEOUT]
    for m in stale:
        finish_match(m)
        del active_matches[m]

def detect_outcome(ms):
    d = ms.data; t, mt = d["turns_played"], d["max_turns"]
    sc, hp = d["last_score"], d["last_health"]
    hh = d.get("health_history", [])
    es = {b: i["last_score"] for b, i in ms.enemy_tracker.items()}
    he = max(es.values()) if es else 0
    hd = len(hh) >= 3 and hh[-1] < hh[0] - 10
    if t >= mt - 2:
        return "won_score" if sc > he else ("tied" if sc == he else "lost_score")
    if hp > 20 and not hd: return "won_elimination"
    if hp <= 15 or (hd and hp < 30): return "died"
    if t < mt * 0.25: return "died"
    return "won_elimination" if hp > 50 else "died"

def finish_match(mid):
    global total_matches, total_won, total_lost, total_died
    ms = active_matches.get(mid)
    if not ms or ms.data["turns_played"] < 3: return
    total_matches += 1
    o = detect_outcome(ms); d = ms.data; d["outcome"] = o
    d["enemy_scores"] = {b: i["last_score"] for b, i in ms.enemy_tracker.items()}
    if o.startswith("won"): total_won += 1
    elif o in ("lost_score", "tied"): total_lost += 1
    else: total_died += 1
    sp = d["turns_played"] / max(1, d["max_turns"])
    r = d["last_score"] + sp * 40
    bonus = {"won_elimination": 90, "won_score": 70, "tied": 30, "lost_score": -10}.get(o, -50*(1-sp))
    r += bonus
    d["reward"] = round(r, 1)
    match_history.append(dict(d))
    save_brain()


# =====================================================================
# MAIN DECISION ENGINE
# =====================================================================

def choose_action(ms: MatchState, state: dict) -> MoveResponse:
    global turns_since_save

    me = state["self"]
    mx, my = me["x"], me["y"]
    enemies = state.get("enemies", [])
    tiles = state.get("visible_tiles", [])
    d = ms.data

    # Track
    hp, sc = me["health"], me["score"]
    dmg_taken = max(0, ms.prev_health - hp)
    score_gained = max(0, sc - ms.prev_score)
    d["turns_played"] += 1; d["max_turns"] = state["max_turns"]
    d["num_bots"] = state.get("num_bots", 2)
    d["total_damage_taken"] += dmg_taken; d["total_score_gained"] += score_gained
    d["last_health"] = hp; d["last_score"] = sc
    d["enemies_alive_last_turn"] = len(enemies)
    d["health_history"].append(hp)
    if len(d["health_history"]) > 10: d["health_history"].pop(0)
    if score_gained >= 30: d["kills"] += 1
    ms.positions.append((mx, my))
    if len(ms.positions) > 12: ms.positions.pop(0)

    # Update opponent models
    for e in enemies:
        update_opponent(e["bot_id"], e)
        ms.enemy_tracker[e["bot_id"]] = {"last_score": e.get("score", 0),
            "last_health": e.get("health", 100), "last_seen_turn": state["turn"]}

    # === Q-LEARNING UPDATE from previous turn ===
    current_state = discretize_state(me, enemies, tiles, state["turn"], state["max_turns"],
                                      state["arena_size"], state["safe_zone_radius"])

    if ms.prev_state_key and ms.prev_action_idx >= 0:
        # Calculate reward for previous action
        reward = score_gained * 1.0 - dmg_taken * 0.5 + 0.3  # small alive bonus
        if score_gained >= 30: reward += 10  # kill bonus
        if dmg_taken >= 15: reward -= 3  # took a big hit

        # Opponent prediction bonus: if we attacked and enemy was predicted to not defend
        update_q(ms.prev_state_key, ms.prev_action_idx, reward, current_state)
        decay_epsilon()

    # === ACTION SELECTION ===
    q_action_idx = select_action_q(current_state)

    if q_action_idx >= 0:
        # Q-table has enough data — use it
        response = translate_action(q_action_idx, me, enemies, tiles, state, ms)
        action_idx = q_action_idx
        d["q_actions"] = d.get("q_actions", 0) + 1
    else:
        # Fall back to rules
        action_idx, response = rule_based_action(me, enemies, tiles, state, ms)
        d["rule_actions"] = d.get("rule_actions", 0) + 1

    # Store for next turn's Q-update
    ms.prev_state_key = current_state
    ms.prev_action_idx = action_idx
    ms.prev_health = hp
    ms.prev_score = sc
    ms.last_update = time.time()

    # Periodic save
    turns_since_save += 1
    if turns_since_save >= SAVE_INTERVAL:
        save_brain(); turns_since_save = 0

    return response


# =====================================================================
# HTTP
# =====================================================================

@app.post("/move")
async def move(state: dict[str, Any]) -> MoveResponse:
    mid = state["match_id"]
    cleanup_stale()
    if mid not in active_matches:
        active_matches[mid] = MatchState(mid)
    ms = active_matches[mid]
    r = choose_action(ms, state)
    if ARENA_URL:
        try:
            async with httpx.AsyncClient() as c:
                await c.post(f"{ARENA_URL}/arena/bot-update",
                    json={"bot_id": state["self"]["bot_id"], "status": r.mood,
                          "message": f"T{state['turn']}:{r.action}", "color": BOT_COLOR}, timeout=0.3)
        except: pass
    return r

@app.get("/health")
async def hc(): return {"status": "ok", "bot_id": BOT_ID}

@app.get("/stats/json")
async def sj():
    rec = match_history[-20:]
    return {"matches": total_matches, "won": total_won, "lost": total_lost, "died": total_died,
            "q_table_size": len(q_table), "total_turns_learned": total_turns_learned,
            "epsilon": round(epsilon, 4), "active_matches": len(active_matches),
            "opponents_tracked": len(opponent_data),
            "opponent_profiles": {k: {kk: round(vv, 2) if isinstance(vv, float) else vv
                                       for kk, vv in v.items()} for k, v in list(opponent_data.items())[:10]},
            "recent": [{"outcome": m.get("outcome"), "score": m["last_score"],
                         "q_pct": f"{m.get('q_actions',0)/(m.get('q_actions',0)+m.get('rule_actions',1))*100:.0f}%",
                         "turns": m["turns_played"], "kills": m["kills"]} for m in rec]}

@app.get("/stats", response_class=HTMLResponse)
async def sp():
    wr = f"{total_won/max(1,total_matches)*100:.0f}" if total_matches else "0"
    rec = match_history[-20:]
    avs = sum(m["last_score"] for m in rec) / max(1, len(rec)) if rec else 0

    # Active matches
    am = ""
    for mid, ms in active_matches.items():
        d = ms.data; hp = d["last_health"]
        hc = "#4ade80" if hp > 60 else ("#fbbf24" if hp > 30 else "#ef4444")
        es = ", ".join(f"{b[:6]}:{i['last_score']}" for b, i in ms.enemy_tracker.items())
        qa, ra = d.get("q_actions", 0), d.get("rule_actions", 0)
        qp = f"{qa/(qa+ra)*100:.0f}" if qa + ra > 0 else "0"
        am += f'<div class="card live"><h2>🔴 {mid[:8]}</h2><div class="sr"><span>Turn</span><strong>{d["turns_played"]}/{d["max_turns"]}</strong></div><div class="sr"><span>HP</span><div class="bar-bg"><div class="bar" style="width:{hp}%;background:{hc}"></div></div><strong>{hp}</strong></div><div class="sr"><span>Score</span><strong>{d["last_score"]}</strong></div><div class="sr"><span>Q-brain</span><strong>{qp}% of decisions</strong></div><div class="sr"><span>Enemies</span><strong style="font-size:.7em">{es or "none"}</strong></div></div>'

    # History
    rows = ""
    for m in reversed(rec[-15:]):
        o = m.get("outcome", "?")
        if o == "won_elimination": i, c, l = "🏆", "won", "WON(elim)"
        elif o == "won_score": i, c, l = "🥇", "won", "WON(score)"
        elif o == "tied": i, c, l = "🤝", "tied", "TIED"
        elif o == "lost_score": i, c, l = "🥈", "lost", "LOST"
        else: i, c, l = "💀", "died", "DIED"
        qa, ra = m.get("q_actions", 0), m.get("rule_actions", 0)
        qp = f"{qa/(qa+ra)*100:.0f}%" if qa + ra > 0 else "0%"
        rows += f'<tr class="{c}"><td>{i} {l}</td><td>{m["last_score"]}</td><td>{m["kills"]}</td><td>{qp}</td><td>{m["turns_played"]}/{m["max_turns"]}</td><td>{m.get("reward",0):.0f}</td></tr>'

    # Top opponents
    opp = ""
    for bid, od in sorted(opponent_data.items(), key=lambda x: x[1].get("seen", 0), reverse=True)[:8]:
        s = max(od.get("seen", 0), 1)
        ar = od.get("attacks", 0) / s * 100
        dr = od.get("defends", 0) / s * 100
        style = "Aggressive" if ar > 40 else ("Defensive" if dr > 30 else "Passive" if od.get("waits", 0) / s > 0.3 else "Balanced")
        opp += f'<div class="opp"><strong>{bid[:12]}</strong> <span class="tag">{style}</span> <span style="font-size:.7em;color:#64748b">seen {s}x · atk {ar:.0f}% · def {dr:.0f}%</span></div>'

    return HTMLResponse(content=f'''<!DOCTYPE html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta http-equiv="refresh" content="5"><title>🧠 Q-Bot</title><style>*{{margin:0;padding:0;box-sizing:border-box}}body{{font-family:-apple-system,sans-serif;background:#0f172a;color:#e2e8f0;padding:16px}}h1{{text-align:center;font-size:1.5em;margin-bottom:16px;color:#38bdf8}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(100px,1fr));gap:8px;margin-bottom:14px}}.kpi{{background:#1e293b;border-radius:10px;padding:10px;text-align:center}}.kpi .num{{font-size:1.5em;font-weight:bold}}.kpi .label{{font-size:.7em;color:#94a3b8}}.green{{color:#4ade80}}.red{{color:#ef4444}}.blue{{color:#38bdf8}}.yellow{{color:#fbbf24}}.purple{{color:#a78bfa}}.orange{{color:#fb923c}}.card{{background:#1e293b;border-radius:10px;padding:12px;margin-bottom:12px}}.card h2{{font-size:.95em;margin-bottom:8px;color:#94a3b8}}.card.live{{border:1px solid #ef4444}}.sr{{display:flex;align-items:center;gap:5px;margin-bottom:4px}}.sr span{{width:70px;font-size:.75em;color:#94a3b8}}.sr strong{{font-size:.85em}}.bar-bg{{flex:1;background:#334155;border-radius:4px;height:10px;overflow:hidden}}.bar{{height:100%;border-radius:4px;background:#38bdf8}}table{{width:100%;border-collapse:collapse;font-size:.75em}}th{{text-align:left;color:#94a3b8;padding:5px;border-bottom:1px solid #334155}}td{{padding:5px;border-bottom:1px solid #1e293b}}tr.won td{{color:#4ade80}}tr.lost td{{color:#fbbf24}}tr.died td{{color:#ef4444}}tr.tied td{{color:#94a3b8}}.two-col{{display:grid;grid-template-columns:1fr 1fr;gap:12px}}@media(max-width:600px){{.two-col{{grid-template-columns:1fr}}}}.opp{{padding:6px 0;border-bottom:1px solid #1e293b}}.tag{{font-size:.7em;padding:2px 6px;border-radius:4px;background:#334155;color:#94a3b8}}</style></head><body>
<h1>🧠 RoboKova Q-Learning Bot</h1>
<div class="grid">
<div class="kpi"><div class="num blue">{total_matches}</div><div class="label">Matches</div></div>
<div class="kpi"><div class="num green">{total_won}</div><div class="label">Won</div></div>
<div class="kpi"><div class="num yellow">{total_lost}</div><div class="label">Lost</div></div>
<div class="kpi"><div class="num red">{total_died}</div><div class="label">Died</div></div>
<div class="kpi"><div class="num {"green" if int(wr)>=50 else "red"}">{wr}%</div><div class="label">Win Rate</div></div>
<div class="kpi"><div class="num purple">{len(q_table)}</div><div class="label">States</div></div>
<div class="kpi"><div class="num orange">{total_turns_learned:,}</div><div class="label">Turns Learned</div></div>
<div class="kpi"><div class="num blue">{epsilon:.2f}</div><div class="label">Explore ε</div></div>
</div>
{am}
<div class="two-col">
<div class="card"><h2>📜 History</h2><table><tr><th>Result</th><th>Score</th><th>Kills</th><th>Q-Brain</th><th>Turns</th><th>Reward</th></tr>{rows or'<tr><td colspan="6" style="color:#64748b">No matches</td></tr>'}</table></div>
<div class="card"><h2>🕵️ Opponent Intel ({len(opponent_data)} tracked)</h2>{opp or'<div style="color:#64748b">No opponents seen yet</div>'}</div>
</div>
<p style="text-align:center;color:#475569;margin-top:12px;font-size:.65em">Refresh 5s · {len(q_table)} states · {total_turns_learned:,} turns learned · ε={epsilon:.3f} · {len(active_matches)} active</p></body></html>''')
