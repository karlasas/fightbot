"""RoboKova Self-Learning Bot v8 — Aggressive & Smart

UPGRADES:
1. Weights saved to file — survives server restarts (not redeploys)
2. MUCH more aggressive — actively hunts, prioritizes kills for score
3. Phase-specific weights — different strategy for early/mid/late
4. Committed zone escape — doesn't stop until safe, even if 5+ steps
5. Wall trap avoidance — prefers positions with multiple escape routes

Run: uvicorn bot:app --host 0.0.0.0 --port 5001 --reload
"""

from __future__ import annotations
import json, os, random
from collections import deque
from pathlib import Path
from typing import Any
import httpx
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

app = FastAPI(title="RoboKova Bot v8")
BOT_ID = os.environ.get("BOT_ID", "v8-aggro")
BOT_COLOR = os.environ.get("BOT_COLOR", "#e94560")
ARENA_URL = os.environ.get("ARENA_URL", "")
WEIGHTS_FILE = Path("learned_weights.json")

class MoveResponse(BaseModel):
    action: str
    emoji: str = ""
    mood: str = ""

# =====================================================================
# PHASE-SPECIFIC LEARNABLE WEIGHTS
# Each phase has its own strategy that evolves independently
# =====================================================================

DEFAULT_WEIGHTS = {
    "early": {
        "aggression": 0.35, "flee_health_pct": 0.20, "defend_pref": 0.25,
        "kill_chase": 0.55, "helpless_bonus": 0.85, "wait_defense": 0.70,
        "resource_pri": 0.90, "energy_pack_pri": 0.80, "dmg_boost_pri": 0.95,
        "shield_pri": 0.65, "speed_pri": 0.50, "vision_pri": 0.30,
        "min_energy": 3.0, "center_pull": 0.20,
    },
    "mid": {
        "aggression": 0.65, "flee_health_pct": 0.25, "defend_pref": 0.30,
        "kill_chase": 0.80, "helpless_bonus": 0.90, "wait_defense": 0.75,
        "resource_pri": 0.70, "energy_pack_pri": 0.85, "dmg_boost_pri": 0.90,
        "shield_pri": 0.80, "speed_pri": 0.50, "vision_pri": 0.20,
        "min_energy": 3.0, "center_pull": 0.35,
    },
    "late": {
        "aggression": 0.75, "flee_health_pct": 0.22, "defend_pref": 0.40,
        "kill_chase": 0.85, "helpless_bonus": 0.90, "wait_defense": 0.60,
        "resource_pri": 0.60, "energy_pack_pri": 0.90, "dmg_boost_pri": 0.85,
        "shield_pri": 0.85, "speed_pri": 0.40, "vision_pri": 0.15,
        "min_energy": 2.0, "center_pull": 0.55,
    },
}

BOUNDS = {
    "aggression": (0.10, 0.95), "flee_health_pct": (0.08, 0.45), "defend_pref": (0.10, 0.75),
    "kill_chase": (0.25, 0.95), "helpless_bonus": (0.40, 0.95), "wait_defense": (0.30, 0.90),
    "resource_pri": (0.30, 0.95), "energy_pack_pri": (0.40, 0.95), "dmg_boost_pri": (0.50, 0.99),
    "shield_pri": (0.30, 0.95), "speed_pri": (0.15, 0.75), "vision_pri": (0.05, 0.55),
    "min_energy": (1.0, 7.0), "center_pull": (0.10, 0.70),
}

LR = 0.10  # slightly higher learning rate for faster adaptation
NOISE = 0.03

phase_weights: dict[str, dict] = {}

def load_weights():
    global phase_weights
    if WEIGHTS_FILE.exists():
        try:
            phase_weights = json.loads(WEIGHTS_FILE.read_text())
            print(f"Loaded weights from {WEIGHTS_FILE}")
            return
        except: pass
    phase_weights = {p: dict(w) for p, w in DEFAULT_WEIGHTS.items()}

def save_weights():
    try:
        WEIGHTS_FILE.write_text(json.dumps(phase_weights, indent=2))
    except: pass

load_weights()  # load on startup

def get_w(phase): return phase_weights.get(phase, phase_weights.get("mid", {}))

def get_active_w(phase):
    w = dict(get_w(phase))
    if len(match_history) < 30:
        for k in w:
            lo, hi = BOUNDS.get(k, (0, 1))
            w[k] = max(lo, min(hi, w[k] + random.gauss(0, NOISE * 0.4)))
    return w

# =====================================================================
# MATCH TRACKING
# =====================================================================
match_history: list[dict] = []
current_match: dict = {}
total_matches = total_won = total_lost = total_died = 0
positions: list[tuple[int, int]] = []
prev_action = ""
consecutive_waits = 0
escaping_zone = False  # committed escape mode
enemy_tracker: dict[str, dict] = {}

# Phase performance tracking within a match
phase_perf: dict[str, dict] = {}

def new_match(mid):
    global current_match, enemy_tracker, phase_perf, escaping_zone
    enemy_tracker = {}
    phase_perf = {p: {"score_gained": 0, "dmg_taken": 0, "kills": 0, "turns": 0} for p in ("early", "mid", "late")}
    escaping_zone = False
    current_match = {"match_id": mid, "turns_played": 0, "max_turns": 100, "num_bots": 2,
        "last_health": 100, "last_score": 0, "total_damage_taken": 0,
        "total_score_gained": 0, "kills": 0, "health_history": [],
        "enemies_alive_last_turn": 0}

def get_phase(pct):
    if pct < 0.30: return "early"
    if pct < 0.65: return "mid"
    return "late"

def track_turn(state):
    me = state["self"]; hp, sc = me["health"], me["score"]
    ph, ps = current_match.get("last_health", 100), current_match.get("last_score", 0)
    dmg, sg = max(0, ph - hp), max(0, sc - ps)
    pct = state["turn"] / state["max_turns"]
    phase = get_phase(pct)

    current_match["turns_played"] += 1
    current_match["max_turns"] = state["max_turns"]
    current_match["num_bots"] = state.get("num_bots", 2)
    current_match["total_damage_taken"] += dmg
    current_match["total_score_gained"] += sg
    current_match["last_health"] = hp; current_match["last_score"] = sc
    current_match["enemies_alive_last_turn"] = len(state.get("enemies", []))
    current_match["health_history"].append(hp)
    if len(current_match["health_history"]) > 10: current_match["health_history"].pop(0)
    if sg >= 30: current_match["kills"] += 1

    # Track per-phase performance
    pp = phase_perf[phase]
    pp["score_gained"] += sg; pp["dmg_taken"] += dmg; pp["turns"] += 1
    if sg >= 30: pp["kills"] += 1

    for e in state.get("enemies", []):
        enemy_tracker[e["bot_id"]] = {"last_score": e.get("score", 0),
            "last_health": e.get("health", 100), "last_seen_turn": state["turn"],
            "seen_count": enemy_tracker.get(e["bot_id"], {}).get("seen_count", 0) + 1}

def detect_outcome():
    t, mt = current_match["turns_played"], current_match["max_turns"]
    sc, hp = current_match["last_score"], current_match["last_health"]
    hh = current_match.get("health_history", [])
    el = current_match["enemies_alive_last_turn"]
    es = {b: i["last_score"] for b, i in enemy_tracker.items()}
    he = max(es.values()) if es else 0
    hd = len(hh) >= 3 and hh[-1] < hh[0] - 10
    if t >= mt - 2:
        return "won_score" if sc > he else ("tied" if sc == he else "lost_score")
    if hp > 20 and not hd and (el == 0 or hp > 40): return "won_elimination"
    if (hd and hp < 30) or hp <= 15: return "died"
    if el > 0 and hd: return "died"
    if t < mt * 0.3: return "died"
    return "won_elimination" if hp > 50 else "died"

def finish_match():
    global total_matches, total_won, total_lost, total_died
    if not current_match.get("match_id"): return
    total_matches += 1; o = detect_outcome()
    current_match["outcome"] = o
    current_match["enemy_scores"] = {b: i["last_score"] for b, i in enemy_tracker.items()}
    if o.startswith("won"): total_won += 1
    elif o in ("lost_score", "tied"): total_lost += 1
    else: total_died += 1

    # Calculate per-phase rewards and learn each phase independently
    sp = current_match["turns_played"] / max(1, current_match["max_turns"])
    base_bonus = 0
    if o == "won_elimination": base_bonus = 90
    elif o == "won_score": base_bonus = 70
    elif o == "tied": base_bonus = 30
    elif o == "lost_score": base_bonus = -10
    else: base_bonus = -50 * (1 - sp)

    total_reward = current_match["last_score"] + sp * 40 + base_bonus
    current_match["reward"] = round(total_reward, 1)
    match_history.append(dict(current_match))

    # Learn per phase
    if len(match_history) >= 3:
        for phase_name, pp in phase_perf.items():
            if pp["turns"] == 0: continue
            # Phase reward = score gained - damage cost + share of match outcome
            pr = pp["score_gained"] - pp["dmg_taken"] * 0.5 + pp["kills"] * 15
            pr += base_bonus * (pp["turns"] / max(1, current_match["turns_played"]))
            learn_phase(phase_name, pr)
        save_weights()

def learn_phase(phase_name, reward):
    global phase_weights
    history_rewards = []
    for m in match_history:
        # Approximate phase reward from overall
        history_rewards.append(m.get("reward", 0))
    avg = sum(history_rewards) / len(history_rewards) if history_rewards else 0
    # Scale reward relative to average match reward
    adv = max(-1, min(1, (reward - avg * 0.3) / max(abs(avg * 0.3), 1)))

    w = phase_weights[phase_name]
    for k in w:
        lo, hi = BOUNDS.get(k, (0, 1))
        n = w[k] + LR * adv + random.gauss(0, NOISE * max(0.25, 1 - len(match_history) / 100))
        w[k] = max(lo, min(hi, n))

# =====================================================================
# ZONE ENGINE
# =====================================================================
def predict_sr(turn, mt, sz):
    ss = int(mt * 0.7); ir = sz // 2
    return max(0, ir - max(0, (turn - ss) // 5)) if turn >= ss else ir

def zone_dmg(mx, my, sz, sr):
    return 5 if dist_center(mx, my, sz) > sr else 0

def turns_til_danger(mx, my, turn, mt, sz, sr):
    d = dist_center(mx, my, sz)
    if d > sr: return 0
    for f in range(1, mt - turn + 1):
        if d > predict_sr(turn + f, mt, sz): return f
    return mt - turn

def future_safe(x, y, turn, mt, sz, ahead=15):
    return dist_center(x, y, sz) <= predict_sr(turn + ahead, mt, sz)

def bfs_to_safe(mx, my, sz, walls, sr, blocked=None):
    cx, cy = center_of(sz)
    if max(abs(mx-cx), abs(my-cy)) <= sr: return (0, None)
    avoid = walls | (blocked or set())
    visited = {(mx, my)}; q = deque()
    for dx, dy, a in DIRS:
        nx, ny = mx+dx, my+dy
        if 0<=nx<sz and 0<=ny<sz and (nx,ny) not in avoid:
            if max(abs(nx-cx),abs(ny-cy))<=sr: return (1, a)
            visited.add((nx,ny)); q.append((nx,ny,1,a))
    if not q and blocked:
        for dx, dy, a in DIRS:
            nx, ny = mx+dx, my+dy
            if 0<=nx<sz and 0<=ny<sz and (nx,ny) not in walls:
                if max(abs(nx-cx),abs(ny-cy))<=sr: return (1, a)
                visited.add((nx,ny)); q.append((nx,ny,1,a))
    while q:
        x,y,d,f = q.popleft()
        if d > sz*2: break
        for dx,dy,_ in DIRS:
            nx,ny=x+dx,y+dy
            if (nx,ny) not in visited and 0<=nx<sz and 0<=ny<sz and (nx,ny) not in walls:
                visited.add((nx,ny))
                if max(abs(nx-cx),abs(ny-cy))<=sr: return (d+1,f)
                q.append((nx,ny,d+1,f))
    return (999, None)

# =====================================================================
# WALL TRAP DETECTION
# =====================================================================
def escape_routes(x, y, sz, walls):
    """Count how many non-wall adjacent tiles exist. Fewer = more trapped."""
    count = 0
    for dx, dy, _ in DIRS:
        nx, ny = x+dx, y+dy
        if 0<=nx<sz and 0<=ny<sz and (nx,ny) not in walls: count += 1
    return count

def is_trap_position(x, y, sz, walls):
    """Position with only 1 exit = trap. 2 exits = risky."""
    return escape_routes(x, y, sz, walls) <= 1

# =====================================================================
# UTILITIES
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
def okp(x,y,sz,walls): return 0<=x<sz and 0<=y<sz and (x,y) not in walls
MOVES=["MOVE_UP","MOVE_DOWN","MOVE_LEFT","MOVE_RIGHT"]
DIRS=[(0,-1,"MOVE_UP"),(0,1,"MOVE_DOWN"),(-1,0,"MOVE_LEFT"),(1,0,"MOVE_RIGHT")]
def get_walls(tiles): return {(t["x"],t["y"]) for t in tiles if t.get("type")=="wall"}

def smart_move(mx,my,tx,ty,sz,walls,sr,enemies,avoid_en,cw,turn,mt):
    best_a,best_s=None,-9999
    prev_pos=positions[-2] if len(positions)>=2 else None
    for a in MOVES:
        nx,ny=apply(mx,my,a)
        if not okp(nx,ny,sz,walls): continue
        s=(manhattan(mx,my,tx,ty)-manhattan(nx,ny,tx,ty))*8
        if in_safe(nx,ny,sz,sr): s+=8
        else: s-=20
        if avoid_en:
            for e in enemies:
                d=chebyshev(nx,ny,e["x"],e["y"])
                if d<=1: s-=25
                elif d<=2: s-=8
        if prev_pos and (nx,ny)==prev_pos: s-=12
        s-=dist_center(nx,ny,sz)*cw
        # Avoid trap positions
        if is_trap_position(nx,ny,sz,walls): s-=15
        # Prefer future-safe positions
        if future_safe(nx,ny,turn,mt,sz): s+=5
        elif not in_safe(nx,ny,sz,sr): s-=10
        if s>best_s: best_s,best_a=s,a
    return best_a

def flee_from(mx,my,enemies,sz,walls,sr):
    best_a,best_s=None,-9999
    prev_pos=positions[-2] if len(positions)>=2 else None
    for a in MOVES:
        nx,ny=apply(mx,my,a)
        if not okp(nx,ny,sz,walls): continue
        s=sum((chebyshev(nx,ny,e["x"],e["y"])-chebyshev(mx,my,e["x"],e["y"]))*15 for e in enemies)
        if in_safe(nx,ny,sz,sr): s+=10
        else: s-=20
        if prev_pos and (nx,ny)==prev_pos: s-=8
        if is_trap_position(nx,ny,sz,walls): s-=12
        if s>best_s: best_s,best_a=s,a
    return best_a

def my_dmg(me): return 15*(1+me.get("damage_boost_stacks",0))
def eff_dmg(me,en):
    d=my_dmg(me); return d//2 if en.get("is_defending") else d
def can_kill(me,en): return eff_dmg(me,en)>=en["health"]

def want_fight(me,en,adj_n,phase,w):
    if en["health"]<=eff_dmg(me,en): return True
    if en.get("is_defending") and me.get("damage_boost_stacks",0)==0: return False
    if en.get("energy",10)<3: return random.random()<w["helpless_bonus"]
    sc=w["aggression"]
    if me["health"]>en["health"]: sc+=0.15
    if me["energy"]>en.get("energy",10): sc+=0.12
    if me.get("damage_boost_stacks",0)>=1: sc+=0.25
    if me.get("shield_charges",0)>=1: sc+=0.15
    if adj_n>=2: sc-=0.25
    if me["health"]<35: sc-=0.20
    if me["energy"]<5: sc-=0.15
    # Score awareness: if losing, be MORE aggressive
    enemy_scores = [e.get("score", 0) for e in [en]]
    if me.get("score", 0) < en.get("score", 0): sc += 0.15
    return sc>0.45  # lower threshold = more aggressive

def want_flee(me,adj,w):
    hp=me["health"]/100
    if hp<=w["flee_health_pct"]: return True
    if len(adj)>=2 and hp<w["flee_health_pct"]+0.18: return True
    if me["health"]<40 and me["energy"]<w["min_energy"]+1: return True
    return False

def score_target(t,mx,my,energy,health,me,phase,sz,sr,w,turn,mt):
    tx,ty=t["x"],t["y"]; d=manhattan(mx,my,tx,ty)
    hr,pu=t.get("has_resource",False),t.get("power_up")
    if energy<d+2: return -1
    if not in_safe(tx,ty,sz,sr) and phase!="early": return -1
    if not future_safe(tx,ty,turn,mt,sz,8) and phase!="early": return -1
    s=0.0
    if pu:
        if pu=="damage_boost": s=20+w["dmg_boost_pri"]*30
        elif pu=="shield": s=15+w["shield_pri"]*25+(12 if health<50 else 0)
        elif pu=="energy_pack": s=15+w["energy_pack_pri"]*25+(18 if energy<10 else 0)
        elif pu=="speed_boost": s=10+w["speed_pri"]*20
        elif pu=="vision_boost": s=5+w["vision_pri"]*15
    elif hr: s=10+w["resource_pri"]*20+(8 if energy<10 else 0)+(5 if phase=="early" else 0)
    else: return -1
    s-=d*2.5
    if in_safe(tx,ty,sz,sr): s+=4
    return s

# =====================================================================
# MAIN DECISION ENGINE
# =====================================================================
def choose_action(state):
    global prev_action,consecutive_waits,escaping_zone
    me=state["self"]; mx,my,energy,health=me["x"],me["y"],me["energy"],me["health"]
    turn,mt=state["turn"],state["max_turns"]
    sz,sr=state["arena_size"],state["safe_zone_radius"]
    enemies,tiles=state.get("enemies",[]),state.get("visible_tiles",[])
    pct=turn/mt; phase=get_phase(pct)
    w=get_active_w(phase)
    walls=get_walls(tiles); safe_now=in_safe(mx,my,sz,sr)
    cw=w.get("center_pull",0.4)
    epos={(e["x"],e["y"]) for e in enemies}
    positions.append((mx,my))
    if len(positions)>12: positions.pop(0)
    adj=[e for e in enemies if chebyshev(mx,my,e["x"],e["y"])<=1]
    nearby=[e for e in enemies if 1<chebyshev(mx,my,e["x"],e["y"])<=3]
    my_tile=[t for t in tiles if t["x"]==mx and t["y"]==my]
    on_collect=any(t.get("has_resource") or t.get("power_up") for t in my_tile)
    other_tgt=[t for t in tiles if(t.get("has_resource")or t.get("power_up"))and not(t["x"]==mx and t["y"]==my)]
    near_pu=[t for t in other_tgt if t.get("power_up") and manhattan(mx,my,t["x"],t["y"])<=3]

    taking_dmg=zone_dmg(mx,my,sz,sr)>0
    dist_s,esc_a=bfs_to_safe(mx,my,sz,walls,sr,epos)
    tud=turns_til_danger(mx,my,turn,mt,sz,sr)
    zone_urgent=taking_dmg
    zone_soon=not safe_now and tud<=dist_s+4
    f_safe=future_safe(mx,my,turn,mt,sz)

    # If safe now, cancel escape mode
    if safe_now: escaping_zone=False

    def respond(a,e,m):
        global prev_action,consecutive_waits
        prev_action=a; consecutive_waits=consecutive_waits+1 if a=="WAIT" else 0
        return MoveResponse(action=a,emoji=e,mood=m)

    # === 1. ZONE EMERGENCY ===
    if zone_urgent:
        escaping_zone=True
        if health<=15 and energy>=2 and dist_s>1:
            return respond("DEFEND","🛡️","zone shield")
        if esc_a and energy>=1: return respond(esc_a,"🚨","ESCAPE")
        _,e2=bfs_to_safe(mx,my,sz,walls,sr,None)
        if e2 and energy>=1: return respond(e2,"🚨","ESCAPE!")
        if energy>=2: return respond("DEFEND","🛡️","zone survive")

    # === 2. ZONE ESCAPE (committed — don't stop until safe!) ===
    if escaping_zone or zone_soon:
        escaping_zone=True
        if on_collect and energy>=2 and dist_s<=2:
            return respond("COLLECT","💰","grab+run")
        if esc_a and energy>=1: return respond(esc_a,"⚠️","escaping")
        _,e2=bfs_to_safe(mx,my,sz,walls,sr,None)
        if e2 and energy>=1: return respond(e2,"⚠️","escaping!")

    # === 3. FUTURE ZONE ===
    if not f_safe and phase!="early":
        if on_collect and energy>=2: return respond("COLLECT","💰","collect+move")
        cx,cy=int(center_of(sz)[0]),int(center_of(sz)[1])
        a=smart_move(mx,my,cx,cy,sz,walls,sr,enemies,True,cw,turn,mt)
        if a and energy>=1: return respond(a,"📍","reposition")

    # === 4. COLLECT ===
    if on_collect and energy>=2:
        return respond("COLLECT","💰","collecting")

    # === 5. FREE KILLS (moved up! always take easy kills for score) ===
    if adj and energy>=3 and any(can_kill(me,e) for e in adj):
        return respond("ATTACK","💀","executing")

    # === 6. NEARBY POWER-UP ===
    if near_pu and energy>=4 and not adj:
        safe_pu=[t for t in near_pu if in_safe(t["x"],t["y"],sz,sr) or phase=="early"]
        if safe_pu:
            bp=max(safe_pu,key=lambda t:score_target(t,mx,my,energy,health,me,phase,sz,sr,w,turn,mt))
            if score_target(bp,mx,my,energy,health,me,phase,sz,sr,w,turn,mt)>10:
                a=smart_move(mx,my,bp["x"],bp["y"],sz,walls,sr,enemies,False,cw,turn,mt)
                if a: return respond(a,"✨",f"grab {bp['power_up']}")

    # === 7. FLEE (only when really needed — we're aggressive now) ===
    if adj and want_flee(me,adj,w):
        if health<=20 and energy>=2: return respond("DEFEND","🛡️","emergency")
        a=flee_from(mx,my,enemies,sz,walls,sr)
        if a and energy>=1: return respond(a,"💨","retreat")

    # === 8. HIT & RUN ===
    if prev_action=="ATTACK" and adj and energy>=1:
        # If we can kill next hit, stay and fight!
        if any(can_kill(me,e) for e in adj) and energy>=3:
            return respond("ATTACK","💀","finish!")
        a=flee_from(mx,my,enemies,sz,walls,sr)
        if a: return respond(a,"💨","hit&run")

    # === 9. AGGRESSIVE COMBAT (the big change — fight more!) ===
    if adj and energy>=4:
        for en in adj:
            if want_fight(me,en,len(adj),phase,w):
                if en.get("is_defending") and w["wait_defense"]>0.5 and me.get("damage_boost_stacks",0)==0:
                    return respond("WAIT","⏳","wait def")
                return respond("ATTACK","⚔️","ATTACK!")
        # Even if we don't "want" to fight, attack if we're losing on score
        if len(adj)==1 and me.get("score",0)<adj[0].get("score",0) and energy>=4 and health>40:
            return respond("ATTACK","⚔️","fight for score!")
        if energy>=1:
            a=flee_from(mx,my,enemies,sz,walls,sr)
            if a: return respond(a,"💨","disengage")

    # === 10. HUNT ENEMIES (more aggressive — chase them!) ===
    if nearby and energy>=5 and health>40:
        md=my_dmg(me)
        # Hunt anyone we can kill in 3 hits or less
        hunt=[e for e in nearby if e["health"]<=md*3]
        # Also hunt anyone with lower health than us
        if not hunt: hunt=[e for e in nearby if e["health"]<health and me.get("damage_boost_stacks",0)>=1]
        if hunt and random.random()<w["kill_chase"]:
            t=min(hunt,key=lambda e:e["health"])
            a=smart_move(mx,my,t["x"],t["y"],sz,walls,sr,enemies,False,cw,turn,mt)
            if a:
                nx,ny=apply(mx,my,a)
                if in_safe(nx,ny,sz,sr) or phase=="early":
                    return respond(a,"🐺","hunting!")

    # === 11. CHASE TARGETS ===
    if other_tgt and energy>=3:
        scored=[(score_target(t,mx,my,energy,health,me,phase,sz,sr,w,turn,mt),t) for t in other_tgt]
        scored=[(s,t) for s,t in scored if s>0]
        scored.sort(key=lambda x:x[0],reverse=True)
        if scored:
            best=scored[0][1]; av=health<50
            a=smart_move(mx,my,best["x"],best["y"],sz,walls,sr,enemies,av,cw,turn,mt)
            if a:
                nx,ny=apply(mx,my,a)
                if in_safe(nx,ny,sz,sr) or phase=="early":
                    return respond(a,"🎯",f"get {best.get('power_up','resource')}")

    # === 12. ENERGY ===
    if energy<w.get("min_energy",3):
        if consecutive_waits>=4:
            for a in MOVES:
                nx,ny=apply(mx,my,a)
                if okp(nx,ny,sz,walls) and in_safe(nx,ny,sz,sr) and not is_trap_position(nx,ny,sz,walls):
                    return respond(a,"🔍","break loop")
        return respond("WAIT","🔋","recharge")

    # === 13. EXPLORE (safe zone only, avoid traps) ===
    best_a,best_s=None,-9999
    shuffled=list(MOVES); random.shuffle(shuffled)
    pp=positions[-2] if len(positions)>=2 else None
    for a in shuffled:
        nx,ny=apply(mx,my,a)
        if not okp(nx,ny,sz,walls): continue
        if safe_now and not in_safe(nx,ny,sz,sr): continue
        s=0.0
        if pp and (nx,ny)==pp: s-=12
        if in_safe(nx,ny,sz,sr): s+=8
        else: s-=15
        if future_safe(nx,ny,turn,mt,sz): s+=5
        else: s-=10
        if is_trap_position(nx,ny,sz,walls): s-=10
        s-=dist_center(nx,ny,sz)*cw
        for e in enemies:
            # In aggressive mode, move TOWARD enemies not away!
            if chebyshev(nx,ny,e["x"],e["y"])<=3 and health>50 and energy>=4:
                s+=3  # slight pull toward enemies
            elif chebyshev(nx,ny,e["x"],e["y"])<=2: s-=5
        s+=random.random()*4
        if s>best_s: best_s,best_a=s,a
    if best_a and energy>=1: return respond(best_a,"🔍","scout")
    return respond("WAIT","😴","idle")

# =====================================================================
# HTTP ENDPOINTS
# =====================================================================
@app.post("/move")
async def move(state:dict[str,Any])->MoveResponse:
    global positions,prev_action,consecutive_waits,escaping_zone
    mid=state["match_id"]
    if mid!=current_match.get("match_id"):
        if current_match.get("match_id"): finish_match()
        positions,prev_action,consecutive_waits=[],""  ,0
        escaping_zone=False
        new_match(mid)
    track_turn(state)
    r=choose_action(state)
    if ARENA_URL:
        try:
            async with httpx.AsyncClient() as c:
                await c.post(f"{ARENA_URL}/arena/bot-update",
                    json={"bot_id":state["self"]["bot_id"],"status":r.mood.upper(),
                          "message":f"T{state['turn']}:{r.action}","color":BOT_COLOR},timeout=0.3)
        except: pass
    return r

@app.get("/health")
async def hc(): return {"status":"ok","bot_id":BOT_ID}

@app.get("/stats/json")
async def sj():
    rec=match_history[-20:]; ci=None
    if current_match.get("match_id"):
        ci={"turns":current_match["turns_played"],"max_turns":current_match["max_turns"],
            "health":current_match["last_health"],"score":current_match["last_score"],
            "kills":current_match["kills"],
            "enemy_scores":{b:i["last_score"] for b,i in enemy_tracker.items()}}
    return {"matches":total_matches,"won":total_won,"lost":total_lost,"died":total_died,
        "current_match":ci,"phase_weights":phase_weights,
        "recent":[{"outcome":m.get("outcome"),"score":m["last_score"],"reward":m.get("reward",0),
            "turns":m["turns_played"],"kills":m["kills"],"enemy_scores":m.get("enemy_scores",{})} for m in rec]}

@app.get("/stats",response_class=HTMLResponse)
async def sp():
    wr=f"{total_won/max(1,total_matches)*100:.0f}" if total_matches else "0"
    rec=match_history[-20:]
    ar=sum(m.get("reward",0) for m in rec)/max(1,len(rec)) if rec else 0
    avs=sum(m["last_score"] for m in rec)/max(1,len(rec)) if rec else 0
    cm=""
    if current_match.get("match_id"):
        hp=current_match["last_health"];hc="#4ade80" if hp>60 else("#fbbf24" if hp>30 else "#ef4444")
        osc=current_match["last_score"]
        ens=", ".join(f"{b[:8]}:{i['last_score']}" for b,i in enemy_tracker.items())
        ss=""
        if enemy_tracker:
            me2=max(i["last_score"] for i in enemy_tracker.values())
            ss='<span style="color:#4ade80">📈 WIN</span>' if osc>me2 else('<span style="color:#fbbf24">🤝 TIE</span>' if osc==me2 else '<span style="color:#ef4444">📉 BEHIND</span>')
        cm=f'<div class="card live"><h2>🔴 LIVE</h2><div class="sr"><span>Turn</span><strong>{current_match["turns_played"]}/{current_match["max_turns"]}</strong></div><div class="sr"><span>HP</span><div class="bar-bg"><div class="bar" style="width:{hp}%;background:{hc}"></div></div><strong>{hp}</strong></div><div class="sr"><span>Score</span><strong>{osc}</strong> {ss}</div><div class="sr"><span>Enemy</span><strong style="font-size:.8em">{ens or"none"}</strong></div><div class="sr"><span>Kills</span><strong>{current_match["kills"]}</strong></div></div>'
    rows=""
    for m in reversed(rec[-15:]):
        o=m.get("outcome","?")
        if o=="won_elimination":i,c,l="🏆","won","WON(elim)"
        elif o=="won_score":i,c,l="🥇","won","WON(score)"
        elif o=="tied":i,c,l="🤝","tied","TIED"
        elif o=="lost_score":i,c,l="🥈","lost","LOST"
        else:i,c,l="💀","died","DIED"
        es=",".join(f"{v}" for v in m.get("enemy_scores",{}).values()) or"-"
        rows+=f'<tr class="{c}"><td>{i} {l}</td><td>{m["last_score"]}</td><td>{es}</td><td>{m["kills"]}</td><td>{m["turns_played"]}/{m["max_turns"]}</td><td>{m.get("reward",0):.0f}</td></tr>'
    # Phase weights display
    pw_html=""
    for pname in ("early","mid","late"):
        pw=phase_weights.get(pname,{})
        bars=""
        for k,v in pw.items():
            lo,hi=BOUNDS.get(k,(0,1));p=((v-lo)/(hi-lo))*100 if hi>lo else 50
            bars+=f'<div class="wr"><span class="wl">{k}</span><div class="bar-bg sm"><div class="bar ac" style="width:{p:.0f}%"></div></div><span class="wv">{v:.2f}</span></div>'
        emoji="🌱" if pname=="early" else("⚔️" if pname=="mid" else "🏁")
        pw_html+=f'<div class="card"><h2>{emoji} {pname.upper()} weights</h2>{bars}</div>'

    return HTMLResponse(content=f'''<!DOCTYPE html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta http-equiv="refresh" content="4"><title>🤖 v8</title><style>*{{margin:0;padding:0;box-sizing:border-box}}body{{font-family:-apple-system,sans-serif;background:#0f172a;color:#e2e8f0;padding:16px}}h1{{text-align:center;font-size:1.6em;margin-bottom:16px;color:#38bdf8}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(110px,1fr));gap:10px;margin-bottom:16px}}.kpi{{background:#1e293b;border-radius:10px;padding:12px;text-align:center}}.kpi .num{{font-size:1.8em;font-weight:bold}}.kpi .label{{font-size:.75em;color:#94a3b8;margin-top:2px}}.green{{color:#4ade80}}.red{{color:#ef4444}}.blue{{color:#38bdf8}}.yellow{{color:#fbbf24}}.purple{{color:#a78bfa}}.card{{background:#1e293b;border-radius:10px;padding:14px;margin-bottom:14px}}.card h2{{font-size:1em;margin-bottom:10px;color:#94a3b8}}.card.live{{border:1px solid #ef4444}}.sr{{display:flex;align-items:center;gap:6px;margin-bottom:5px}}.sr span{{width:80px;font-size:.8em;color:#94a3b8}}.sr strong{{font-size:.9em}}.bar-bg{{flex:1;background:#334155;border-radius:4px;height:12px;overflow:hidden}}.bar-bg.sm{{height:7px}}.bar{{height:100%;border-radius:4px;background:#38bdf8}}.bar.ac{{background:#818cf8}}table{{width:100%;border-collapse:collapse;font-size:.8em}}th{{text-align:left;color:#94a3b8;padding:6px;border-bottom:1px solid #334155}}td{{padding:6px;border-bottom:1px solid #1e293b}}tr.won td{{color:#4ade80}}tr.lost td{{color:#fbbf24}}tr.died td{{color:#ef4444}}tr.tied td{{color:#94a3b8}}.wr{{display:flex;align-items:center;gap:4px;margin-bottom:3px}}.wl{{width:100px;font-size:.65em;color:#94a3b8}}.wv{{width:36px;font-size:.65em;text-align:right}}.phases{{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:12px}}</style></head><body>
<h1>🤖 RoboKova v8 — Aggressive Learner</h1>
<div class="grid"><div class="kpi"><div class="num blue">{total_matches}</div><div class="label">Matches</div></div><div class="kpi"><div class="num green">{total_won}</div><div class="label">Won</div></div><div class="kpi"><div class="num yellow">{total_lost}</div><div class="label">Lost</div></div><div class="kpi"><div class="num red">{total_died}</div><div class="label">Died</div></div><div class="kpi"><div class="num {"green" if int(wr)>=50 else "red"}">{wr}%</div><div class="label">Win Rate</div></div><div class="kpi"><div class="num purple">{avs:.0f}</div><div class="label">Avg Score</div></div></div>
{cm}
<div class="card"><h2>📜 History</h2><table><tr><th>Result</th><th>Score</th><th>Enemy</th><th>Kills</th><th>Turns</th><th>Reward</th></tr>{rows or'<tr><td colspan="6" style="color:#64748b">No matches</td></tr>'}</table></div>
<div class="phases">{pw_html}</div>
<p style="text-align:center;color:#475569;margin-top:16px;font-size:.7em">Auto-refresh 4s · {len(match_history)} matches · Weights saved to disk</p></body></html>''')
