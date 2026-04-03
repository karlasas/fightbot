"""RoboKova Self-Learning Bot v9

CRITICAL FIX: Supports multiple concurrent matches!
- Each match_id gets its own state (positions, enemies, etc.)
- No more cross-match contamination
- Stale matches auto-finish after 60 seconds of inactivity
- Corrupted weights reset to aggressive defaults
- All v8 features preserved (phases, persistence, wall awareness, etc.)

Run: uvicorn bot:app --host 0.0.0.0 --port 5001 --reload
"""

from __future__ import annotations
import json, os, random, time
from collections import deque
from pathlib import Path
from typing import Any
import httpx
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

app = FastAPI(title="RoboKova Bot v9")
BOT_ID = os.environ.get("BOT_ID", "v9")
BOT_COLOR = os.environ.get("BOT_COLOR", "#e94560")
ARENA_URL = os.environ.get("ARENA_URL", "")
WEIGHTS_FILE = Path("learned_weights_v9.json")  # new file, ignores corrupted v8
STALE_TIMEOUT = 60  # seconds before considering a match finished

class MoveResponse(BaseModel):
    action: str
    emoji: str = ""
    mood: str = ""

# =====================================================================
# PHASE-SPECIFIC WEIGHTS (reset to aggressive defaults)
# =====================================================================
DEFAULT_WEIGHTS = {
    "early": {
        "aggression": 0.40, "flee_health_pct": 0.20, "defend_pref": 0.25,
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
    "aggression": (0.15, 0.90), "flee_health_pct": (0.10, 0.40),
    "defend_pref": (0.10, 0.70), "kill_chase": (0.30, 0.95),
    "helpless_bonus": (0.45, 0.95), "wait_defense": (0.30, 0.85),
    "resource_pri": (0.40, 0.95), "energy_pack_pri": (0.45, 0.95),
    "dmg_boost_pri": (0.55, 0.99), "shield_pri": (0.35, 0.95),
    "speed_pri": (0.15, 0.70), "vision_pri": (0.05, 0.50),
    "min_energy": (1.5, 6.0), "center_pull": (0.10, 0.65),
}
LR = 0.08
NOISE = 0.025

phase_weights: dict[str, dict] = {}

def load_weights():
    global phase_weights
    if WEIGHTS_FILE.exists():
        try:
            phase_weights = json.loads(WEIGHTS_FILE.read_text())
            # Validate loaded weights have all keys
            for p in ("early", "mid", "late"):
                if p not in phase_weights or len(phase_weights[p]) < 10:
                    phase_weights = {p: dict(w) for p, w in DEFAULT_WEIGHTS.items()}
                    return
            print(f"Loaded weights from {WEIGHTS_FILE}")
            return
        except: pass
    phase_weights = {p: dict(w) for p, w in DEFAULT_WEIGHTS.items()}

def save_weights():
    try: WEIGHTS_FILE.write_text(json.dumps(phase_weights, indent=2))
    except: pass

load_weights()

def get_w(phase): return phase_weights.get(phase, DEFAULT_WEIGHTS["mid"])
def get_active_w(phase):
    w = dict(get_w(phase))
    if len(match_history) < 30:
        for k in w:
            lo, hi = BOUNDS.get(k, (0, 1))
            w[k] = max(lo, min(hi, w[k] + random.gauss(0, NOISE * 0.3)))
    return w

# =====================================================================
# PER-MATCH STATE — each concurrent match has its own state
# =====================================================================
class MatchState:
    def __init__(self, match_id):
        self.match_id = match_id
        self.last_update = time.time()
        self.positions: list[tuple[int, int]] = []
        self.prev_action = ""
        self.consecutive_waits = 0
        self.escaping_zone = False
        self.enemy_tracker: dict[str, dict] = {}
        self.phase_perf = {p: {"score_gained": 0, "dmg_taken": 0, "kills": 0, "turns": 0}
                           for p in ("early", "mid", "late")}
        self.data = {
            "match_id": match_id, "turns_played": 0, "max_turns": 100, "num_bots": 2,
            "last_health": 100, "last_score": 0, "total_damage_taken": 0,
            "total_score_gained": 0, "kills": 0, "health_history": [],
            "enemies_alive_last_turn": 0,
        }

active_matches: dict[str, MatchState] = {}
match_history: list[dict] = []
total_matches = total_won = total_lost = total_died = 0

def get_phase(pct):
    if pct < 0.30: return "early"
    if pct < 0.65: return "mid"
    return "late"

def cleanup_stale():
    """Finish matches that haven't been updated in a while."""
    now = time.time()
    stale = [mid for mid, ms in active_matches.items() if now - ms.last_update > STALE_TIMEOUT]
    for mid in stale:
        finish_match(mid)
        del active_matches[mid]

def track_turn(ms: MatchState, state: dict):
    me = state["self"]; hp, sc = me["health"], me["score"]
    d = ms.data
    ph, ps = d["last_health"], d["last_score"]
    dmg, sg = max(0, ph - hp), max(0, sc - ps)
    pct = state["turn"] / state["max_turns"]
    phase = get_phase(pct)

    d["turns_played"] += 1; d["max_turns"] = state["max_turns"]
    d["num_bots"] = state.get("num_bots", 2)
    d["total_damage_taken"] += dmg; d["total_score_gained"] += sg
    d["last_health"] = hp; d["last_score"] = sc
    d["enemies_alive_last_turn"] = len(state.get("enemies", []))
    d["health_history"].append(hp)
    if len(d["health_history"]) > 10: d["health_history"].pop(0)
    if sg >= 30: d["kills"] += 1

    pp = ms.phase_perf[phase]
    pp["score_gained"] += sg; pp["dmg_taken"] += dmg; pp["turns"] += 1
    if sg >= 30: pp["kills"] += 1

    for e in state.get("enemies", []):
        ms.enemy_tracker[e["bot_id"]] = {
            "last_score": e.get("score", 0), "last_health": e.get("health", 100),
            "last_seen_turn": state["turn"],
        }
    ms.last_update = time.time()

def detect_outcome(ms: MatchState):
    d = ms.data; t, mt = d["turns_played"], d["max_turns"]
    sc, hp = d["last_score"], d["last_health"]
    hh = d.get("health_history", [])
    el = d["enemies_alive_last_turn"]
    es = {b: i["last_score"] for b, i in ms.enemy_tracker.items()}
    he = max(es.values()) if es else 0
    hd = len(hh) >= 3 and hh[-1] < hh[0] - 10

    if t >= mt - 2:
        return "won_score" if sc > he else ("tied" if sc == he else "lost_score")
    if hp > 20 and not hd and (el == 0 or hp > 40): return "won_elimination"
    if (hd and hp < 30) or hp <= 15: return "died"
    if el > 0 and hd: return "died"
    if t < mt * 0.25 and t < 15: return "died"
    return "won_elimination" if hp > 50 else "died"

def finish_match(match_id: str):
    global total_matches, total_won, total_lost, total_died
    ms = active_matches.get(match_id)
    if not ms or ms.data["turns_played"] < 3: return  # ignore super-short matches

    total_matches += 1
    o = detect_outcome(ms); d = ms.data
    d["outcome"] = o
    d["enemy_scores"] = {b: i["last_score"] for b, i in ms.enemy_tracker.items()}

    if o.startswith("won"): total_won += 1
    elif o in ("lost_score", "tied"): total_lost += 1
    else: total_died += 1

    sp = d["turns_played"] / max(1, d["max_turns"])
    r = d["last_score"] + sp * 40
    if o == "won_elimination": r += 90
    elif o == "won_score": r += 70
    elif o == "tied": r += 30
    elif o == "lost_score": r -= 10
    else: r -= 50 * (1 - sp)
    dm, sg = d["total_damage_taken"], d["total_score_gained"]
    if dm > 0: r += min((sg / dm) * 8, 25)
    elif sg > 0: r += 25
    r += d["kills"] * 12
    r *= 1.0 + (d.get("num_bots", 2) - 2) * 0.08
    d["reward"] = round(r, 1)
    match_history.append(dict(d))

    if len(match_history) >= 3:
        base_bonus = {"won_elimination": 90, "won_score": 70, "tied": 30, "lost_score": -10}.get(o, -50 * (1 - sp))
        for pn, pp in ms.phase_perf.items():
            if pp["turns"] == 0: continue
            pr = pp["score_gained"] - pp["dmg_taken"] * 0.5 + pp["kills"] * 15
            pr += base_bonus * (pp["turns"] / max(1, d["turns_played"]))
            learn_phase(pn, pr)
        save_weights()

def learn_phase(phase_name, reward):
    if not match_history: return
    avg = sum(m.get("reward", 0) for m in match_history[-20:]) / min(20, len(match_history))
    adv = max(-0.8, min(0.8, (reward - avg * 0.3) / max(abs(avg * 0.3), 1)))
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

def zone_dmg(mx, my, sz, sr): return 5 if dist_center(mx, my, sz) > sr else 0
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
def escape_routes(x,y,sz,walls):
    return sum(1 for dx,dy,_ in DIRS if 0<=x+dx<sz and 0<=y+dy<sz and (x+dx,y+dy) not in walls)
def is_trap(x,y,sz,walls): return escape_routes(x,y,sz,walls)<=1

def smart_move(mx,my,tx,ty,sz,walls,sr,enemies,avoid_en,cw,turn,mt,ms):
    best_a,best_s=None,-9999
    pp=ms.positions[-2] if len(ms.positions)>=2 else None
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
        if pp and (nx,ny)==pp: s-=12
        s-=dist_center(nx,ny,sz)*cw
        if is_trap(nx,ny,sz,walls): s-=15
        if future_safe(nx,ny,turn,mt,sz): s+=5
        if s>best_s: best_s,best_a=s,a
    return best_a

def flee_move(mx,my,enemies,sz,walls,sr,ms):
    best_a,best_s=None,-9999
    pp=ms.positions[-2] if len(ms.positions)>=2 else None
    for a in MOVES:
        nx,ny=apply(mx,my,a)
        if not okp(nx,ny,sz,walls): continue
        s=sum((chebyshev(nx,ny,e["x"],e["y"])-chebyshev(mx,my,e["x"],e["y"]))*15 for e in enemies)
        if in_safe(nx,ny,sz,sr): s+=10
        else: s-=20
        if pp and (nx,ny)==pp: s-=8
        if is_trap(nx,ny,sz,walls): s-=12
        if s>best_s: best_s,best_a=s,a
    return best_a

def my_dmg(me): return 15*(1+me.get("damage_boost_stacks",0))
def eff_dmg(me,en): d=my_dmg(me); return d//2 if en.get("is_defending") else d
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
    if me.get("score",0)<en.get("score",0): sc+=0.15
    return sc>0.45

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
    s-=d*2.5; s+=4 if in_safe(tx,ty,sz,sr) else 0
    return s

# =====================================================================
# MAIN DECISION ENGINE
# =====================================================================
def choose_action(ms: MatchState, state: dict) -> MoveResponse:
    me=state["self"]; mx,my,energy,health=me["x"],me["y"],me["energy"],me["health"]
    turn,mt=state["turn"],state["max_turns"]
    sz,sr=state["arena_size"],state["safe_zone_radius"]
    enemies,tiles=state.get("enemies",[]),state.get("visible_tiles",[])
    pct=turn/mt; phase=get_phase(pct)
    w=get_active_w(phase)
    walls=get_walls(tiles); safe_now=in_safe(mx,my,sz,sr)
    cw=w.get("center_pull",0.4)
    epos={(e["x"],e["y"]) for e in enemies}

    ms.positions.append((mx,my))
    if len(ms.positions)>12: ms.positions.pop(0)
    if safe_now: ms.escaping_zone=False

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

    def respond(a,e,m):
        ms.prev_action=a
        ms.consecutive_waits=ms.consecutive_waits+1 if a=="WAIT" else 0
        return MoveResponse(action=a,emoji=e,mood=m)

    # 1. ZONE EMERGENCY
    if zone_urgent:
        ms.escaping_zone=True
        if health<=15 and energy>=2 and dist_s>1: return respond("DEFEND","🛡️","zone shield")
        if esc_a and energy>=1: return respond(esc_a,"🚨","ESCAPE")
        _,e2=bfs_to_safe(mx,my,sz,walls,sr,None)
        if e2 and energy>=1: return respond(e2,"🚨","ESCAPE!")
        if energy>=2: return respond("DEFEND","🛡️","zone survive")

    # 2. COMMITTED ESCAPE
    if ms.escaping_zone or zone_soon:
        ms.escaping_zone=True
        if on_collect and energy>=2 and dist_s<=2: return respond("COLLECT","💰","grab+run")
        if esc_a and energy>=1: return respond(esc_a,"⚠️","escaping")
        _,e2=bfs_to_safe(mx,my,sz,walls,sr,None)
        if e2 and energy>=1: return respond(e2,"⚠️","escaping!")

    # 3. FUTURE ZONE
    if not future_safe(mx,my,turn,mt,sz) and phase!="early":
        if on_collect and energy>=2: return respond("COLLECT","💰","collect+move")
        cx,cy=int(center_of(sz)[0]),int(center_of(sz)[1])
        a=smart_move(mx,my,cx,cy,sz,walls,sr,enemies,True,cw,turn,mt,ms)
        if a and energy>=1: return respond(a,"📍","reposition")

    # 4. COLLECT
    if on_collect and energy>=2: return respond("COLLECT","💰","collecting")

    # 5. FREE KILLS
    if adj and energy>=3 and any(can_kill(me,e) for e in adj):
        return respond("ATTACK","💀","executing")

    # 6. NEARBY POWER-UP
    if near_pu and energy>=4 and not adj:
        safe_pu=[t for t in near_pu if in_safe(t["x"],t["y"],sz,sr) or phase=="early"]
        if safe_pu:
            bp=max(safe_pu,key=lambda t:score_target(t,mx,my,energy,health,me,phase,sz,sr,w,turn,mt))
            if score_target(bp,mx,my,energy,health,me,phase,sz,sr,w,turn,mt)>10:
                a=smart_move(mx,my,bp["x"],bp["y"],sz,walls,sr,enemies,False,cw,turn,mt,ms)
                if a: return respond(a,"✨",f"grab {bp['power_up']}")

    # 7. FLEE
    if adj and want_flee(me,adj,w):
        if health<=20 and energy>=2: return respond("DEFEND","🛡️","emergency")
        a=flee_move(mx,my,enemies,sz,walls,sr,ms)
        if a and energy>=1: return respond(a,"💨","retreat")

    # 8. HIT & RUN
    if ms.prev_action=="ATTACK" and adj and energy>=1:
        if any(can_kill(me,e) for e in adj) and energy>=3:
            return respond("ATTACK","💀","finish!")
        a=flee_move(mx,my,enemies,sz,walls,sr,ms)
        if a: return respond(a,"💨","hit&run")

    # 9. AGGRESSIVE COMBAT
    if adj and energy>=4:
        for en in adj:
            if want_fight(me,en,len(adj),phase,w):
                if en.get("is_defending") and w["wait_defense"]>0.5 and me.get("damage_boost_stacks",0)==0:
                    return respond("WAIT","⏳","wait def")
                return respond("ATTACK","⚔️","ATTACK!")
        if len(adj)==1 and me.get("score",0)<adj[0].get("score",0) and energy>=4 and health>40:
            return respond("ATTACK","⚔️","fight for score!")
        if energy>=1:
            a=flee_move(mx,my,enemies,sz,walls,sr,ms)
            if a: return respond(a,"💨","disengage")

    # 10. HUNT ENEMIES
    if nearby and energy>=5 and health>40:
        md=my_dmg(me)
        hunt=[e for e in nearby if e["health"]<=md*3]
        if not hunt: hunt=[e for e in nearby if e["health"]<health and me.get("damage_boost_stacks",0)>=1]
        if hunt and random.random()<w["kill_chase"]:
            t=min(hunt,key=lambda e:e["health"])
            a=smart_move(mx,my,t["x"],t["y"],sz,walls,sr,enemies,False,cw,turn,mt,ms)
            if a:
                nx,ny=apply(mx,my,a)
                if in_safe(nx,ny,sz,sr) or phase=="early": return respond(a,"🐺","hunting!")

    # 11. CHASE TARGETS
    if other_tgt and energy>=3:
        scored=[(score_target(t,mx,my,energy,health,me,phase,sz,sr,w,turn,mt),t) for t in other_tgt]
        scored=[(s,t) for s,t in scored if s>0]; scored.sort(key=lambda x:x[0],reverse=True)
        if scored:
            best=scored[0][1]; av=health<50
            a=smart_move(mx,my,best["x"],best["y"],sz,walls,sr,enemies,av,cw,turn,mt,ms)
            if a:
                nx,ny=apply(mx,my,a)
                if in_safe(nx,ny,sz,sr) or phase=="early":
                    return respond(a,"🎯",f"get {best.get('power_up','resource')}")

    # 12. ENERGY
    if energy<w.get("min_energy",3):
        if ms.consecutive_waits>=4:
            for a in MOVES:
                nx,ny=apply(mx,my,a)
                if okp(nx,ny,sz,walls) and in_safe(nx,ny,sz,sr) and not is_trap(nx,ny,sz,walls):
                    return respond(a,"🔍","break loop")
        return respond("WAIT","🔋","recharge")

    # 13. EXPLORE
    best_a,best_s=None,-9999
    shuffled=list(MOVES); random.shuffle(shuffled)
    pp=ms.positions[-2] if len(ms.positions)>=2 else None
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
        if is_trap(nx,ny,sz,walls): s-=10
        s-=dist_center(nx,ny,sz)*cw
        for e in enemies:
            if chebyshev(nx,ny,e["x"],e["y"])<=3 and health>50 and energy>=4: s+=3
            elif chebyshev(nx,ny,e["x"],e["y"])<=2: s-=5
        s+=random.random()*4
        if s>best_s: best_s,best_a=s,a
    if best_a and energy>=1: return respond(best_a,"🔍","scout")
    return respond("WAIT","😴","idle")

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
    track_turn(ms, state)
    r = choose_action(ms, state)

    if ARENA_URL:
        try:
            async with httpx.AsyncClient() as c:
                await c.post(f"{ARENA_URL}/arena/bot-update",
                    json={"bot_id": state["self"]["bot_id"], "status": r.mood.upper(),
                          "message": f"T{state['turn']}:{r.action}", "color": BOT_COLOR}, timeout=0.3)
        except: pass
    return r

@app.get("/health")
async def hc(): return {"status": "ok", "bot_id": BOT_ID}

@app.get("/stats/json")
async def sj():
    rec = match_history[-20:]
    am_info = {mid: {"turns": ms.data["turns_played"], "health": ms.data["last_health"],
                      "score": ms.data["last_score"]} for mid, ms in active_matches.items()}
    return {"matches": total_matches, "won": total_won, "lost": total_lost, "died": total_died,
            "active_matches": len(active_matches), "active_info": am_info,
            "phase_weights": phase_weights,
            "recent": [{"outcome": m.get("outcome"), "score": m["last_score"],
                         "reward": m.get("reward", 0), "turns": m["turns_played"],
                         "kills": m["kills"]} for m in rec]}

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
        am += f'<div class="card live"><h2>🔴 {mid[:8]}</h2><div class="sr"><span>Turn</span><strong>{d["turns_played"]}/{d["max_turns"]}</strong></div><div class="sr"><span>HP</span><div class="bar-bg"><div class="bar" style="width:{hp}%;background:{hc}"></div></div><strong>{hp}</strong></div><div class="sr"><span>Score</span><strong>{d["last_score"]}</strong></div><div class="sr"><span>Enemies</span><strong style="font-size:.75em">{es or "none"}</strong></div></div>'

    rows = ""
    for m in reversed(rec[-15:]):
        o = m.get("outcome", "?")
        if o == "won_elimination": i, c, l = "🏆", "won", "WON(elim)"
        elif o == "won_score": i, c, l = "🥇", "won", "WON(score)"
        elif o == "tied": i, c, l = "🤝", "tied", "TIED"
        elif o == "lost_score": i, c, l = "🥈", "lost", "LOST"
        else: i, c, l = "💀", "died", "DIED"
        es = ",".join(f"{v}" for v in m.get("enemy_scores", {}).values()) or "-"
        rows += f'<tr class="{c}"><td>{i} {l}</td><td>{m["last_score"]}</td><td>{es}</td><td>{m["kills"]}</td><td>{m["turns_played"]}/{m["max_turns"]}</td><td>{m.get("reward",0):.0f}</td></tr>'

    pw = ""
    for pn in ("early", "mid", "late"):
        ww = phase_weights.get(pn, {})
        bars = "".join(f'<div class="wr"><span class="wl">{k}</span><div class="bar-bg sm"><div class="bar ac" style="width:{((v-BOUNDS.get(k,(0,1))[0])/(BOUNDS.get(k,(0,1))[1]-BOUNDS.get(k,(0,1))[0]))*100:.0f}%"></div></div><span class="wv">{v:.2f}</span></div>' for k, v in ww.items())
        em = "🌱" if pn == "early" else ("⚔️" if pn == "mid" else "🏁")
        pw += f'<div class="card"><h2>{em} {pn.upper()}</h2>{bars}</div>'

    return HTMLResponse(content=f'''<!DOCTYPE html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta http-equiv="refresh" content="4"><title>🤖 v9</title><style>*{{margin:0;padding:0;box-sizing:border-box}}body{{font-family:-apple-system,sans-serif;background:#0f172a;color:#e2e8f0;padding:16px}}h1{{text-align:center;font-size:1.5em;margin-bottom:16px;color:#38bdf8}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(100px,1fr));gap:8px;margin-bottom:14px}}.kpi{{background:#1e293b;border-radius:10px;padding:10px;text-align:center}}.kpi .num{{font-size:1.6em;font-weight:bold}}.kpi .label{{font-size:.7em;color:#94a3b8}}.green{{color:#4ade80}}.red{{color:#ef4444}}.blue{{color:#38bdf8}}.yellow{{color:#fbbf24}}.purple{{color:#a78bfa}}.card{{background:#1e293b;border-radius:10px;padding:12px;margin-bottom:12px}}.card h2{{font-size:.95em;margin-bottom:8px;color:#94a3b8}}.card.live{{border:1px solid #ef4444}}.sr{{display:flex;align-items:center;gap:5px;margin-bottom:4px}}.sr span{{width:70px;font-size:.75em;color:#94a3b8}}.sr strong{{font-size:.85em}}.bar-bg{{flex:1;background:#334155;border-radius:4px;height:10px;overflow:hidden}}.bar-bg.sm{{height:6px}}.bar{{height:100%;border-radius:4px;background:#38bdf8}}.bar.ac{{background:#818cf8}}table{{width:100%;border-collapse:collapse;font-size:.75em}}th{{text-align:left;color:#94a3b8;padding:5px;border-bottom:1px solid #334155}}td{{padding:5px;border-bottom:1px solid #1e293b}}tr.won td{{color:#4ade80}}tr.lost td{{color:#fbbf24}}tr.died td{{color:#ef4444}}tr.tied td{{color:#94a3b8}}.phases{{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:10px}}.wr{{display:flex;align-items:center;gap:3px;margin-bottom:2px}}.wl{{width:95px;font-size:.6em;color:#94a3b8}}.wv{{width:32px;font-size:.6em;text-align:right}}</style></head><body>
<h1>🤖 RoboKova v9</h1>
<div class="grid"><div class="kpi"><div class="num blue">{total_matches}</div><div class="label">Matches</div></div><div class="kpi"><div class="num green">{total_won}</div><div class="label">Won</div></div><div class="kpi"><div class="num yellow">{total_lost}</div><div class="label">Lost</div></div><div class="kpi"><div class="num red">{total_died}</div><div class="label">Died</div></div><div class="kpi"><div class="num {"green" if int(wr)>=50 else "red"}">{wr}%</div><div class="label">Win Rate</div></div><div class="kpi"><div class="num purple">{avs:.0f}</div><div class="label">Avg Score</div></div><div class="kpi"><div class="num blue">{len(active_matches)}</div><div class="label">Active</div></div></div>
{am}
<div class="card"><h2>📜 History</h2><table><tr><th>Result</th><th>Score</th><th>Enemy</th><th>Kills</th><th>Turns</th><th>Reward</th></tr>{rows or'<tr><td colspan="6" style="color:#64748b">No matches</td></tr>'}</table></div>
<div class="phases">{pw}</div>
<p style="text-align:center;color:#475569;margin-top:12px;font-size:.65em">Refresh 4s · {len(match_history)} learned · {len(active_matches)} active · Weights on disk</p></body></html>''')
