"""RoboKova Self-Learning Bot v7

FIXES:
- BFS zone escape now routes AROUND enemies, not through them
- Anti-oscillation only penalizes ping-pong (last 1 position), not last 3-6
- HARD RULE: never step outside safe zone when exploring/chasing targets
- Zone escape ignores anti-oscillation entirely (survival > everything)
- When enemy blocks path to safety, tries sideways moves automatically
- Exploration never enters danger zone tiles

Run: uvicorn bot:app --host 0.0.0.0 --port 5001 --reload
"""

from __future__ import annotations
import os, random
from collections import deque
from typing import Any
import httpx
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

app = FastAPI(title="RoboKova Bot v7")
BOT_ID = os.environ.get("BOT_ID", "v7")
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
    "min_energy_reserve": 3.0, "survival_value": 0.65, "zone_prep_timing": 0.55,
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
# MATCH TRACKING
# =====================================================================
match_history: list[dict] = []
current_match: dict = {}
total_matches = total_won = total_lost = total_died = 0
positions: list[tuple[int, int]] = []
prev_action: str = ""
consecutive_waits: int = 0
active_weights: dict = {}
enemy_tracker: dict[str, dict] = {}

def new_match(mid, w):
    global current_match, enemy_tracker
    enemy_tracker = {}
    current_match = {"match_id": mid, "weights_used": dict(w), "turns_played": 0,
        "max_turns": 100, "num_bots": 2, "last_health": 100, "last_score": 0,
        "total_damage_taken": 0, "total_score_gained": 0, "kills": 0,
        "health_history": [], "enemies_alive_last_turn": 0}

def track_turn(state):
    me = state["self"]; hp, sc = me["health"], me["score"]
    ph, ps = current_match.get("last_health", 100), current_match.get("last_score", 0)
    current_match["turns_played"] += 1
    current_match["max_turns"] = state["max_turns"]
    current_match["num_bots"] = state.get("num_bots", 2)
    current_match["total_damage_taken"] += max(0, ph - hp)
    current_match["total_score_gained"] += max(0, sc - ps)
    current_match["last_health"] = hp; current_match["last_score"] = sc
    current_match["enemies_alive_last_turn"] = len(state.get("enemies", []))
    current_match["health_history"].append(hp)
    if len(current_match["health_history"]) > 10: current_match["health_history"].pop(0)
    if sc - ps >= 30: current_match["kills"] += 1
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
    r = current_match["last_score"]
    sp = current_match["turns_played"] / max(1, current_match["max_turns"])
    r += sp * 40
    if o == "won_elimination": r += 90
    elif o == "won_score": r += 70
    elif o == "tied": r += 30
    elif o == "lost_score": r -= 10
    else: r -= 50 * (1 - sp)
    d, s = current_match["total_damage_taken"], current_match["total_score_gained"]
    if d > 0: r += min((s/d)*8, 25)
    elif s > 0: r += 25
    r += current_match["kills"] * 12
    r *= 1.0 + (current_match.get("num_bots", 2) - 2) * 0.08
    current_match["reward"] = round(r, 1)
    match_history.append(dict(current_match))
    if len(match_history) >= 3: learn()

def learn():
    global strategy_weights
    rec = match_history[-1]
    avg = sum(m["reward"] for m in match_history) / len(match_history)
    adv = max(-1, min(1, (rec["reward"] - avg) / max(abs(avg), 1)))
    used = rec.get("weights_used", {})
    for k in strategy_weights:
        if k not in used: continue
        c = strategy_weights[k]
        n = c + LEARNING_RATE * adv * (1 + abs(used[k] - c))
        n += random.gauss(0, EXPLORATION_NOISE * max(0.25, 1 - len(match_history)/120))
        lo, hi = WEIGHT_BOUNDS.get(k, (0, 1))
        strategy_weights[k] = max(lo, min(hi, n))

def get_active_weights():
    w = dict(strategy_weights)
    if len(match_history) < 40:
        for k in w:
            lo, hi = WEIGHT_BOUNDS.get(k, (0, 1))
            w[k] = max(lo, min(hi, w[k] + random.gauss(0, EXPLORATION_NOISE * 0.4)))
    return w

# =====================================================================
# CORE UTILITIES
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

# =====================================================================
# ZONE ENGINE
# =====================================================================
def predict_safe_radius(turn, max_turns, sz):
    ss = int(max_turns * 0.7); ir = sz // 2
    if turn < ss: return ir
    return max(0, ir - (turn - ss) // 5)

def zone_dmg(mx, my, sz, sr):
    return 5 if dist_center(mx, my, sz) > sr else 0

def turns_until_danger(mx, my, turn, max_turns, sz, sr):
    d = dist_center(mx, my, sz)
    if d > sr: return 0
    for f in range(1, max_turns - turn + 1):
        if d > predict_safe_radius(turn + f, max_turns, sz): return f
    return max_turns - turn

def is_future_safe(x, y, turn, max_turns, sz, ahead=15):
    return dist_center(x, y, sz) <= predict_safe_radius(turn + ahead, max_turns, sz)

def bfs_to_safe(mx, my, sz, walls, sr, blocked_tiles=None):
    """BFS shortest path to safe zone, routing around walls AND blocked tiles (enemies).
    Returns (distance, first_action) or (999, None)."""
    cx, cy = center_of(sz)
    if max(abs(mx-cx), abs(my-cy)) <= sr: return (0, None)

    avoid = walls | (blocked_tiles or set())
    visited = {(mx, my)}
    q = deque()

    # Try all first moves
    for dx, dy, action in DIRS:
        nx, ny = mx+dx, my+dy
        if 0<=nx<sz and 0<=ny<sz and (nx,ny) not in avoid:
            if max(abs(nx-cx), abs(ny-cy)) <= sr: return (1, action)
            visited.add((nx,ny))
            q.append((nx, ny, 1, action))

    # If all first moves blocked by enemies, try WITHOUT enemy blocking
    if not q and blocked_tiles:
        for dx, dy, action in DIRS:
            nx, ny = mx+dx, my+dy
            if 0<=nx<sz and 0<=ny<sz and (nx,ny) not in walls:
                if max(abs(nx-cx), abs(ny-cy)) <= sr: return (1, action)
                visited.add((nx,ny))
                q.append((nx, ny, 1, action))

    while q:
        x, y, dist, first = q.popleft()
        if dist > sz * 2: break
        for dx, dy, _ in DIRS:
            nx, ny = x+dx, y+dy
            if (nx,ny) not in visited and 0<=nx<sz and 0<=ny<sz and (nx,ny) not in walls:
                visited.add((nx,ny))
                if max(abs(nx-cx), abs(ny-cy)) <= sr: return (dist+1, first)
                q.append((nx, ny, dist+1, first))

    return (999, None)

# =====================================================================
# MOVEMENT
# =====================================================================
def move_to_target(mx, my, tx, ty, sz, walls, sr, enemies, avoid_en, cw):
    """Move toward target. Only penalizes EXACT last position (anti ping-pong)."""
    best_a, best_s = None, -9999
    last_pos = positions[-1] if len(positions) >= 2 else None
    # Actually last_pos is current position. We want the position BEFORE current.
    prev_pos = positions[-2] if len(positions) >= 2 else None

    for a in MOVES:
        nx, ny = apply(mx, my, a)
        if not okp(nx, ny, sz, walls): continue
        s = (manhattan(mx,my,tx,ty) - manhattan(nx,ny,tx,ty)) * 8
        if in_safe(nx,ny,sz,sr): s += 8
        else: s -= 20
        if avoid_en:
            for e in enemies:
                d = chebyshev(nx,ny,e["x"],e["y"])
                if d <= 1: s -= 28
                elif d <= 2: s -= 10
        # ONLY penalize ping-pong (going back to where we just were)
        if prev_pos and (nx, ny) == prev_pos: s -= 15
        s -= dist_center(nx, ny, sz) * cw
        if s > best_s: best_s, best_a = s, a
    return best_a

def flee_from(mx, my, enemies, sz, walls, sr):
    best_a, best_s = None, -9999
    prev_pos = positions[-2] if len(positions) >= 2 else None
    for a in MOVES:
        nx, ny = apply(mx, my, a)
        if not okp(nx, ny, sz, walls): continue
        s = sum((chebyshev(nx,ny,e["x"],e["y"])-chebyshev(mx,my,e["x"],e["y"]))*15 for e in enemies)
        if in_safe(nx,ny,sz,sr): s += 10
        else: s -= 24
        if prev_pos and (nx,ny) == prev_pos: s -= 8
        s -= dist_center(nx,ny,sz) * 0.5
        if s > best_s: best_s, best_a = s, a
    return best_a

# =====================================================================
# COMBAT
# =====================================================================
def my_dmg(me): return 15*(1+me.get("damage_boost_stacks",0))
def eff_dmg(me,en):
    d=my_dmg(me); return d//2 if en.get("is_defending") else d
def can_kill(me,en): return eff_dmg(me,en)>=en["health"]
def want_fight(me,en,adj_n,phase,w):
    if en["health"]<=eff_dmg(me,en): return True
    if en.get("is_defending") and me.get("damage_boost_stacks",0)==0: return False
    if en.get("energy",10)<3: return random.random()<w["helpless_enemy_bonus"]
    sc=w["aggression"]
    if me["health"]>en["health"]: sc+=0.15
    if me["energy"]>en.get("energy",10): sc+=0.10
    if me.get("damage_boost_stacks",0)>=1: sc+=0.20
    if me.get("shield_charges",0)>=1: sc+=0.15
    if adj_n>=2: sc-=0.30
    if me["health"]<40: sc-=0.25
    if me["energy"]<6: sc-=0.20
    if phase=="late": sc+=0.10*(1-w["survival_value"])
    return sc>0.50
def want_flee(me,adj,w):
    hp=me["health"]/100
    if hp<=w["flee_health_pct"]: return True
    if len(adj)>=2 and hp<w["flee_health_pct"]+0.20: return True
    if me["health"]<45 and me["energy"]<w["min_energy_reserve"]+1: return True
    return False

def score_target(tile,mx,my,energy,health,me,phase,sz,sr,w,turn,mt):
    tx,ty=tile["x"],tile["y"]; dist=manhattan(mx,my,tx,ty)
    hr,pu=tile.get("has_resource",False),tile.get("power_up")
    if energy<dist+2: return -1
    # HARD RULE: never chase targets outside safe zone in mid/late
    if not in_safe(tx,ty,sz,sr) and phase!="early": return -1
    if not is_future_safe(tx,ty,turn,mt,sz,ahead=8) and phase!="early": return -1
    s=0.0
    if pu:
        if pu=="damage_boost": s=20+w["damage_boost_priority"]*30
        elif pu=="shield": s=15+w["shield_priority"]*25+(12 if health<50 else 0)
        elif pu=="energy_pack": s=15+w["energy_pack_priority"]*25+(18 if energy<10 else 0)
        elif pu=="speed_boost": s=10+w["speed_priority"]*20
        elif pu=="vision_boost": s=5+w["vision_priority"]*15
    elif hr: s=10+w["resource_priority"]*20+(8 if energy<10 else 0)+(5 if phase=="early" else 0)
    else: return -1
    s-=dist*2.5
    if in_safe(tx,ty,sz,sr): s+=4
    return s

# =====================================================================
# MAIN DECISION ENGINE
# =====================================================================
def choose_action(state):
    global prev_action, consecutive_waits
    me=state["self"]; mx,my,energy,health=me["x"],me["y"],me["energy"],me["health"]
    turn,mt=state["turn"],state["max_turns"]
    sz,sr=state["arena_size"],state["safe_zone_radius"]
    enemies,tiles=state.get("enemies",[]),state.get("visible_tiles",[])
    w=active_weights
    pct=turn/mt; phase="early" if pct<0.30 else("mid" if pct<0.65 else "late")
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
    near_pu=[t for t in other_tgt if t.get("power_up")and manhattan(mx,my,t["x"],t["y"])<=3]

    taking_dmg=zone_dmg(mx,my,sz,sr)>0
    dist_safe,escape_action=bfs_to_safe(mx,my,sz,walls,sr,epos)
    tud=turns_until_danger(mx,my,turn,mt,sz,sr)
    zone_urgent=taking_dmg
    zone_soon=not safe_now and tud<=dist_safe+3
    f_safe=is_future_safe(mx,my,turn,mt,sz)

    def respond(a,e,m):
        global prev_action,consecutive_waits
        prev_action=a; consecutive_waits=consecutive_waits+1 if a=="WAIT" else 0
        return MoveResponse(action=a,emoji=e,mood=m)

    # === 1. ZONE EMERGENCY — taking damage NOW ===
    if zone_urgent:
        if health<=15 and energy>=2 and dist_safe>1:
            return respond("DEFEND","🛡️","zone shield")
        # BFS escape (routes around enemies)
        if escape_action and energy>=1:
            return respond(escape_action,"🚨","ESCAPE ZONE")
        # BFS without enemy avoidance (walk through them if needed)
        _,esc2=bfs_to_safe(mx,my,sz,walls,sr,None)
        if esc2 and energy>=1:
            return respond(esc2,"🚨","ESCAPE through enemy")
        if energy>=2:
            return respond("DEFEND","🛡️","zone survive")

    # === 2. ZONE WARNING — need to move soon ===
    if zone_soon:
        if on_collect and energy>=2:
            return respond("COLLECT","💰","grab before run")
        if escape_action and energy>=1:
            return respond(escape_action,"⚠️","pre-escape")
        _,esc2=bfs_to_safe(mx,my,sz,walls,sr,None)
        if esc2 and energy>=1:
            return respond(esc2,"⚠️","heading safe")

    # === 3. FUTURE ZONE — our position won't be safe later ===
    if not f_safe and phase!="early":
        if on_collect and energy>=2:
            return respond("COLLECT","💰","collect on way")
        cx,cy=int(center_of(sz)[0]),int(center_of(sz)[1])
        a=move_to_target(mx,my,cx,cy,sz,walls,sr,enemies,True,cw)
        if a and energy>=1:
            return respond(a,"📍","repositioning")

    # === 4. COLLECT ===
    if on_collect and energy>=2:
        return respond("COLLECT","💰","collecting")

    # === 5. NEARBY POWER-UP (only in safe zone) ===
    if near_pu and energy>=4 and not adj:
        safe_pu=[t for t in near_pu if in_safe(t["x"],t["y"],sz,sr) or phase=="early"]
        if safe_pu:
            bp=max(safe_pu,key=lambda t:score_target(t,mx,my,energy,health,me,phase,sz,sr,w,turn,mt))
            if score_target(bp,mx,my,energy,health,me,phase,sz,sr,w,turn,mt)>10:
                a=move_to_target(mx,my,bp["x"],bp["y"],sz,walls,sr,enemies,False,cw)
                if a: return respond(a,"✨",f"grab {bp['power_up']}")

    # === 6. FLEE ===
    if adj and want_flee(me,adj,w):
        if any(can_kill(me,e) for e in adj) and energy>=3:
            return respond("ATTACK","💀","parting kill")
        if health<=20 and energy>=2:
            return respond("DEFEND","🛡️","emergency")
        a=flee_from(mx,my,enemies,sz,walls,sr)
        if a and energy>=1: return respond(a,"💨","retreat")

    # === 7. FREE KILLS ===
    if adj and energy>=3 and any(can_kill(me,e) for e in adj):
        return respond("ATTACK","💀","executing")

    # === 8. HIT & RUN ===
    if prev_action=="ATTACK" and adj and energy>=1:
        a=flee_from(mx,my,enemies,sz,walls,sr)
        if a: return respond(a,"💨","hit & run")

    # === 9. SMART COMBAT ===
    if adj and energy>=4:
        for en in adj:
            if want_fight(me,en,len(adj),phase,w):
                if en.get("is_defending") and w["wait_out_defense"]>0.5 and me.get("damage_boost_stacks",0)==0:
                    return respond("WAIT","⏳","wait defense")
                return respond("ATTACK","⚔️","strike")
        if energy>=1:
            a=flee_from(mx,my,enemies,sz,walls,sr)
            if a: return respond(a,"💨","disengage")

    # === 10. CHASE TARGETS ===
    if other_tgt and energy>=3:
        scored=[(score_target(t,mx,my,energy,health,me,phase,sz,sr,w,turn,mt),t) for t in other_tgt]
        scored=[(s,t) for s,t in scored if s>0]
        scored.sort(key=lambda x:x[0],reverse=True)
        if scored:
            best=scored[0][1]; av=health<50 and w["survival_value"]>0.4
            a=move_to_target(mx,my,best["x"],best["y"],sz,walls,sr,enemies,av,cw)
            if a:
                # HARD CHECK: don't step outside safe zone for a target
                nx,ny=apply(mx,my,a)
                if in_safe(nx,ny,sz,sr) or phase=="early":
                    return respond(a,"🎯",f"get {best.get('power_up','resource')}")

    # === 11. HUNT WOUNDED ===
    if nearby and energy>=6 and health>50:
        md=my_dmg(me)
        hunt=[e for e in nearby if e["health"]<health and e["health"]<=md*3]
        if hunt and random.random()<w["kill_chase_value"]:
            t=min(hunt,key=lambda e:e["health"])
            a=move_to_target(mx,my,t["x"],t["y"],sz,walls,sr,enemies,False,cw)
            if a:
                nx,ny=apply(mx,my,a)
                if in_safe(nx,ny,sz,sr) or phase=="early":
                    return respond(a,"🐺","hunting")

    # === 12. ENERGY ===
    if energy<w.get("min_energy_reserve",3):
        if consecutive_waits>=4:
            for a in MOVES:
                nx,ny=apply(mx,my,a)
                if okp(nx,ny,sz,walls) and in_safe(nx,ny,sz,sr):
                    return respond(a,"🔍","break loop")
        return respond("WAIT","🔋","recharge")

    # === 13. EXPLORE — NEVER leave safe zone ===
    best_a,best_s=None,-9999
    shuffled=list(MOVES); random.shuffle(shuffled)
    prev_pos=positions[-2] if len(positions)>=2 else None
    for a in shuffled:
        nx,ny=apply(mx,my,a)
        if not okp(nx,ny,sz,walls): continue
        # HARD RULE: don't step outside safe zone
        if safe_now and not in_safe(nx,ny,sz,sr): continue
        s=0.0
        # Only penalize exact ping-pong
        if prev_pos and (nx,ny)==prev_pos: s-=12
        if in_safe(nx,ny,sz,sr): s+=8
        else: s-=15
        if is_future_safe(nx,ny,turn,mt,sz): s+=5
        else: s-=10
        s-=dist_center(nx,ny,sz)*cw
        for e in enemies:
            if chebyshev(nx,ny,e["x"],e["y"])<=2: s-=8
        s+=random.random()*4
        if s>best_s: best_s,best_a=s,a
    if best_a and energy>=1: return respond(best_a,"🔍","scout")
    return respond("WAIT","😴","idle")

# =====================================================================
# HTTP ENDPOINTS
# =====================================================================
@app.post("/move")
async def move(state: dict[str, Any]) -> MoveResponse:
    global positions,prev_action,consecutive_waits,active_weights
    mid=state["match_id"]
    if mid!=current_match.get("match_id"):
        if current_match.get("match_id"): finish_match()
        positions,prev_action,consecutive_waits=[],""  ,0
        active_weights=get_active_weights()
        new_match(mid,active_weights)
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
async def health_check(): return {"status":"ok","bot_id":BOT_ID}

@app.get("/stats/json")
async def stats_json():
    rec=match_history[-20:]; ci=None
    if current_match.get("match_id"):
        ci={"turns":current_match["turns_played"],"max_turns":current_match["max_turns"],
            "health":current_match["last_health"],"score":current_match["last_score"],
            "kills":current_match["kills"],
            "enemy_scores":{b:i["last_score"] for b,i in enemy_tracker.items()}}
    return {"matches":total_matches,"won":total_won,"lost":total_lost,"died":total_died,
        "current_match":ci,
        "recent":[{"outcome":m.get("outcome"),"score":m["last_score"],"reward":m.get("reward",0),
            "turns":m["turns_played"],"kills":m["kills"],"enemy_scores":m.get("enemy_scores",{})} for m in rec],
        "weights":{k:round(v,3) for k,v in strategy_weights.items()}}

@app.get("/stats",response_class=HTMLResponse)
async def stats_page():
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
            me=max(i["last_score"] for i in enemy_tracker.values())
            ss='<span style="color:#4ade80">📈 WIN</span>' if osc>me else('<span style="color:#fbbf24">🤝 TIE</span>' if osc==me else '<span style="color:#ef4444">📉 BEHIND</span>')
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
    wb=""
    for k,v in strategy_weights.items():
        lo,hi=WEIGHT_BOUNDS.get(k,(0,1));p=((v-lo)/(hi-lo))*100 if hi>lo else 50
        wb+=f'<div class="wr"><span class="wl">{k.replace("_"," ").title()}</span><div class="bar-bg sm"><div class="bar ac" style="width:{p:.0f}%"></div></div><span class="wv">{v:.2f}</span></div>'
    return HTMLResponse(content=f'''<!DOCTYPE html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta http-equiv="refresh" content="4"><title>🤖 Stats</title><style>*{{margin:0;padding:0;box-sizing:border-box}}body{{font-family:-apple-system,sans-serif;background:#0f172a;color:#e2e8f0;padding:16px}}h1{{text-align:center;font-size:1.6em;margin-bottom:16px;color:#38bdf8}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(110px,1fr));gap:10px;margin-bottom:16px}}.kpi{{background:#1e293b;border-radius:10px;padding:12px;text-align:center}}.kpi .num{{font-size:1.8em;font-weight:bold}}.kpi .label{{font-size:.75em;color:#94a3b8;margin-top:2px}}.green{{color:#4ade80}}.red{{color:#ef4444}}.blue{{color:#38bdf8}}.yellow{{color:#fbbf24}}.purple{{color:#a78bfa}}.card{{background:#1e293b;border-radius:10px;padding:14px;margin-bottom:14px}}.card h2{{font-size:1em;margin-bottom:10px;color:#94a3b8}}.card.live{{border:1px solid #ef4444}}.sr{{display:flex;align-items:center;gap:6px;margin-bottom:5px}}.sr span{{width:80px;font-size:.8em;color:#94a3b8}}.sr strong{{font-size:.9em}}.bar-bg{{flex:1;background:#334155;border-radius:4px;height:12px;overflow:hidden}}.bar-bg.sm{{height:7px}}.bar{{height:100%;border-radius:4px;background:#38bdf8}}.bar.ac{{background:#818cf8}}table{{width:100%;border-collapse:collapse;font-size:.8em}}th{{text-align:left;color:#94a3b8;padding:6px;border-bottom:1px solid #334155}}td{{padding:6px;border-bottom:1px solid #1e293b}}tr.won td{{color:#4ade80}}tr.lost td{{color:#fbbf24}}tr.died td{{color:#ef4444}}tr.tied td{{color:#94a3b8}}.two-col{{display:grid;grid-template-columns:1fr 1fr;gap:14px}}@media(max-width:600px){{.two-col{{grid-template-columns:1fr}}}}.wr{{display:flex;align-items:center;gap:4px;margin-bottom:3px}}.wl{{width:130px;font-size:.7em;color:#94a3b8}}.wv{{width:36px;font-size:.7em;text-align:right}}</style></head><body><h1>🤖 RoboKova Bot v7</h1><div class="grid"><div class="kpi"><div class="num blue">{total_matches}</div><div class="label">Matches</div></div><div class="kpi"><div class="num green">{total_won}</div><div class="label">Won</div></div><div class="kpi"><div class="num yellow">{total_lost}</div><div class="label">Lost</div></div><div class="kpi"><div class="num red">{total_died}</div><div class="label">Died</div></div><div class="kpi"><div class="num {"green" if int(wr)>=50 else "red"}">{wr}%</div><div class="label">Win Rate</div></div><div class="kpi"><div class="num purple">{avs:.0f}</div><div class="label">Avg Score</div></div></div>{cm}<div class="two-col"><div class="card"><h2>📜 History</h2><table><tr><th>Result</th><th>Score</th><th>Enemy</th><th>Kills</th><th>Turns</th><th>Reward</th></tr>{rows or'<tr><td colspan="6" style="color:#64748b">No matches</td></tr>'}</table></div><div class="card"><h2>🧠 Weights</h2>{wb}</div></div><p style="text-align:center;color:#475569;margin-top:16px;font-size:.7em">Refreshes 4s · {len(match_history)} learned</p></body></html>''')
