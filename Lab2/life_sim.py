#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Artificial Life - food chain with deterministic planner, Tk UI, time slider, population chart.

Two-phase tick: PLAN all targets -> RESOLVE conflicts (predators outrank) -> MOVE.
Supports swaps, panic sidestep, and an anti-stuck fallback after N idle ticks.

UI:
- Map size, Plants, New plants per eat, Herbivores, Predators, Iterations
- Checkbox: Disable agent limits (25-50% of N²/5)
- Buttons: Run algorithm, Previous, Next
- Map view + populations chart + slider, stats label.
"""

from __future__ import annotations
import random, time, argparse, signal, sys
from dataclasses import dataclass, field
from enum import IntEnum
from typing import List, Tuple, Optional, Dict

# -------------------- constants --------------------
class Obj(IntEnum):
    EMPTY = 0
    HERBIVORE = 2
    PREDATOR  = 3

DIRS = [(0,-1),(1,0),(0,1),(-1,0)]  # N,E,S,W

MAX_ENERGY_HERB = 100
MAX_ENERGY_PRED = 100
REPRO_THRESHOLD_HERB = int(0.9 * MAX_ENERGY_HERB)
REPRO_THRESHOLD_PRED = int(0.9 * MAX_ENERGY_PRED)

E_START_HERB = 80
E_START_PRED = 40

LOSS_HERB_PER_DAY = 2
LOSS_PRED_PER_DAY = 1

E_GAIN_EAT_HERB  = 20
E_GAIN_EAT_PLANT = 10

TURN_LEFT, TURN_RIGHT, FORWARD, EAT = 0, 1, 2, 3


# -------------------- tiny “brain” --------------------
@dataclass
class Brain:
    W: List[List[float]] = field(default_factory=lambda: [[random.uniform(-1,1) for _ in range(12)] for _ in range(4)])
    b: List[float]       = field(default_factory=lambda: [random.uniform(-0.5,0.5) for _ in range(4)])

    def mutate(self, sigma: float = 0.05) -> 'Brain':
        Wm = [[w + random.gauss(0, sigma) for w in row] for row in self.W]
        bm = [bb + random.gauss(0, sigma) for bb in self.b]
        return Brain(Wm, bm)

    def forward(self, x12: List[float]) -> List[float]:
        y = []
        for r in range(4):
            s = self.b[r]
            wr = self.W[r]
            for i in range(12): s += wr[i] * x12[i]
            y.append(s)
        return y

# IMPORTANT: eq=False keeps identity equality & hash -> Agent is hashable for dict/set
@dataclass(eq=False)
class Agent:
    kind: Obj
    energy: int
    age: int
    generation: int
    x: int
    y: int
    gaze: int
    brain: Brain
    last_outputs: Tuple[float,float,float,float] = (0,0,0,0)
    idle: int = 0  # anti-stuck counter


# -------------------- world --------------------
class World:
    def __init__(self, N:int, plants:int, herbivores:int, predators:int,
                 seed:Optional[int]=None, per_eat_respawn:int=1, enforce_caps:bool=True,
                 herbs_first:bool=True, panic_sidestep:bool=True, anti_stuck:bool=True,
                 force_after:int=3):
        if seed is not None: random.seed(seed)
        self.N = N
        self.iteration = 0
        self.per_eat_respawn = max(0, int(per_eat_respawn))
        self.herbs_first = herbs_first
        self.panic_sidestep = panic_sidestep
        self.anti_stuck = anti_stuck
        self.force_after = max(1, force_after)

        self.anim_grid = [[Obj.EMPTY for _ in range(N)] for _ in range(N)]
        self.plant_grid = [[False for _ in range(N)] for _ in range(N)]
        self.animals: List[Agent] = []

        # centered 75% forest
        margin = max(1, int(round(N * 0.125)))
        self.forest = (margin, margin, N - margin - 1, N - margin - 1)

        # caps
        if enforce_caps:
            base = max(1, (N * N) // 5)
            lo, hi = int(0.25 * base), int(0.50 * base)
            herbivores = min(max(herbivores, lo), hi)
            predators  = min(max(predators,  lo), hi)

        self._place_plants(plants)
        self._spawn(Obj.HERBIVORE, herbivores)
        self._spawn(Obj.PREDATOR,  predators)

        self.stats = {
            "counts": [],
            "max_age": {Obj.HERBIVORE:0, Obj.PREDATOR:0},
            "reproductions": {Obj.HERBIVORE:0, Obj.PREDATOR:0},
        }
        self._log_counts(0)

    # ---- placement
    def _place_plants(self, count:int):
        x0,y0,x1,y1 = self.forest
        cells = [(x,y) for y in range(y0,y1+1) for x in range(x0,x1+1) if not self.plant_grid[y][x]]
        random.shuffle(cells)
        for (x,y) in cells[:min(count, len(cells))]:
            self.plant_grid[y][x] = True

    def _spawn(self, kind:Obj, count:int):
        cells = [(x,y) for y in range(self.N) for x in range(self.N) if self.anim_grid[y][x] == Obj.EMPTY]
        random.shuffle(cells)
        for (x,y) in cells[:min(count, len(cells))]:
            self.anim_grid[y][x] = kind
            e = E_START_HERB if kind==Obj.HERBIVORE else E_START_PRED
            self.animals.append(
                Agent(kind=kind, energy=e, age=0, generation=1,
                      x=x, y=y, gaze=random.randrange(4), brain=Brain())
            )

    def _forest_empties(self):
        x0,y0,x1,y1 = self.forest
        return [(x,y) for y in range(y0,y1+1) for x in range(x0,x1+1) if not self.plant_grid[y][x]]

    def _respawn_plants(self, k:int):
        if k <= 0: return
        cells = self._forest_empties()
        if not cells: return
        random.shuffle(cells)
        for (x,y) in cells[:min(k, len(cells))]:
            self.plant_grid[y][x] = True

    # ---- helpers
    def _forward(self, a:Agent) -> Tuple[int,int]:
        dx,dy = DIRS[a.gaze]
        return (a.x + dx) % self.N, (a.y + dy) % self.N

    def _neighbors(self, x:int, y:int) -> List[Tuple[int,int]]:
        N=self.N
        return [((x+dx)%N,(y+dy)%N) for dx,dy in DIRS]

    def _sense12(self, a:Agent) -> List[float]:
        N=self.N
        def rot(dx,dy,g):
            if g==0: return dx,dy
            if g==1: return dy,-dx
            if g==2: return -dx,-dy
            return -dy,dx
        sec={"front":{"herb":0,"pred":0,"plant":0},
             "left":{"herb":0,"pred":0,"plant":0},
             "right":{"herb":0,"pred":0,"plant":0},
             "near":{"herb":0,"pred":0,"plant":0}}
        for dy in (-1,0,1):
            for dx in (-1,0,1):
                if dx==0 and dy==0: continue
                x=(a.x+dx)%N; y=(a.y+dy)%N
                k=self.anim_grid[y][x]
                if k==Obj.HERBIVORE: sec["near"]["herb"]+=1
                elif k==Obj.PREDATOR: sec["near"]["pred"]+=1
                if self.plant_grid[y][x]: sec["near"]["plant"]+=1
        for dy in range(-2,3):
            for dx in range(-2,3):
                if dx==0 and dy==0: continue
                rx,ry=rot(dx,dy,a.gaze)
                if abs(rx)<=2 and abs(ry)<=2:
                    if   ry < 0: name="front"
                    elif rx < 0: name="left"
                    elif rx > 0: name="right"
                    else: continue
                    x=(a.x+dx)%N; y=(a.y+dy)%N
                    k=self.anim_grid[y][x]
                    if k==Obj.HERBIVORE: sec[name]["herb"]+=1
                    elif k==Obj.PREDATOR: sec[name]["pred"]+=1
                    if self.plant_grid[y][x]: sec[name]["plant"]+=1
        out=[]; norm=1/12
        for key in ("front","left","right","near"):
            out.extend([sec[key]["herb"]*norm, sec[key]["pred"]*norm, sec[key]["plant"]*norm])
        return out

    def _nearest(self, a:Agent, *, want_pred=False, want_herb=False, want_plant=False)\
            -> Optional[Tuple[int,int,int]]:
        best=None; bestd=99
        for dy in range(-2,3):
            for dx in range(-2,3):
                if dx==0 and dy==0: continue
                x=(a.x+dx)%self.N; y=(a.y+dy)%self.N
                ok=False
                if want_pred and self.anim_grid[y][x]==Obj.PREDATOR: ok=True
                if want_herb and self.anim_grid[y][x]==Obj.HERBIVORE: ok=True
                if want_plant and self.plant_grid[y][x]: ok=True
                if not ok: continue
                d=abs(dx)+abs(dy)
                if d<bestd: bestd=d; best=(dx,dy,d)
        return best

    def _dir_toward(self, dx:int, dy:int) -> int:
        if abs(dx) >= abs(dy): return 1 if dx>0 else 3
        else: return 2 if dy>0 else 0

    def _cheb_dist(self, x1,y1,x2,y2):
        return max(abs(x1-x2), abs(y1-y2))

    def _plan_target(self, a:Agent) -> Tuple[int,int,bool]:
        N=self.N
        pounce = False
        pref_dir = a.gaze
        if a.kind == Obj.HERBIVORE:
            near_pred = self._nearest(a, want_pred=True)
            if near_pred is not None and near_pred[2] <= 2:
                dx,dy,_ = near_pred
                pref_dir = self._dir_toward(-dx, -dy)
            else:
                near_plant = self._nearest(a, want_plant=True)
                if near_plant is not None:
                    pref_dir = self._dir_toward(near_plant[0], near_plant[1])
        else:
            near_herb = self._nearest(a, want_herb=True)
            if near_herb is not None:
                pref_dir = self._dir_toward(near_herb[0], near_herb[1])
                fx, fy = (a.x + DIRS[pref_dir][0]) % N, (a.y + DIRS[pref_dir][1]) % N
                if self.anim_grid[fy][fx] == Obj.HERBIVORE:
                    pounce = True

        a.gaze = pref_dir
        fx, fy = self._forward(a)
        if self.anim_grid[fy][fx] == Obj.EMPTY:
            return fx, fy, pounce

        # panic sidestep
        left_dir  = (pref_dir - 1) % 4
        right_dir = (pref_dir + 1) % 4
        lx,ly = (a.x + DIRS[left_dir][0]) % N, (a.y + DIRS[left_dir][1]) % N
        rx,ry = (a.x + DIRS[right_dir][0]) % N, (a.y + DIRS[right_dir][1]) % N
        def safe(xx,yy): return self.anim_grid[yy][xx] == Obj.EMPTY
        if a.kind == Obj.HERBIVORE:
            pred = self._nearest(a, want_pred=True)
            if pred is not None:
                px,py = (a.x+pred[0])%N, (a.y+pred[1])%N
                dl = self._cheb_dist(lx,ly,px,py)
                dr = self._cheb_dist(rx,ry,px,py)
                for _,xx,yy in sorted([(dl,lx,ly),(dr,rx,ry)], reverse=True):
                    if safe(xx,yy): return xx,yy,False
        else:
            herb = self._nearest(a, want_herb=True)
            if herb is not None:
                hx,hy = (a.x+herb[0])%N, (a.y+herb[1])%N
                dl = self._cheb_dist(lx,ly,hx,hy)
                dr = self._cheb_dist(rx,ry,hx,hy)
                for _,xx,yy in sorted([(dl,lx,ly),(dr,rx,ry)]):
                    if safe(xx,yy): return xx,yy,False
        if self.anim_grid[ly][lx] == Obj.EMPTY: return lx,ly,False
        if self.anim_grid[ry][rx] == Obj.EMPTY: return rx,ry,False

        neigh = self._neighbors(a.x,a.y)
        empties = [(xx,yy) for (xx,yy) in neigh if self.anim_grid[yy][xx]==Obj.EMPTY]
        if len(empties) == 1:
            return empties[0][0], empties[0][1], False

        if a.kind == Obj.HERBIVORE:
            pred = self._nearest(a, want_pred=True)
            plant = self._nearest(a, want_plant=True)
            best=None; best_score=-1e9
            for xx,yy in empties:
                score = 0.0
                if pred is not None:
                    px,py = (a.x+pred[0])%N,(a.y+pred[1])%N
                    score += 10.0 * self._cheb_dist(xx,yy,px,py)
                if plant is not None:
                    plx,ply = (a.x+plant[0])%N,(a.y+plant[1])%N
                    score += -1.0 * self._cheb_dist(xx,yy,plx,ply)
                if score > best_score:
                    best_score=score; best=(xx,yy)
            if best is not None:
                return best[0], best[1], False
        else:
            herb = self._nearest(a, want_herb=True)
            if herb is not None:
                hx,hy = (a.x+herb[0])%N,(a.y+herb[1])%N
                best=None; best_d=1e9
                for xx,yy in empties:
                    d=self._cheb_dist(xx,yy,hx,hy)
                    if d<best_d: best_d=d; best=(xx,yy)
                if best is not None:
                    return best[0],best[1],False

        return a.x,a.y,False

    # -------------------- one day --------------------
    def step(self):
        self.iteration += 1

        herbs = [a for a in self.animals if a.kind==Obj.HERBIVORE]
        preds = [a for a in self.animals if a.kind==Obj.PREDATOR]
        move_order = random.sample(herbs, k=len(herbs)) + random.sample(preds, k=len(preds))

        plans: Dict[Agent, Tuple[int,int,bool]] = {}
        for a in move_order:
            a.last_outputs = tuple(a.brain.forward(self._sense12(a)))
            plans[a] = self._plan_target(a)

        wishes: Dict[Tuple[int,int], List[Agent]] = {}
        for a,(tx,ty,_) in plans.items():
            wishes.setdefault((tx,ty), []).append(a)

        def rank(agent:Agent) -> int:
            return 1 if agent.kind==Obj.PREDATOR else 2

        moved=set()

        # swaps
        for a,(tx,ty,_) in list(plans.items()):
            if a in moved or (tx,ty)==(a.x,a.y): continue
            other_kind = self.anim_grid[ty][tx]
            if other_kind == Obj.EMPTY: continue
            b = next((o for o in self.animals if o.x==tx and o.y==ty and o.kind==other_kind), None)
            if not b: continue
            btx,bty,_ = plans.get(b, (b.x,b.y,False))
            if (btx,bty) == (a.x,a.y):
                self.anim_grid[a.y][a.x], self.anim_grid[b.y][b.x] = self.anim_grid[b.y][b.x], self.anim_grid[a.y][a.x]
                a.x,a.y, b.x,b.y = b.x,b.y, a.x,a.y
                a.idle = 0; b.idle = 0
                moved.add(a); moved.add(b)
                continue

        # contested cells
        for (cx,cy), claimers in wishes.items():
            movable = [ag for ag in claimers if (plans[ag][0],plans[ag][1]) != (ag.x,ag.y)]
            if not movable: 
                continue
            if self.anim_grid[cy][cx] != Obj.EMPTY:
                continue
            by_rank={}
            for ag in movable:
                by_rank.setdefault(rank(ag), []).append(ag)
            top=min(by_rank.keys())
            winner = random.choice(by_rank[top]) if len(by_rank[top])>1 else by_rank[top][0]
            if winner not in moved:
                self.anim_grid[winner.y][winner.x] = Obj.EMPTY
                winner.x,winner.y = cx,cy
                self.anim_grid[cy][cx] = winner.kind
                winner.idle = 0
                moved.add(winner)

        # remaining uncontested
        for a in move_order:
            if a in moved: continue
            tx,ty,_ = plans[a]
            if (tx,ty) == (a.x,a.y):
                a.idle += 1
                continue
            if self.anim_grid[ty][tx] == Obj.EMPTY:
                self.anim_grid[a.y][a.x] = Obj.EMPTY
                a.x,a.y = tx,ty
                self.anim_grid[ty][tx] = a.kind
                a.idle = 0
                moved.add(a)
            else:
                a.idle += 1

        # anti-stuck
        for a in self.animals:
            if a.idle >= 3:
                neigh = [(xx,yy) for (xx,yy) in self._neighbors(a.x,a.y) if self.anim_grid[yy][xx]==Obj.EMPTY]
                if neigh:
                    if a.kind==Obj.HERBIVORE:
                        pred = self._nearest(a, want_pred=True)
                        if pred is not None:
                            px,py=(a.x+pred[0])%self.N,(a.y+pred[1])%self.N
                            neigh.sort(key=lambda p: -self._cheb_dist(p[0],p[1],px,py))
                    else:
                        herb = self._nearest(a, want_herb=True)
                        if herb is not None:
                            hx,hy=(a.x+herb[0])%self.N,(a.y+herb[1])%self.N
                            neigh.sort(key=lambda p: self._cheb_dist(p[0],p[1],hx,hy))
                    nx,ny = neigh[0]
                    self.anim_grid[a.y][a.x] = Obj.EMPTY
                    a.x,a.y = nx,ny
                    self.anim_grid[ny][nx] = a.kind
                    a.idle = 0

        # pounce
        preds_now = [aa for aa in self.animals if aa.kind==Obj.PREDATOR]
        for a in random.sample(preds_now, k=len(preds_now)):
            nx,ny = self._forward(a)
            if self.anim_grid[ny][nx] == Obj.HERBIVORE:
                self.anim_grid[a.y][a.x] = Obj.EMPTY
                self.anim_grid[ny][nx]  = Obj.PREDATOR
                a.x,a.y = nx,ny
                a.energy = min(MAX_ENERGY_PRED, a.energy + E_GAIN_EAT_HERB)

        self.animals = [aa for aa in self.animals if self.anim_grid[aa.y][aa.x] == aa.kind]

        # herb eat
        herbs_now = [aa for aa in self.animals if aa.kind==Obj.HERBIVORE]
        for a in random.sample(herbs_now, k=len(herbs_now)):
            nx, ny = self._forward(a)
            if self.plant_grid[ny][nx]:
                self.plant_grid[ny][nx] = False
                a.energy = min(MAX_ENERGY_HERB, a.energy + E_GAIN_EAT_PLANT)
                self._respawn_plants(self.per_eat_respawn)

        # reproduce
        for a in random.sample(self.animals, k=len(self.animals)):
            thr = REPRO_THRESHOLD_HERB if a.kind==Obj.HERBIVORE else REPRO_THRESHOLD_PRED
            if a.energy < thr: 
                continue
            neigh = self._neighbors(a.x,a.y)
            random.shuffle(neigh)
            for nx,ny in neigh:
                if self.anim_grid[ny][nx] == Obj.EMPTY:
                    half = a.energy // 2
                    child_e = a.energy - half
                    a.energy = half
                    self.anim_grid[ny][nx] = a.kind
                    self.animals.append(
                        Agent(kind=a.kind, energy=child_e, age=0, generation=a.generation+1,
                              x=nx, y=ny, gaze=random.randrange(4), brain=a.brain.mutate(0.05))
                    )
                    self.stats["reproductions"][a.kind] += 1
                    break

        # metabolism
        survivors: List[Agent] = []
        for a in self.animals:
            a.age += 1
            if a.kind==Obj.HERBIVORE and a.age > self.stats["max_age"][Obj.HERBIVORE]:
                self.stats["max_age"][Obj.HERBIVORE] = a.age
            if a.kind==Obj.PREDATOR and a.age > self.stats["max_age"][Obj.PREDATOR]:
                self.stats["max_age"][Obj.PREDATOR] = a.age
            a.energy -= (LOSS_HERB_PER_DAY if a.kind==Obj.HERBIVORE else LOSS_PRED_PER_DAY)
            if a.energy <= 0:
                self.anim_grid[a.y][a.x] = Obj.EMPTY
            else:
                survivors.append(a)
        self.animals = survivors

        self._log_counts(self.iteration)

    # -------------------- stats --------------------
    def _log_counts(self, i:int):
        plants = sum(1 for y in range(self.N) for x in range(self.N) if self.plant_grid[y][x])
        herb   = sum(1 for a in self.animals if a.kind==Obj.HERBIVORE)
        pred   = sum(1 for a in self.animals if a.kind==Obj.PREDATOR)
        self.stats["counts"].append({"iter":i, "plants":plants, "herbivores":herb, "predators":pred})


# -------------------- UI --------------------
def run_ui(defaults=None):
    import tkinter as tk
    from tkinter import ttk, messagebox
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    import numpy as np

    defaults = defaults or dict(N=64, plants=2000, eat_respawn=1, herb=300, pred=300, steps=200)

    root = tk.Tk(); root.title("Life simulation")

    def on_close():
        # schedule a graceful quit then destroy & close figures
        try:
            root.after(0, root.quit)
        except Exception:
            pass
        # give Tk a breath to exit mainloop if we're inside one
        root.after(50, _destroy_all)

    def _destroy_all():
        try:
            # Close matplotlib figures first to detach from Tk
            import matplotlib.pyplot as _plt
            _plt.close('all')
        except Exception:
            pass
        try:
            root.destroy()
        except Exception:
            pass

    # Bind OS window close and signals to the same handler
    root.protocol("WM_DELETE_WINDOW", on_close)
    signal.signal(signal.SIGINT,  lambda *_: on_close())
    try:
        signal.signal(signal.SIGTERM, lambda *_: on_close())
    except Exception:
        # SIGTERM may not exist on some platforms (e.g., Windows Python launched without console)
        pass

    lf = ttk.Frame(root, padding=6); lf.grid(row=0, column=0, sticky="ns")

    def L(parent, text, init):
        ttk.Label(parent, text=text).pack(anchor="w")
        e = ttk.Entry(parent, width=18); e.insert(0, str(init))
        e.pack(anchor="w", pady=(0,6)); return e

    eN      = L(lf, "Map size (N)", defaults["N"])
    ePlants = L(lf, "Plants count (forest only)", defaults["plants"])
    eResp   = L(lf, "New plants per eat (0=off)", defaults["eat_respawn"])
    eHerb   = L(lf, "Herbivore count", defaults["herb"])
    ePred   = L(lf, "Predator count",  defaults["pred"])
    eSteps  = L(lf, "Iterations (days)", defaults["steps"])

    caps_var = tk.BooleanVar(value=False)
    ttk.Checkbutton(lf, text="Disable agent limits (25-50% of N²/5)", variable=caps_var)\
        .pack(anchor="w", pady=(0,6))

    btn_run  = ttk.Button(lf, text="Run algorithm")
    btn_prev = ttk.Button(lf, text="Previous")
    btn_next = ttk.Button(lf, text="Next")
    btn_run.pack(anchor="w", pady=10); btn_prev.pack(anchor="w", pady=2); btn_next.pack(anchor="w", pady=2)

    stats_var = tk.StringVar(value="Max age H: –, P: – | Reprod H: –, P: –")
    ttk.Label(lf, textvariable=stats_var, wraplength=240, justify="left").pack(anchor="w", pady=(10,4))
    info_var = tk.StringVar(value="Ready.")
    ttk.Label(lf, textvariable=info_var, wraplength=240, justify="left", foreground="#0a5").pack(anchor="w")

    # --- figure for map
    fig_map, ax_map = plt.subplots(figsize=(6.8,6.8))
    canvas_map = FigureCanvasTkAgg(fig_map, master=root)
    canvas_map.get_tk_widget().grid(row=0, column=1, padx=10, pady=(10,4), sticky="n")

    # --- figure for populations time series
    fig_ts, ax_ts = plt.subplots(figsize=(6.8,2.8))
    canvas_ts = FigureCanvasTkAgg(fig_ts, master=root)
    canvas_ts.get_tk_widget().grid(row=1, column=1, padx=10, pady=(0,4), sticky="ew")

    # --- slider (fast navigation)
    slider_frame = ttk.Frame(root)
    slider_frame.grid(row=2, column=1, padx=10, pady=(0,10), sticky="ew")
    slider_var = tk.DoubleVar(value=0.0)  # ttk.Scale uses float
    step_label_var = tk.StringVar(value="Step 0 / 0")
    step_label = ttk.Label(slider_frame, textvariable=step_label_var)
    step_label.pack(side="left")
    step_scale = ttk.Scale(slider_frame, from_=0.0, to=0.0, orient="horizontal",
                           variable=slider_var, length=520)
    step_scale.pack(side="left", padx=10, fill="x", expand=True)

    world: Optional[World] = None
    snapshots: List["np.ndarray"] = []
    series: List[Dict[str,int]] = []
    k_idx = 0

    PLANT=(0.1,0.9,0.1); EMPTY=(0.95,0.95,0.2); HERB=(0.1,0.2,0.9); PRED=(0.9,0.1,0.1)

    def raster(w:World)->"np.ndarray":
        N=w.N
        import numpy as np
        img = np.zeros((N,N,3), dtype=float); img[:,:,:]=EMPTY
        for y in range(N):
            for x in range(N):
                if w.plant_grid[y][x]: img[y,x,:]=PLANT
        for a in w.animals:
            img[a.y,a.x,:] = HERB if a.kind==Obj.HERBIVORE else PRED
        return img

    def title_for(i:int)->str:
        if not series: return f"Step {i}"
        i = max(0, min(i, len(series)-1)); s = series[i]
        return f"Simulation Step {s['iter']}  Herbovines - {s['herbivores']}  Predators - {s['predators']}"

    def draw_map(img, title):
        ax_map.clear(); ax_map.imshow(img, interpolation="nearest")
        ax_map.set_xticks([]); ax_map.set_yticks([]); ax_map.set_title(title)
        canvas_map.draw_idle()

    def draw_ts(cur_index:int):
        ax_ts.clear()
        if not series:
            canvas_ts.draw_idle(); return
        x = [s["iter"] for s in series]
        y_plants = [s["plants"] for s in series]
        y_herb   = [s["herbivores"] for s in series]
        y_pred   = [s["predators"] for s in series]

        ax_ts.plot(x, y_plants, label="Plants", color=PLANT)
        ax_ts.plot(x, y_herb,   label="Herbivores", color=HERB)
        ax_ts.plot(x, y_pred,   label="Predators", color=PRED)
        ax_ts.set_xlabel("Step")
        ax_ts.set_ylabel("Count")
        ax_ts.grid(True, alpha=0.25)
        ax_ts.legend(loc="upper right", frameon=False)

        cur_index = max(0, min(cur_index, len(series)-1))
        cx = series[cur_index]["iter"]
        ax_ts.axvline(cx, linestyle="--", alpha=0.7, color="black")
        ax_ts.set_title("Populations per step")
        canvas_ts.draw_idle()

    def update_step_label(idx:int):
        total = max(0, len(snapshots)-1)
        step_label_var.set(f"Step {idx} / {total}")

    def draw_both(img, title, cur_index):
        draw_map(img, title)
        draw_ts(cur_index)
        update_step_label(cur_index)

    def set_slider_bounds():
        last = max(0, len(snapshots)-1)
        step_scale.configure(to=float(last))
        slider_var.set(float(k_idx))
        update_step_label(k_idx)

    def on_slider_move(_val=None):
        nonlocal k_idx
        if not snapshots: 
            return
        idx = int(round(float(slider_var.get())))
        idx = max(0, min(idx, len(snapshots)-1))
        if idx != k_idx:
            k_idx = idx
            draw_both(snapshots[k_idx], title_for(k_idx), k_idx)

    step_scale.configure(command=on_slider_move)

    def as_int(name, entry, minv, maxv=None):
        try: v=int(entry.get())
        except: messagebox.showerror("Invalid parameter", f"{name} must be integer."); return None
        if v<minv: messagebox.showerror("Invalid parameter", f"{name} must be ≥ {minv}."); return None
        if maxv is not None and v>maxv: messagebox.showerror("Invalid parameter", f"{name} must be ≤ {maxv}."); return None
        return v

    def validate(N,plants,resp,herb,pred,steps,enforce_caps:bool):
        msgs=[]
        margin=max(1,int(round(N*0.125))); x0,y0,x1,y1=margin,margin,N-margin-1,N-margin-1
        forest_area=max(0,(x1-x0+1)*(y1-y0+1))
        if plants>forest_area:
            old=plants; plants=forest_area; msgs.append(f"Plants: {old} -> {plants} (forest capacity).")
        if resp<0: old=resp; resp=0; msgs.append(f"New plants per eat: {old} -> 0.")
        if enforce_caps:
            base=max(1,(N*N)//5); lo,hi=int(0.25*base),int(0.50*base)
            if herb<lo or herb>hi: old=herb; herb=min(max(herb,lo),hi); msgs.append(f"Herbivores: {old} -> {herb} ([{lo}..{hi}]).")
            if pred<lo or pred>hi: old=pred; pred=min(max(pred,lo),hi); msgs.append(f"Predators: {old} -> {pred} ([{lo}..{hi}]).")
        else:
            msgs.append("Agent limits DISABLED.")
        if steps>10000: old=steps; steps=10000; msgs.append(f"Iterations: {old} -> {steps} (max 10000).")
        return plants,resp,herb,pred,steps,msgs

    def run_algo():
        nonlocal world, snapshots, series, k_idx
        N=as_int("Map size", eN, 8); plants=as_int("Plants count", ePlants, 0)
        resp=as_int("New plants per eat", eResp, 0)
        herb=as_int("Herbivore count", eHerb, 0); pred=as_int("Predator count", ePred, 0)
        steps=as_int("Iterations (days)", eSteps, 1)
        if None in (N,plants,resp,herb,pred,steps): return
        enforce = not caps_var.get()
        plants,resp,herb,pred,steps,msgs = validate(N,plants,resp,herb,pred,steps,enforce)
        for ent,val in ((ePlants,plants),(eResp,resp),(eHerb,herb),(ePred,pred),(eSteps,steps)):
            ent.delete(0,"end"); ent.insert(0,str(val))
        info_var.set(" | ".join(msgs) if msgs else "Parameters OK.")

        world = World(N, plants, herb, pred, seed=int(time.time()) & 0xFFFF,
                      per_eat_respawn=resp, enforce_caps=enforce,
                      herbs_first=True, panic_sidestep=True, anti_stuck=True, force_after=3)

        snapshots = [raster(world)]
        series    = world.stats["counts"][:]
        for _ in range(steps):
            world.step()
            snapshots.append(raster(world))
            series = world.stats["counts"][:]

        k_idx = len(snapshots) - 1
        stats_var.set(
            f"Max age H: {world.stats['max_age'][Obj.HERBIVORE]}, "
            f"P: {world.stats['max_age'][Obj.PREDATOR]} | "
            f"Reprod H: {world.stats['reproductions'][Obj.HERBIVORE]}, "
            f"P: {world.stats['reproductions'][Obj.PREDATOR]}"
        )
        set_slider_bounds()
        draw_both(snapshots[k_idx], title_for(k_idx), k_idx)

    def prev():
        nonlocal k_idx
        if not snapshots: return
        k_idx = max(0, k_idx-1)
        slider_var.set(float(k_idx))
        draw_both(snapshots[k_idx], title_for(k_idx), k_idx)

    def next_():
        nonlocal k_idx
        if not snapshots: return
        k_idx = min(len(snapshots)-1, k_idx+1)
        slider_var.set(float(k_idx))
        draw_both(snapshots[k_idx], title_for(k_idx), k_idx)

    btn_run.configure(command=run_algo)
    btn_prev.configure(command=prev)
    btn_next.configure(command=next_)

    # Tk mainloop; suppress Ctrl+C traceback; ensure close works via on_close()
    try:
        root.mainloop()
    except KeyboardInterrupt:
        on_close()
        # fall through and exit once Tk is destroyed

# -------------------- entry --------------------
def main():
    p = argparse.ArgumentParser(description="Artificial life simulator (jam-resistant, Tk UI)")
    p.add_argument("--ui", action="store_true", help="Start UI (default).")
    args = p.parse_args()
    run_ui()

if __name__ == "__main__":
    main()
