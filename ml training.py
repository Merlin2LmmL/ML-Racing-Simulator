"""
train_ai.py  —  NEAT neuroevolution trainer for Drift Racing
=============================================================
GPU acceleration: uses PyTorch (CUDA) if available, else numpy.
Run `pip install torch` for GPU support.

Controls (Training)
-------------------
  SPACE      pause / resume
  R          restart current generation
  W          open replay menu
  F11        toggle fullscreen
  E          export population to CSV
  I          import population from CSV
  ESC        back to track menu
  +/-        speed up / slow down simulation
  TAB        cycle track

Inputs (12, all ego-relative — generalises across tracks):
  0-6   ray distances (normalised 0-1)
  7     speed (normalised 0-1)
  8     angle to next checkpoint relative to heading (normalised -1..1)
  9     distance to next checkpoint (normalised 0-1)
  10    on_track flag (0 or 1)
  11    drift angle (normalised -1..1)
"""

import pygame
import math
import sys
import csv
import os
import random
import copy
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

import numpy as np

# ── Optional GPU via PyTorch ──────────────────────────────────────────────────
try:
    import torch
    _DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _USE_TORCH = True
    print(f"[AI] PyTorch {torch.__version__} — device: {_DEVICE}")
except ImportError:
    _USE_TORCH = False
    _DEVICE    = None
    print("[AI] PyTorch not found — using numpy (CPU). "
          "Install torch for GPU acceleration.")

# ── import shared sim code ────────────────────────────────────────────────────
from drift_racing import (
    TRACKS, Track, Car, CarState,
    WIDTH, HEIGHT, FPS,
    GRASS, TRACK_DARK, BORDER_COL, CAR_RED, CAR_ACCENT,
    WHITE, BLACK, CYAN, ORANGE, GREY,
    normalize, lerp, clamp,
)

PANEL_W = 380
BASE_W  = WIDTH + PANEL_W
BASE_H  = HEIGHT

# ─────────────────────────────────────────────────────────────────────────────
#  Hyper-parameters (population size set via slider in menu)
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_POPULATION = 30
MAX_EPISODE_TIME   = 15.0
DEATH_FITNESS      = -250.0
CHECKPOINT_REWARD  = 50.0
SKIP_PENALTY       = 150.0
TIME_BONUS_MAX     = 30.0
OFFTRACK_PENALTY   = 0.15
STILL_PENALTY      = 4.0
STILL_SPEED_THR    = 15.0
SIM_SPEEDS         = [1, 2, 4, 8, 20, 60]
PHYSICS_DT         = 1 / 60.0

# ─────────────────────────────────────────────────────────────────────────────
#  Checkpoint builder
# ─────────────────────────────────────────────────────────────────────────────
NUM_CHECKPOINTS = 16

def build_checkpoints(track: Track) -> List[Tuple[float, float]]:
    cl   = track.centerline
    n    = len(cl) - 1
    step = max(1, n // NUM_CHECKPOINTS)
    return [cl[i * step] for i in range(NUM_CHECKPOINTS)]

# ─────────────────────────────────────────────────────────────────────────────
#  Ray-cast sensor
# ─────────────────────────────────────────────────────────────────────────────
RAY_ANGLES = [-90, -45, -20, 0, 20, 45, 90]
RAY_LEN    = 360.0
RAY_STEPS  = 8

def cast_rays(car: Car, track: Track) -> List[float]:
    cx, cy, angle = car.s.x, car.s.y, car.s.angle
    out = []
    for deg in RAY_ANGLES:
        rad = angle + math.radians(deg)
        rdx, rdy = math.cos(rad), math.sin(rad)
        dist = RAY_LEN
        for k in range(1, RAY_STEPS + 1):
            t  = k / RAY_STEPS
            px = cx + rdx * RAY_LEN * t
            py = cy + rdy * RAY_LEN * t
            if not track.point_on_track(px, py, margin=0):
                dist = RAY_LEN * (k - 1) / RAY_STEPS
                break
        out.append(dist / RAY_LEN)
    return out

# ─────────────────────────────────────────────────────────────────────────────
#  NEAT
#  12 inputs: 7 rays + speed + cp_angle + cp_dist + drift_angle
#  3 outputs: throttle, brake, steer
# ─────────────────────────────────────────────────────────────────────────────
N_INPUTS  = 11
N_OUTPUTS = 3

@dataclass
class Gene:
    in_node:  int
    out_node: int
    weight:   float
    enabled:  bool = True
    innov:    int  = 0

@dataclass
class Genome:
    genes:    List[Gene] = field(default_factory=list)
    fitness:  float      = 0.0
    n_hidden: int        = 0

    def copy(self):
        return Genome(
            genes    = [copy.copy(g) for g in self.genes],
            fitness  = self.fitness,
            n_hidden = self.n_hidden,
        )

_INNOV = [N_INPUTS * N_OUTPUTS]

def _new_innov() -> int:
    _INNOV[0] += 1
    return _INNOV[0]

def make_minimal_genome() -> Genome:
    genes, k = [], 0
    for i in range(N_INPUTS):
        for o in range(N_OUTPUTS):
            k += 1
            genes.append(Gene(i, N_INPUTS + o, random.gauss(0, 0.8), True, k))
    return Genome(genes=genes)

# ─────────────────────────────────────────────────────────────────────────────
#  NeuralNet — topological sort + numpy/torch forward
# ─────────────────────────────────────────────────────────────────────────────
class NeuralNet:
    def __init__(self, g: Genome):
        self.n_total = N_INPUTS + N_OUTPUTS + g.n_hidden
        n = self.n_total
        W = np.zeros((n, n), dtype=np.float32)
        for gene in g.genes:
            if gene.enabled and 0 <= gene.in_node < n and 0 <= gene.out_node < n:
                W[gene.out_node, gene.in_node] = gene.weight
        self.W     = W
        self._topo = self._topo_sort(g)
        if _USE_TORCH:
            self.W_t = torch.tensor(W, device=_DEVICE)

    @staticmethod
    def _topo_sort(g: Genome) -> List[int]:
        n      = N_INPUTS + N_OUTPUTS + g.n_hidden
        in_deg = [0] * n
        adj    = [[] for _ in range(n)]
        for gene in g.genes:
            if gene.enabled and gene.in_node < n and gene.out_node < n:
                adj[gene.in_node].append(gene.out_node)
                in_deg[gene.out_node] += 1
        queue = list(range(N_INPUTS))
        order = []
        while queue:
            node = queue.pop(0)
            order.append(node)
            for dst in adj[node]:
                in_deg[dst] -= 1
                if in_deg[dst] == 0:
                    queue.append(dst)
        return [nd for nd in order if nd >= N_INPUTS]

    def forward(self, inputs: List[float]) -> List[float]:
        v = np.zeros(self.n_total, dtype=np.float32)
        v[:N_INPUTS] = inputs[:N_INPUTS]
        for node in self._topo:
            v[node] = np.tanh(np.dot(self.W[node], v))
        return v[N_INPUTS: N_INPUTS + N_OUTPUTS].tolist()

    def forward_batch(self, inputs_batch: np.ndarray) -> np.ndarray:
        batch = inputs_batch.shape[0]
        if _USE_TORCH:
            v = torch.zeros(batch, self.n_total, device=_DEVICE)
            v[:, :N_INPUTS] = torch.tensor(inputs_batch, device=_DEVICE)
            for node in self._topo:
                v[:, node] = torch.tanh(v @ self.W_t[node])
            return v[:, N_INPUTS: N_INPUTS + N_OUTPUTS].cpu().numpy()
        else:
            v = np.zeros((batch, self.n_total), dtype=np.float32)
            v[:, :N_INPUTS] = inputs_batch
            for node in self._topo:
                v[:, node] = np.tanh(v @ self.W[node])
            return v[:, N_INPUTS: N_INPUTS + N_OUTPUTS]

# ── Mutation ──────────────────────────────────────────────────────────────────
def mutate(g: Genome) -> Genome:
    g     = g.copy()
    total = N_INPUTS + N_OUTPUTS + g.n_hidden
    if random.random() < 0.80:
        for gene in g.genes:
            if random.random() < 0.90:
                gene.weight += random.gauss(0, 0.25)
            else:
                gene.weight = random.gauss(0, 1.0)
            gene.weight = clamp(gene.weight, -8.0, 8.0)
    if random.random() < 0.06 and total > 1:
        src = random.randint(0, total - 1)
        dst = random.randint(N_INPUTS, total - 1)
        if src != dst and not any(gn.in_node == src and gn.out_node == dst
                                   for gn in g.genes):
            g.genes.append(Gene(src, dst, random.gauss(0, 1), True, _new_innov()))
    if random.random() < 0.12 and g.genes:
        cands = [gn for gn in g.genes if gn.enabled]
        if cands:
            old = random.choice(cands)
            old.enabled = False
            nid = N_INPUTS + N_OUTPUTS + g.n_hidden
            g.n_hidden += 1
            g.genes.append(Gene(old.in_node, nid,          1.0,        True, _new_innov()))
            g.genes.append(Gene(nid,          old.out_node, old.weight, True, _new_innov()))
    if random.random() < 0.01 and g.genes:
        random.choice(g.genes).enabled ^= True
    return g

def crossover(a: Genome, b: Genome) -> Genome:
    if b.fitness > a.fitness:
        a, b = b, a
    child = a.copy()
    b_map = {gn.innov: gn for gn in b.genes}
    for gn in child.genes:
        if gn.innov in b_map and random.random() < 0.5:
            gn.weight = b_map[gn.innov].weight
    return child

# ─────────────────────────────────────────────────────────────────────────────
#  Agent
# ─────────────────────────────────────────────────────────────────────────────
CP_DIST_NORM = RAY_LEN * 3.0

def _angle_toward(fx, fy, tx, ty) -> float:
    return math.atan2(ty - fy, tx - fx)

class Agent:
    def __init__(self, genome: Genome, track: Track, checkpoints: List):
        self.genome      = genome
        self.net         = NeuralNet(genome)
        self.checkpoints = checkpoints
        sx, sy = track.start_pos
        cp0    = checkpoints[0]
        self.car = Car(sx, sy, angle=_angle_toward(sx, sy, cp0[0], cp0[1]))
        self.fitness  = 0.0
        self.alive    = True
        self.cp_index = 0
        self.cp_time  = 0.0
        self.elapsed  = 0.0
        self.laps     = 0

    def reset(self, track: Track):
        sx, sy = track.start_pos
        cp0    = self.checkpoints[0]
        self.car.reset(sx, sy, angle=_angle_toward(sx, sy, cp0[0], cp0[1]))
        self.fitness  = 0.0
        self.alive    = True
        self.cp_index = 0
        self.cp_time  = 0.0
        self.elapsed  = 0.0
        self.laps     = 0

    def _build_inputs(self, track: Track) -> List[float]:
        rays = cast_rays(self.car, track)
        s    = self.car.s
        spd  = clamp(s.speed / Car.MAX_SPEED, 0.0, 1.0)
        n_cp = len(self.checkpoints)
        cp   = self.checkpoints[self.cp_index % n_cp]
        cdx, cdy    = cp[0] - s.x, cp[1] - s.y
        world_angle = math.atan2(cdy, cdx)
        rel_angle   = (world_angle - s.angle + math.pi) % (2 * math.pi) - math.pi
        cp_ang      = rel_angle / math.pi
        cp_dist     = clamp(math.hypot(cdx, cdy) / CP_DIST_NORM, 0.0, 1.0)
        drift       = clamp(s.drift_angle / math.pi, -1.0, 1.0)
        return rays + [spd, cp_ang, cp_dist, drift]

    def _build_inputs_np(self, track: Track) -> np.ndarray:
        return np.array(self._build_inputs(track), dtype=np.float32)

    def apply_output(self, out: np.ndarray, track: Track):
        if not self.alive:
            return
        self.elapsed += PHYSICS_DT
        self.cp_time += PHYSICS_DT
        thr   = float(clamp(out[0], 0.0, 1.0))
        brk   = float(clamp(out[1], 0.0, 1.0))
        steer = float(clamp(out[2], -1.0, 1.0))
        self.car.update(PHYSICS_DT, thr, brk, steer, track)
        s = self.car.s
        if not s.on_track:
            self.fitness -= OFFTRACK_PENALTY
        if s.speed < STILL_SPEED_THR:
            self.fitness -= STILL_PENALTY * PHYSICS_DT
        n_cp   = len(self.checkpoints)
        cp_pos = self.checkpoints[self.cp_index % n_cp]
        dist   = math.hypot(s.x - cp_pos[0], s.y - cp_pos[1])
        if dist < 65:
            bonus          = max(0.0, TIME_BONUS_MAX - self.cp_time)
            self.fitness  += CHECKPOINT_REWARD + bonus
            self.cp_index += 1
            self.cp_time   = 0.0
            if self.cp_index >= n_cp:
                self.cp_index = 0
                self.laps    += 1
        else:
            for j in range(1, 5):
                wi = (self.cp_index + j) % n_cp
                if math.hypot(s.x - self.checkpoints[wi][0],
                              s.y - self.checkpoints[wi][1]) < 65:
                    self.fitness -= SKIP_PENALTY
                    break
        if self.car.s.on_track == False or self.elapsed > MAX_EPISODE_TIME:
            self.alive          = False
            self.genome.fitness = self.fitness

    def step(self, track: Track):
        if not self.alive:
            return
        inp = self._build_inputs(track)
        out = self.net.forward(inp)
        self.apply_output(np.array(out, dtype=np.float32), track)

# ─────────────────────────────────────────────────────────────────────────────
#  Population manager
# ─────────────────────────────────────────────────────────────────────────────
class Population:
    def __init__(self, size: int):
        self.genomes    = [make_minimal_genome() for _ in range(size)]
        self.generation = 0
        self.best_ever: Optional[Genome] = None
        self.best_fitness_history: List[float] = []
        self.gen_history: List[Dict] = []   # {gen, genome, fitness}

    def evolve(self):
        self.genomes.sort(key=lambda g: g.fitness, reverse=True)
        best = self.genomes[0]
        self.best_fitness_history.append(best.fitness)
        if self.best_ever is None or best.fitness > self.best_ever.fitness:
            self.best_ever = best.copy()
        self.gen_history.append({
            'gen':     self.generation,
            'genome':  best.copy(),
            'fitness': best.fitness,
        })
        n     = len(self.genomes)
        keep  = max(2, n // 5)
        elite = [self.genomes[i].copy() for i in range(keep)]
        new_gen = [elite[0]]
        while len(new_gen) < n:
            if random.random() < 0.75 and keep >= 2:
                child = mutate(crossover(*random.sample(elite, 2)))
            else:
                child = mutate(random.choice(elite).copy())
            new_gen.append(child)
        self.genomes    = new_gen
        self.generation += 1

    def to_csv_rows(self) -> List[List]:
        rows = [["generation","genome_idx","fitness","n_hidden",
                 "gene_in","gene_out","weight","enabled","innov"]]
        for gi, genome in enumerate(self.genomes):
            if not genome.genes:
                rows.append([self.generation, gi, genome.fitness,
                              genome.n_hidden, "","","","",""])
            for gn in genome.genes:
                rows.append([self.generation, gi, genome.fitness,
                              genome.n_hidden, gn.in_node, gn.out_node,
                              f"{gn.weight:.6f}", int(gn.enabled), gn.innov])
        return rows

    @staticmethod
    def from_csv_rows(rows: List[List]) -> "Population":
        genome_map: Dict[int, Genome] = {}
        generation = 0
        for row in rows[1:]:
            if len(row) < 4 or row[0] == "generation":
                continue
            try:
                gi  = int(row[1]); fit = float(row[2]); n_h = int(row[3])
                generation = max(generation, int(row[0]))
            except (ValueError, IndexError):
                continue
            if gi not in genome_map:
                genome_map[gi] = Genome(fitness=fit, n_hidden=n_h)
            try:
                if row[4] != "":
                    genome_map[gi].genes.append(Gene(
                        int(row[4]), int(row[5]), float(row[6]),
                        bool(int(row[7])), int(row[8])))
            except (ValueError, IndexError):
                pass
        pop = Population.__new__(Population)
        pop.genomes    = list(genome_map.values()) or [make_minimal_genome()]
        pop.generation = generation
        pop.best_ever  = None
        pop.best_fitness_history = []
        pop.gen_history = []
        return pop

# ─────────────────────────────────────────────────────────────────────────────
#  Neural-network visualiser
# ─────────────────────────────────────────────────────────────────────────────
IN_LABELS  = ["R-90","R-45","R-20","R0","R+20","R+45","R+90",
               "SPD","CP_ANG","CP_DST","DRFT"]
OUT_LABELS = ["THR","BRK","STR"]

def _effective_n_hidden(genome: Genome) -> int:
    """Infer hidden node count from gene indices so we always show all nodes."""
    if not genome.genes:
        return genome.n_hidden
    max_idx = max(max(gn.in_node, gn.out_node) for gn in genome.genes)
    inferred = max_idx - N_INPUTS - N_OUTPUTS + 1
    return max(genome.n_hidden, inferred, 0)


def draw_nn(surf: pygame.Surface, genome: Genome, rect: pygame.Rect,
            last_inputs: List[float], last_outputs: List[float]):
    pygame.draw.rect(surf, (18, 18, 26), rect)
    pygame.draw.rect(surf, (60, 60, 80), rect, 1)
    fn    = pygame.font.SysFont("consolas", 10)
    title = pygame.font.SysFont("consolas", 13, bold=True)
    surf.blit(title.render("NEURAL NETWORK", True, CYAN), (rect.x + 8, rect.y + 6))

    n_hidden = _effective_n_hidden(genome)
    total    = N_INPUTS + N_OUTPUTS + n_hidden
    layers   = [list(range(N_INPUTS))]
    if n_hidden:
        layers.append(list(range(N_INPUTS, N_INPUTS + n_hidden)))
    layers.append(list(range(N_INPUTS + n_hidden, total)))

    pad_x, pad_y = rect.x + 14, rect.y + 28
    w, h = rect.width - 28, rect.height - 50
    n_layers = len(layers)

    positions: Dict[int, Tuple[int, int]] = {}
    for li, layer in enumerate(layers):
        lx = pad_x + int(li / max(n_layers - 1, 1) * w)
        for ni, node in enumerate(layer):
            ly = pad_y + int(ni / max(len(layer), 1) * h) + h // max(len(layer) * 2, 1)
            positions[node] = (lx, ly)

    max_w = max((abs(gn.weight) for gn in genome.genes), default=1.0)
    for gn in genome.genes:
        if not gn.enabled: continue
        if gn.in_node not in positions or gn.out_node not in positions: continue
        p1, p2 = positions[gn.in_node], positions[gn.out_node]
        alpha  = int(clamp(abs(gn.weight) / max(max_w, 0.01), 0.1, 1.0) * 180)
        col    = (50, 200, 50, alpha) if gn.weight > 0 else (200, 50, 50, alpha)
        ls     = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
        pygame.draw.line(ls, col, (p1[0]-rect.x, p1[1]-rect.y),
                         (p2[0]-rect.x, p2[1]-rect.y), 1)
        surf.blit(ls, (rect.x, rect.y))

    def safe_col(*c):
        return tuple(int(clamp(v, 0, 255)) for v in c)

    for node, (nx, ny) in positions.items():
        if node < N_INPUTS:
            val   = clamp(last_inputs[node] if node < len(last_inputs) else 0.0, 0.0, 1.0)
            label = IN_LABELS[node] if node < len(IN_LABELS) else f"I{node}"
            col   = safe_col(lerp(30, 0, val), lerp(140, 220, val), lerp(180, 255, val))
        elif node < N_INPUTS + n_hidden:
            label = f"H{node - N_INPUTS}"
            col   = (180, 180, 60)
        else:
            oi    = node - N_INPUTS - n_hidden
            raw   = last_outputs[oi] if oi < len(last_outputs) else 0.0
            val   = clamp((raw + 1) / 2, 0.0, 1.0)
            label = OUT_LABELS[oi] if oi < len(OUT_LABELS) else f"O{oi}"
            col   = safe_col(lerp(50, 220, val), lerp(200, 60, val), 80)
        pygame.draw.circle(surf, col, (nx, ny), 7)
        pygame.draw.circle(surf, WHITE, (nx, ny), 7, 1)
        surf.blit(fn.render(label, True, (200, 200, 200)), (nx + 9, ny - 6))

# ─────────────────────────────────────────────────────────────────────────────
#  Stats / fitness graph
# ─────────────────────────────────────────────────────────────────────────────
def draw_stats(surf, pop, agents, rect, sim_speed, pop_size):
    pygame.draw.rect(surf, (14, 14, 20), rect)
    pygame.draw.rect(surf, (50, 50, 70), rect, 1)
    fn_b = pygame.font.SysFont("consolas", 13, bold=True)
    fn   = pygame.font.SysFont("consolas", 12)
    alive  = sum(1 for a in agents if a.alive)
    best_f = max((a.fitness for a in agents), default=0.0)
    best_cp= max((a.cp_index for a in agents), default=0)
    lines  = [
        ("TRAINING STATS",                              CYAN,   fn_b),
        (f"Generation  : {pop.generation}",             WHITE,  fn),
        (f"Population  : {pop_size}",                   WHITE,  fn),
        (f"Alive       : {alive}",                      (100,255,100) if alive else GREY, fn),
        (f"Best fitness: {best_f:+.1f}",                ORANGE, fn),
        (f"Best CP     : {best_cp}",                    ORANGE, fn),
        (f"All-time    : {pop.best_ever.fitness:.1f}"
          if pop.best_ever else "All-time    : --",     GREY,   fn),
        (f"Sim speed   : x{sim_speed}",                 (180,180,255), fn),
    ]
    y = rect.y + 8
    for txt, col, f in lines:
        surf.blit(f.render(txt, True, col), (rect.x + 8, y))
        y += 16
    history = pop.best_fitness_history[-40:]
    if len(history) >= 2:
        gx, gy = rect.x + 8, y + 8
        gw, gh = rect.width - 16, 55
        pygame.draw.rect(surf, (25,25,35), (gx, gy, gw, gh))
        pygame.draw.rect(surf, (60,60,80), (gx, gy, gw, gh), 1)
        mn = min(history); mx = max(history); rng = max(mx - mn, 1.0)
        pts = []
        for i, v in enumerate(history):
            px = gx + int(i / (len(history)-1) * gw)
            py = gy + gh - int((v - mn) / rng * gh)
            pts.append((px, py))
        pygame.draw.lines(surf, ORANGE, False, pts, 2)
        surf.blit(fn.render(f"{mx:.0f}", True, GREY), (gx+2, gy+2))
        surf.blit(fn.render(f"{mn:.0f}", True, GREY), (gx+2, gy+gh-14))

# ─────────────────────────────────────────────────────────────────────────────
#  UI Widgets
# ─────────────────────────────────────────────────────────────────────────────
class Button:
    def __init__(self, x, y, w, h, label, color=(60, 60, 100)):
        self.rect  = pygame.Rect(x, y, w, h)
        self.label = label
        self.color = color
        self.hover = False
        self.fn    = pygame.font.SysFont("consolas", 13, bold=True)

    def update(self, mx, my):
        self.hover = self.rect.collidepoint(mx, my)

    def draw(self, surf):
        c = tuple(min(255, v+30) for v in self.color) if self.hover else self.color
        pygame.draw.rect(surf, c, self.rect, border_radius=4)
        pygame.draw.rect(surf, CYAN if self.hover else GREY, self.rect, 1, border_radius=4)
        txt = self.fn.render(self.label, True, WHITE)
        surf.blit(txt, (self.rect.centerx - txt.get_width()//2,
                        self.rect.centery - txt.get_height()//2))

    def clicked(self, event):
        return (event.type == pygame.MOUSEBUTTONDOWN and
                event.button == 1 and self.rect.collidepoint(event.pos))


class Slider:
    """Horizontal integer slider."""
    def __init__(self, x, y, w, label, min_val, max_val, value):
        self.x, self.y, self.w = x, y, w
        self.h       = 28
        self.label   = label
        self.min_val = min_val
        self.max_val = max_val
        self.value   = int(value)
        self._drag   = False
        self.fn_b    = pygame.font.SysFont("consolas", 13, bold=True)

    def _val_from_x(self, mx):
        t = clamp((mx - self.x) / max(self.w, 1), 0.0, 1.0)
        return int(round(self.min_val + t * (self.max_val - self.min_val)))

    def handle(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            r = pygame.Rect(self.x, self.y, self.w, self.h)
            if r.collidepoint(event.pos):
                self._drag = True
                self.value = self._val_from_x(event.pos[0])
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            self._drag = False
        elif event.type == pygame.MOUSEMOTION and self._drag:
            self.value = self._val_from_x(event.pos[0])

    def draw(self, surf):
        cy   = self.y + self.h // 2
        # track bar
        pygame.draw.rect(surf, (60,60,80),
                         pygame.Rect(self.x, cy-3, self.w, 6), border_radius=3)
        # filled
        t  = (self.value - self.min_val) / max(self.max_val - self.min_val, 1)
        fw = int(t * self.w)
        pygame.draw.rect(surf, CYAN,
                         pygame.Rect(self.x, cy-3, fw, 6), border_radius=3)
        # handle knob
        hx = self.x + fw
        pygame.draw.circle(surf, WHITE, (hx, cy), 9)
        pygame.draw.circle(surf, CYAN,  (hx, cy), 9, 2)
        # label above
        lbl = self.fn_b.render(f"{self.label}: {self.value}", True, WHITE)
        surf.blit(lbl, (self.x + self.w//2 - lbl.get_width()//2, self.y - 20))

# ─────────────────────────────────────────────────────────────────────────────
#  Track selection menu — 2 rows of 5 cards + population slider
# ─────────────────────────────────────────────────────────────────────────────
CARD_COLS = 5
CARD_W    = 190
CARD_H    = 108
CARD_GAP  = 14

class AITrackMenu:
    def __init__(self, tracks: List[Track]):
        self.tracks = tracks
        self.sel    = 0
        self.fn_t   = pygame.font.SysFont("consolas", 36, bold=True)
        self.fn_m   = pygame.font.SysFont("consolas", 15)
        self.fn_s   = pygame.font.SysFont("consolas", 13)

        total_w = CARD_COLS * CARD_W + (CARD_COLS-1) * CARD_GAP
        ox      = WIDTH//2 - total_w//2
        oy      = 110
        n_rows  = math.ceil(len(tracks) / CARD_COLS)
        self._rects: List[pygame.Rect] = []
        for i in range(len(tracks)):
            row = i // CARD_COLS
            col = i %  CARD_COLS
            x   = ox + col * (CARD_W + CARD_GAP)
            y   = oy + row * (CARD_H + 40)
            self._rects.append(pygame.Rect(x, y, CARD_W, CARD_H + 24))

        # Slider & button below the cards
        bottom_y = oy + n_rows * (CARD_H + 40) + 18
        self.slider = Slider(WIDTH//2 - 200, bottom_y + 22, 400,
                             "Population", 1, 50, DEFAULT_POPULATION)
        self.btn_go = Button(WIDTH//2 - 110, bottom_y + 72, 220, 44,
                             "START TRAINING", color=(30,100,30))

    def _n_rows(self):
        return math.ceil(len(self.tracks) / CARD_COLS)

    def handle(self, event) -> int:
        self.slider.handle(event)
        if event.type == pygame.KEYDOWN:
            k  = event.key
            n  = len(self.tracks)
            if k == pygame.K_LEFT:
                self.sel = (self.sel - 1) % n
            elif k == pygame.K_RIGHT:
                self.sel = (self.sel + 1) % n
            elif k == pygame.K_UP:
                new = self.sel - CARD_COLS
                if new >= 0:
                    self.sel = new
            elif k == pygame.K_DOWN:
                new = self.sel + CARD_COLS
                if new < n:
                    self.sel = new
            elif k in (pygame.K_RETURN, pygame.K_SPACE):
                return self.sel
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            for i, r in enumerate(self._rects):
                if r.collidepoint(event.pos):
                    if i == self.sel:
                        return i        # second click on same card = start
                    self.sel = i
                    return -1
        if self.btn_go.clicked(event):
            return self.sel
        return -1

    def get_population_size(self) -> int:
        return self.slider.value

    def draw(self, surf):
        surf.fill((12, 12, 18))
        title = self.fn_t.render("AI TRAINER — SELECT TRACK", True, ORANGE)
        surf.blit(title, (WIDTH//2 - title.get_width()//2, 28))
        sub = self.fn_s.render(
            "← → ↑ ↓  navigate   ENTER / click card twice to start", True, GREY)
        surf.blit(sub, (WIDTH//2 - sub.get_width()//2, 74))

        for i, (track, r) in enumerate(zip(self.tracks, self._rects)):
            sel = (i == self.sel)
            bc  = ORANGE if sel else (55, 55, 70)
            pygame.draw.rect(surf, (30,30,45) if sel else (18,18,28), r, border_radius=6)
            pygame.draw.rect(surf, bc, r, 1+sel, border_radius=6)

            # Mini preview inside card
            preview_rect = pygame.Rect(r.x+4, r.y+4, r.width-8, CARD_H-8)
            pts = track.centerline
            if pts:
                xs2 = [p[0] for p in pts]; ys2 = [p[1] for p in pts]
                mnx, mxx = min(xs2), max(xs2)
                mny, mxy = min(ys2), max(ys2)
                sc  = min((preview_rect.width  - 8) / max(mxx-mnx, 1),
                          (preview_rect.height - 8) / max(mxy-mny, 1))
                ocx = (mxx+mnx)/2; ocy = (mxy+mny)/2
                cx0 = preview_rect.centerx
                cy0 = preview_rect.centery
                def tp(px, py):
                    return (int((px-ocx)*sc+cx0), int((py-ocy)*sc+cy0))
                for seg in track.segments:
                    la, lb, rb, ra = seg
                    pygame.draw.polygon(surf, (65,65,65),
                                        [tp(*la),tp(*lb),tp(*rb),tp(*ra)])
                le2 = [tp(*p) for p in track._left_edge]
                re2 = [tp(*p) for p in track._right_edge]
                if len(le2) >= 2:
                    pygame.draw.lines(surf, BORDER_COL, True, le2, 1)
                    pygame.draw.lines(surf, BORDER_COL, True, re2, 1)

            nm = self.fn_m.render(track.name, True, WHITE if sel else GREY)
            surf.blit(nm, (r.x + r.width//2 - nm.get_width()//2,
                           r.y + CARD_H + 4))

        mx, my = pygame.mouse.get_pos()
        self.btn_go.update(mx, my)
        self.slider.draw(surf)
        self.btn_go.draw(surf)

        hint = self.fn_s.render("F11 fullscreen   ESC quit", True, (70,70,70))
        surf.blit(hint, (WIDTH//2 - hint.get_width()//2, HEIGHT - 24))

# ─────────────────────────────────────────────────────────────────────────────
#  Replay menu
# ─────────────────────────────────────────────────────────────────────────────
class ReplayMenu:
    VISIBLE = 14

    def __init__(self, pop: Population):
        self.pop      = pop
        self.scroll   = max(0, len(pop.gen_history) - self.VISIBLE)
        self.sel      = max(0, len(pop.gen_history) - 1)
        self.fn_b     = pygame.font.SysFont("consolas", 15, bold=True)
        self.fn       = pygame.font.SysFont("consolas", 14)
        self.fn_s     = pygame.font.SysFont("consolas", 12)
        self.btn_back = Button(WIDTH//2 - 100, HEIGHT - 54, 200, 38,
                               "< BACK", color=(60,30,10))

    def handle(self, event) -> Optional[int]:
        self.btn_back.update(*pygame.mouse.get_pos())
        if event.type == pygame.KEYDOWN:
            k = event.key
            if k in (pygame.K_ESCAPE, pygame.K_w):
                return -1
            elif k == pygame.K_UP:
                self.sel = max(0, self.sel - 1)
                if self.sel < self.scroll:
                    self.scroll = self.sel
            elif k == pygame.K_DOWN:
                mx_sel = max(0, len(self.pop.gen_history) - 1)
                self.sel = min(mx_sel, self.sel + 1)
                if self.sel >= self.scroll + self.VISIBLE:
                    self.scroll = self.sel - self.VISIBLE + 1
            elif k in (pygame.K_RETURN, pygame.K_SPACE):
                if self.pop.gen_history:
                    return self.sel
        elif event.type == pygame.MOUSEWHEEL:
            self.scroll = int(clamp(self.scroll - event.y, 0,
                                    max(0, len(self.pop.gen_history) - self.VISIBLE)))
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.btn_back.clicked(event):
                return -1
            y0 = 120
            for vi in range(self.VISIBLE):
                gi = self.scroll + vi
                if gi >= len(self.pop.gen_history):
                    break
                row_r = pygame.Rect(WIDTH//2 - 340, y0 + vi*34, 680, 30)
                if row_r.collidepoint(event.pos):
                    if gi == self.sel:
                        return gi
                    self.sel = gi
        return None

    def draw(self, surf):
        surf.fill((10, 10, 16))
        title = self.fn_b.render("GENERATION REPLAY", True, CYAN)
        surf.blit(title, (WIDTH//2 - title.get_width()//2, 20))

        if not self.pop.gen_history:
            msg = self.fn.render("No completed generations yet — train first!", True, GREY)
            surf.blit(msg, (WIDTH//2 - msg.get_width()//2, HEIGHT//2))
            self.btn_back.draw(surf)
            return

        sub = self.fn_s.render(
            "↑ ↓ / scroll  |  ENTER or double-click to watch  |  ESC / W back", True, GREY)
        surf.blit(sub, (WIDTH//2 - sub.get_width()//2, 50))

        hx = WIDTH//2 - 340
        surf.blit(self.fn_s.render("GEN",          True, GREY), (hx,       86))
        surf.blit(self.fn_s.render("BEST FITNESS", True, GREY), (hx+80,    86))
        surf.blit(self.fn_s.render("HIDDEN NODES", True, GREY), (hx+260,   86))
        surf.blit(self.fn_s.render("CONNECTIONS",  True, GREY), (hx+440,   86))
        pygame.draw.line(surf, (60,60,80), (hx, 104), (hx+680, 104), 1)

        y0 = 120
        for vi in range(self.VISIBLE):
            gi = self.scroll + vi
            if gi >= len(self.pop.gen_history):
                break
            entry  = self.pop.gen_history[gi]
            ry     = y0 + vi*34
            sel    = (gi == self.sel)
            if sel:
                pygame.draw.rect(surf, (30,50,80), (hx-6, ry-3, 692, 30), border_radius=4)
                pygame.draw.rect(surf, CYAN,        (hx-6, ry-3, 692, 30), 1, border_radius=4)
            col    = WHITE if sel else GREY
            g      = entry['genome']
            n_conn = sum(1 for gn in g.genes if gn.enabled)
            surf.blit(self.fn.render(str(entry['gen']),          True, col),   (hx,       ry))
            surf.blit(self.fn.render(f"{entry['fitness']:+.1f}", True, ORANGE if sel else col),
                      (hx+80, ry))
            surf.blit(self.fn.render(str(g.n_hidden), True, col), (hx+260, ry))
            surf.blit(self.fn.render(str(n_conn),     True, col), (hx+440, ry))
            if sel:
                hint = self.fn_s.render("↵ Watch", True, CYAN)
                surf.blit(hint, (hx+580, ry+4))

        self.btn_back.draw(surf)

# ─────────────────────────────────────────────────────────────────────────────
#  Main Trainer window
# ─────────────────────────────────────────────────────────────────────────────
class Trainer:
    MENU        = "menu"
    TRAINING    = "training"
    REPLAY_MENU = "replay_menu"
    WATCHING    = "watching"

    def __init__(self):
        pygame.init()
        self._fullscreen = False
        self.canvas      = pygame.Surface((BASE_W, BASE_H))
        flags            = pygame.RESIZABLE | pygame.SCALED
        self.screen      = pygame.display.set_mode((BASE_W, BASE_H), flags)
        pygame.display.set_caption("Drift Racing — AI Trainer (NEAT)")
        self.clock = pygame.time.Clock()

        self.all_tracks  = [Track(d) for d in TRACKS]
        self.menu        = AITrackMenu(self.all_tracks)
        self.state       = self.MENU
        self.track       = self.all_tracks[0]
        self.pop         = Population(DEFAULT_POPULATION)
        self.pop_size    = DEFAULT_POPULATION
        self.agents: List[Agent] = []
        self.checkpoints: List   = []
        self.sim_speed_idx = 0
        self.paused        = False
        self.last_inputs   = [0.0] * N_INPUTS
        self.last_outputs  = [0.0] * N_OUTPUTS

        self.replay_menu: Optional[ReplayMenu] = None
        self.watch_agent: Optional[Agent]      = None

        bx = WIDTH + 10
        self.btn_export = Button(bx, BASE_H-104, PANEL_W-20, 36, "EXPORT CSV",        (30,70,100))
        self.btn_import = Button(bx, BASE_H- 60, PANEL_W-20, 36, "IMPORT CSV",        (70,30,100))
        self.btn_replay = Button(bx, BASE_H-148, PANEL_W-20, 36, "WATCH REPLAY [W]",  (50,30,80))
        self.btn_pause  = Button(WIDTH-132,  8, 122, 28, "PAUSE [SPC]", (60,60,20))
        self.btn_faster = Button(WIDTH-264,  8,  62, 28, "+SPEED",      (20,60,20))
        self.btn_slower = Button(WIDTH-202,  8,  62, 28, "-SPEED",      (60,20,20))

    def _toggle_fullscreen(self):
        self._fullscreen = not self._fullscreen
        try:
            pygame.display.toggle_fullscreen()
        except Exception:
            self._fullscreen = not self._fullscreen

    def _flip(self):
        if self._fullscreen:
            sw, sh  = self.screen.get_size()
            sc      = min(sw / BASE_W, sh / BASE_H)
            scaled  = pygame.transform.scale(
                self.canvas, (int(BASE_W*sc), int(BASE_H*sc)))
            self.screen.fill((0, 0, 0))
            ox = (sw - scaled.get_width())  // 2
            oy = (sh - scaled.get_height()) // 2
            self.screen.blit(scaled, (ox, oy))
        else:
            self.screen.blit(self.canvas, (0, 0))
        pygame.display.flip()

    def _start_training(self, track_idx: int):
        self.track       = self.all_tracks[track_idx]
        self.pop_size    = self.menu.get_population_size()
        self.pop         = Population(self.pop_size)
        self.checkpoints = build_checkpoints(self.track)
        self._new_generation()
        self.state = self.TRAINING

    def _new_generation(self):
        self.agents = [Agent(g, self.track, self.checkpoints)
                       for g in self.pop.genomes]

    def _all_dead(self) -> bool:
        return all(not a.alive for a in self.agents)

    def _export_csv(self):
        path = "neat_population.csv"
        with open(path, "w", newline="") as f:
            csv.writer(f).writerows(self.pop.to_csv_rows())
        print(f"[CSV] Exported {len(self.pop.genomes)} genomes → {path}")

    def _import_csv(self):
        path = "neat_population.csv"
        if not os.path.exists(path):
            print(f"[CSV] Not found: {path}"); return
        with open(path, newline="") as f:
            rows = list(csv.reader(f))
        self.pop = Population.from_csv_rows(rows)
        print(f"[CSV] Imported gen {self.pop.generation}")
        self._new_generation()

    def _open_replay(self):
        self.replay_menu = ReplayMenu(self.pop)
        self.state       = self.REPLAY_MENU

    def _watch_gen(self, idx: int):
        entry            = self.pop.gen_history[idx]
        self.watch_agent = Agent(entry['genome'], self.track, self.checkpoints)
        self._watch_entry = entry
        self.state       = self.WATCHING

    def run(self):
        while True:
            self.clock.tick(FPS)
            mx, my = pygame.mouse.get_pos()

            if self.state == self.TRAINING:
                for btn in [self.btn_export, self.btn_import, self.btn_pause,
                            self.btn_faster, self.btn_slower, self.btn_replay]:
                    btn.update(mx, my)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit(); sys.exit()

                if event.type == pygame.KEYDOWN and event.key == pygame.K_F11:
                    self._toggle_fullscreen()
                    continue

                if self.state == self.MENU:
                    chosen = self.menu.handle(event)
                    if chosen >= 0:
                        self._start_training(chosen)
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                        pygame.quit(); sys.exit()

                elif self.state == self.TRAINING:
                    if event.type == pygame.KEYDOWN:
                        k = event.key
                        if k == pygame.K_ESCAPE:
                            self.state = self.MENU
                        elif k == pygame.K_SPACE:
                            self.paused = not self.paused
                        elif k == pygame.K_r:
                            self._new_generation()
                        elif k == pygame.K_w:
                            self._open_replay()
                        elif k == pygame.K_e:
                            self._export_csv()
                        elif k == pygame.K_i:
                            self._import_csv()
                        elif k in (pygame.K_PLUS, pygame.K_EQUALS):
                            self.sim_speed_idx = min(len(SIM_SPEEDS)-1, self.sim_speed_idx+1)
                        elif k == pygame.K_MINUS:
                            self.sim_speed_idx = max(0, self.sim_speed_idx-1)
                        elif k == pygame.K_TAB:
                            cur = self.all_tracks.index(self.track)
                            self.track = self.all_tracks[(cur+1) % len(self.all_tracks)]
                            self.checkpoints = build_checkpoints(self.track)
                            self._new_generation()
                    if self.btn_export.clicked(event):  self._export_csv()
                    if self.btn_import.clicked(event):  self._import_csv()
                    if self.btn_pause.clicked(event):   self.paused = not self.paused
                    if self.btn_replay.clicked(event):  self._open_replay()
                    if self.btn_faster.clicked(event):
                        self.sim_speed_idx = min(len(SIM_SPEEDS)-1, self.sim_speed_idx+1)
                    if self.btn_slower.clicked(event):
                        self.sim_speed_idx = max(0, self.sim_speed_idx-1)

                elif self.state == self.REPLAY_MENU:
                    result = self.replay_menu.handle(event)
                    if result == -1:
                        self.state = self.TRAINING
                    elif result is not None:
                        self._watch_gen(result)

                elif self.state == self.WATCHING:
                    if event.type == pygame.KEYDOWN:
                        if event.key in (pygame.K_ESCAPE, pygame.K_w):
                            self.state = self.REPLAY_MENU

            # ── Simulation ────────────────────────────────────────────────────
            if self.state == self.TRAINING and not self.paused:
                sim_speed = SIM_SPEEDS[self.sim_speed_idx]
                for _ in range(sim_speed):
                    alive = [a for a in self.agents if a.alive]
                    if not alive:
                        break
                    inp = np.stack([a._build_inputs_np(self.track) for a in alive])
                    for i, agent in enumerate(alive):
                        out = agent.net.forward_batch(inp[i:i+1])[0]
                        agent.apply_output(out, self.track)
                if self._all_dead():
                    self.pop.evolve()
                    self._new_generation()

            elif self.state == self.WATCHING and self.watch_agent:
                if self.watch_agent.alive:
                    self.watch_agent.step(self.track)
                else:
                    self.watch_agent.reset(self.track)

            # ── Draw ──────────────────────────────────────────────────────────
            s = self.canvas
            if self.state == self.MENU:
                self.menu.draw(s)
            elif self.state == self.TRAINING:
                self._draw_training(s)
            elif self.state == self.REPLAY_MENU:
                self.replay_menu.draw(s)
            elif self.state == self.WATCHING:
                self._draw_watching(s)
            self._flip()

    # ─────────────────────────────────────────────────────────────────────────
    def _draw_training(self, surf):
        sim_speed = SIM_SPEEDS[self.sim_speed_idx]
        skip = (sim_speed >= 20 and not self.paused)

        if skip:
            surf.fill((8, 8, 12))
            fn = pygame.font.SysFont("consolas", 18, bold=True)
            alive_n = sum(1 for a in self.agents if a.alive)
            lines = [
                (f"FAST TRAINING  ×{sim_speed}",                                 CYAN),
                (f"Gen {self.pop.generation}   Alive {alive_n}/{self.pop_size}", WHITE),
                (f"Best: {max((a.fitness for a in self.agents),default=0):+.1f}", ORANGE),
                (f"All-time: {self.pop.best_ever.fitness:.1f}"
                  if self.pop.best_ever else "",                                   GREY),
                ("",                                                                WHITE),
                ("Press  -  to slow down and see visualisation",                   GREY),
                ("SPACE pause   E export   W replay   F11 fullscreen",             GREY),
            ]
            for i, (line, col) in enumerate(lines):
                if line:
                    s = fn.render(line, True, col)
                    surf.blit(s, (BASE_W//2 - s.get_width()//2, 160 + i*32))
            return

        surf.blit(self.track.surface, (0, 0))

        fn_cp = pygame.font.SysFont("consolas", 11, bold=True)
        for ci, cp in enumerate(self.checkpoints):
            cx_i, cy_i = int(cp[0]), int(cp[1])
            dot = pygame.Surface((22, 22), pygame.SRCALPHA)
            pygame.draw.circle(dot, (0,200,255, 90), (11,11), 11)
            pygame.draw.circle(dot, (0,200,255,180), (11,11), 11, 1)
            surf.blit(dot, (cx_i-11, cy_i-11))
            lbl = fn_cp.render(str(ci), True, WHITE)
            surf.blit(lbl, (cx_i - lbl.get_width()//2, cy_i - lbl.get_height()//2))

        alive_agents = [a for a in self.agents if a.alive]
        dead_agents  = [a for a in self.agents if not a.alive]

        for agent in dead_agents:
            corners = agent.car.get_corners()
            gs = pygame.Surface((40, 20), pygame.SRCALPHA)
            pygame.draw.polygon(gs, (120,120,120,60),
                                [(c[0]-agent.car.s.x+20, c[1]-agent.car.s.y+10)
                                 for c in corners])
            surf.blit(gs, (int(agent.car.s.x)-20, int(agent.car.s.y)-10))
            agent.car._draw_control_arrow(surf)

        best_alive = max(alive_agents, key=lambda a: a.fitness, default=None)

        for agent in alive_agents:
            s2      = agent.car.s
            col     = ORANGE if agent is best_alive else CYAN
            corners = agent.car.get_corners()
            pygame.draw.polygon(surf, (20,20,20), [(x+2,y+2) for x,y in corners])
            pygame.draw.polygon(surf, col, corners)
            agent.car._draw_control_arrow(surf)
            if agent is best_alive:
                self.last_inputs  = agent._build_inputs(self.track)
                self.last_outputs = agent.net.forward(self.last_inputs)
                rays = self.last_inputs[:7]
                for deg, dist in zip(RAY_ANGLES, rays):
                    rad = s2.angle + math.radians(deg)
                    ex  = s2.x + math.cos(rad)*dist*RAY_LEN
                    ey  = s2.y + math.sin(rad)*dist*RAY_LEN
                    pygame.draw.line(surf, (0,200,100),
                                     (int(s2.x),int(s2.y)),(int(ex),int(ey)),1)

        fn2 = pygame.font.SysFont("consolas", 12)
        hud = [
            (f"Gen {self.pop.generation}",                               CYAN),
            (f"Alive {sum(1 for a in self.agents if a.alive)}/{self.pop_size}", WHITE),
            (f"Best  {max((a.fitness for a in self.agents),default=0.0):+.1f}", ORANGE),
            (f"Speed ×{sim_speed}",                                      (180,180,255)),
            ("PAUSED" if self.paused else "",                             (255,80,80)),
        ]
        panel = pygame.Surface((180, len(hud)*18+12), pygame.SRCALPHA)
        panel.fill((0,0,0,160))
        surf.blit(panel, (8, 8))
        for i, (txt, col) in enumerate(hud):
            if txt:
                surf.blit(fn2.render(txt, True, col), (14, 12+i*18))

        pygame.draw.rect(surf, (10,10,16), pygame.Rect(WIDTH, 0, PANEL_W, BASE_H))

        nn_h    = int(BASE_H * 0.52)
        nn_rect = pygame.Rect(WIDTH, 0, PANEL_W, nn_h)
        best_g  = (best_alive.genome if best_alive
                   else (self.pop.best_ever or self.pop.genomes[0]))
        draw_nn(surf, best_g, nn_rect, self.last_inputs, self.last_outputs)

        stats_rect = pygame.Rect(WIDTH, nn_h, PANEL_W, BASE_H - nn_h - 200)
        draw_stats(surf, self.pop, self.agents, stats_rect, sim_speed, self.pop_size)

        for btn in [self.btn_export, self.btn_import, self.btn_replay,
                    self.btn_pause, self.btn_faster, self.btn_slower]:
            btn.draw(surf)

        fn3 = pygame.font.SysFont("consolas", 11)
        hints = ["SPC pause  R restart  TAB cycle track",
                 "E export  I import  W replay  F11 fullscreen",
                 "+/-  sim speed   ESC back to menu"]
        for i, h in enumerate(hints):
            surf.blit(fn3.render(h, True, (90,90,90)),
                      (WIDTH+10, BASE_H - 50 + i*14))

    # ─────────────────────────────────────────────────────────────────────────
    def _draw_watching(self, surf):
        surf.blit(self.track.surface, (0, 0))

        fn_cp = pygame.font.SysFont("consolas", 11, bold=True)
        for ci, cp in enumerate(self.checkpoints):
            cx_i, cy_i = int(cp[0]), int(cp[1])
            dot = pygame.Surface((22, 22), pygame.SRCALPHA)
            pygame.draw.circle(dot, (0,200,255, 90), (11,11), 11)
            pygame.draw.circle(dot, (0,200,255,180), (11,11), 11, 1)
            surf.blit(dot, (cx_i-11, cy_i-11))
            lbl = fn_cp.render(str(ci), True, WHITE)
            surf.blit(lbl, (cx_i - lbl.get_width()//2, cy_i - lbl.get_height()//2))

        agent = self.watch_agent
        if agent:
            s       = agent.car.s
            corners = agent.car.get_corners()
            pygame.draw.polygon(surf, (20,20,20), [(x+2,y+2) for x,y in corners])
            pygame.draw.polygon(surf, CAR_RED, corners)
            agent.car._draw_control_arrow(surf)

            inp  = agent._build_inputs(self.track)
            out  = agent.net.forward(inp)
            rays = inp[:7]
            for deg, dist in zip(RAY_ANGLES, rays):
                rad = s.angle + math.radians(deg)
                ex  = s.x + math.cos(rad)*dist*RAY_LEN
                ey  = s.y + math.sin(rad)*dist*RAY_LEN
                pygame.draw.line(surf, (0,200,100),
                                 (int(s.x),int(s.y)),(int(ex),int(ey)),1)

            # HUD
            fn2   = pygame.font.SysFont("consolas", 13, bold=True)
            entry = getattr(self, '_watch_entry', None)
            label = (f"Gen {entry['gen']}  Fitness: {entry['fitness']:+.1f}"
                     if entry else "Replay")
            lbl = fn2.render(f"WATCHING: {label}", True, ORANGE)
            bg  = pygame.Surface((lbl.get_width()+16, lbl.get_height()+8),
                                 pygame.SRCALPHA)
            bg.fill((0,0,0,180))
            surf.blit(bg,  (8, 8))
            surf.blit(lbl, (16, 12))
            hint = pygame.font.SysFont("consolas", 12).render(
                "ESC / W  →  back to replay list", True, GREY)
            surf.blit(hint, (16, 38))

            # NN panel
            pygame.draw.rect(surf, (10,10,16), pygame.Rect(WIDTH, 0, PANEL_W, BASE_H))
            nn_rect = pygame.Rect(WIDTH, 0, PANEL_W, int(BASE_H * 0.55))
            draw_nn(surf, agent.genome, nn_rect, inp, out)

# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    Trainer().run()