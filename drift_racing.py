"""
Drift Racing Simulator
----------------------
Controls:  W/S = throttle/brake  |  A/D = steer  |  R = reset
           F1  = toggle recording |  ESC = quit / back to menu
           TAB = cycle track in-game
"""

import pygame
import math
import sys
from dataclasses import dataclass
from typing import List, Tuple

# ── Window ───────────────────────────────────────────────────────────────────
WIDTH, HEIGHT = 1200, 800
FPS = 60

# ── Palette ──────────────────────────────────────────────────────────────────
GRASS        = (45, 120, 45)
TRACK_DARK   = (55, 55, 55)
BORDER_COL   = (255, 220, 0)
CAR_RED      = (220, 35, 35)
CAR_ACCENT   = (255, 140, 0)
WHITE        = (255, 255, 255)
BLACK        = (0,   0,   0)
CYAN         = (0,  210, 210)
ORANGE       = (255, 170,  30)
GREY         = (160, 160, 160)

# ─────────────────────────────────────────────────────────────────────────────
#  TRACK DEFINITIONS
#  Rules:
#   • Do NOT repeat the first point at the end.
#   • "steps" controls smoothness per segment (fewer = tighter corners).
#   • Street Circuit uses steps=4 so 90° bends don't overshoot.
# ─────────────────────────────────────────────────────────────────────────────

TRACKS = [
    {
        "name": "Oval Circuit",
        "width": 85,
        "start_angle": 0.0,
        "closed": True,
        "steps": 16,
        "waypoints": [
            (300, 150), (500, 120), (700, 120), (900, 150),
            (1050, 250), (1100, 380), (1050, 520), (900, 630),
            (700, 680), (500, 680), (300, 640), (160, 520),
            (110, 380), (160, 250),
        ],
    },
    {
        "name": "Technical Twisty",
        "width": 56,
        "start_angle": 0.0,
        "closed": True,
        "steps": 16,
        # Waypoints spread far enough apart that the centripetal spline
        # can't overshoot between any two consecutive points.
        "waypoints": [
            (180, 160),
            (420, 110),
            (680, 145),
            (870, 95),
            (1060, 190),
            (1090, 360),
            (920, 330),
            (740, 430),
            (840, 560),
            (1040, 575),
            (1075, 700),
            (830, 740),
            (620, 670),
            (430, 710),
            (230, 650),
            (120, 510),
            (110, 330),
        ],
    },
    {
        "name": "High-Speed Sweeper",
        "width": 95,
        "start_angle": math.pi / 2,
        "closed": True,
        "steps": 18,
        "waypoints": [
            (600,  80), (850, 100), (1050, 200), (1120, 380),
            (1050, 560), (870, 650), (680, 620), (520, 530),
            (420, 380), (520, 230),
        ],
    },
    {
        "name": "Street Circuit",
        "width": 60,
        "start_angle": 0.0,
        "closed": True,
        "steps": 10,
        # Rounded rectangle. Each corner has 3 waypoints (before/apex/after)
        # spaced far enough apart that centripetal CR stays gentle.
        # Straights have an extra mid-point so the spline doesn't bow inward.
        "waypoints": [
            # Top straight (left→right)
            (350, 170), (600, 170), (840, 170),
            # Top-right corner
            (970, 170), (1010, 210),
            # Right straight (top→bottom)
            (1010, 400),
            # Bottom-right corner
            (1010, 590), (970, 630),
            # Bottom straight (right→left)
            (840, 630), (600, 630), (350, 630),
            # Bottom-left corner
            (230, 630), (190, 590),
            # Left straight (bottom→top)
            (190, 400),
            # Top-left corner
            (190, 210), (230, 170),
        ],
    },
    {
        "name": "Figure Eight",
        "width": 62,
        "start_angle": 0.0,
        "closed": True,
        "steps": 16,
        "waypoints": [
            # Centre crossover heading right-and-down
            (600, 310),
            # Right loop
            (720, 260), (870, 220), (1000, 290),
            (1060, 400), (990, 530), (860, 590),
            (720, 660),
            # Back through centre heading left-and-up
            (600, 530),
            # Left loop
            (480, 560), (340, 590), (205, 530),
            (140, 400), (200, 285), (340, 220),
            (480, 260),
        ],
    },
    # ── Second row of tracks ────────────────────────────────────────────────
    {
        "name": "Speed Bowl",
        "width": 90,
        "closed": True,
        "steps": 18,
        "waypoints": [
            (185, 235), (430, 112), (600, 85), (770, 112), (1015, 235),
            (1115, 400), (1015, 565), (770, 688), (600, 715), (430, 688),
            (185, 565), (85, 400),
        ],
    },
    {
        "name": "Hairpin Pass",
        "width": 65,
        "closed": True,
        "steps": 14,
        "waypoints": [
            (150, 180), (420, 145), (700, 180),
            (870, 285), (715, 385), (490, 340),
            (330, 455), (150, 500),
            (150, 650), (390, 705), (660, 670),
            (850, 560), (1050, 430), (1055, 265),
            (890, 160),
        ],
    },
    {
        "name": "Stadium",
        "width": 67,
        "closed": True,
        "steps": 10,
        "waypoints": [
            (280, 155), (580, 155), (870, 155),
            (1010, 260),
            (1060, 400), (975, 350), (880, 415),
            (1010, 555),
            (870, 645), (580, 645), (280, 645),
            (160, 555),
            (160, 400),
            (160, 260),
        ],
    },
    {
        "name": "Slalom",
        "width": 72,
        "closed": True,
        "steps": 16,
        "waypoints": [
            (180, 200), (500, 140), (820, 200),
            (1000, 330), (1020, 470), (900, 570),
            (720, 530), (600, 600), (680, 700),
            (850, 720), (1050, 640), (1070, 480),
            (1060, 300), (870, 160),
        ],
    },
    {
        "name": "Cross Oval",
        "width": 76,
        "closed": True,
        "steps": 16,
        "waypoints": [
            (600, 130),
            (820, 155), (990, 255), (1070, 400), (990, 550),
            (820, 640), (600, 650),
            (420, 560), (340, 460),
            (260, 380), (200, 280), (280, 180), (440, 145),
        ],
    },
]

# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def normalize(vx, vy):
    m = math.hypot(vx, vy)
    return (vx/m, vy/m) if m > 1e-9 else (0.0, 0.0)

def lerp(a, b, t):
    return a + (b - a) * t

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def perp_left(dx, dy):
    return normalize(-dy, dx)

# ─────────────────────────────────────────────────────────────────────────────
#  Track
# ─────────────────────────────────────────────────────────────────────────────

class Track:
    def __init__(self, data: dict):
        self.name        = data["name"]
        self.raw_pts     = data["waypoints"]
        self.half_w      = data["width"]
        self.closed      = data.get("closed", True)
        steps            = data.get("steps", 14)
        self.centerline  = self._centripetal_catmull_rom(self.raw_pts, steps=steps, closed=self.closed)
        self.segments    = self._build_segments()
        self.surface     = self._render()

    # ── Centripetal Catmull-Rom (alpha=0.5) ───────────────────────────────────
    # Unlike the uniform variant, centripetal CR is mathematically guaranteed
    # never to overshoot or form self-intersecting loops between control points.
    @staticmethod
    def _centripetal_catmull_rom(pts, steps=14, closed=True):
        """
        Centripetal Catmull-Rom spline (alpha = 0.5).
        Guaranteed no cusps or self-intersections between consecutive points.
        Returns a polyline whose last point equals the first (closed loop).
        """
        ALPHA = 0.5

        def tj(ti, pi, pj):
            dx, dy = pj[0]-pi[0], pj[1]-pi[1]
            return ti + (dx*dx + dy*dy) ** (ALPHA * 0.5)

        def segment(p0, p1, p2, p3, num_steps):
            t0 = 0.0
            t1 = tj(t0, p0, p1)
            t2 = tj(t1, p1, p2)
            t3 = tj(t2, p2, p3)
            result = []
            for k in range(num_steps):
                t = t1 + (t2 - t1) * k / num_steps
                # Barry & Goldman's pyramidal formulation
                def interp(pa, pb, ta, tb):
                    if abs(tb - ta) < 1e-10:
                        return pa
                    f = (t - ta) / (tb - ta)
                    return (pa[0] + f*(pb[0]-pa[0]), pa[1] + f*(pb[1]-pa[1]))
                A1 = interp(p0, p1, t0, t1)
                A2 = interp(p1, p2, t1, t2)
                A3 = interp(p2, p3, t2, t3)
                B1 = interp(A1, A2, t0, t2)
                B2 = interp(A2, A3, t1, t3)
                C  = interp(B1, B2, t1, t2)
                result.append(C)
            return result

        out = []
        n = len(pts)

        if closed:
            for i in range(n):
                p0 = pts[(i - 1) % n]
                p1 = pts[i]
                p2 = pts[(i + 1) % n]
                p3 = pts[(i + 2) % n]
                out.extend(segment(p0, p1, p2, p3, steps))
            out.append(out[0])   # close the loop
        else:
            for i in range(n - 1):
                p0 = pts[max(0, i-1)]
                p1 = pts[i]
                p2 = pts[i+1]
                p3 = pts[min(n-1, i+2)]
                out.extend(segment(p0, p1, p2, p3, steps))
            out.append(pts[-1])

        return out

    # ── Compute simple per-point left/right offsets (NO miter) ───────────────
    # Each centerline point gets its own left/right offset based on the local
    # forward tangent only (curr→next). Using only the forward direction keeps
    # the border from "over‑reacting" to tight S‑bends, which was creating
    # tiny self‑intersecting loops in the yellow border on some tracks.
    def _compute_offset_edges(self):
        pts = self.centerline
        n   = len(pts) - 1          # pts[n] == pts[0]
        w   = self.half_w
        left_pts, right_pts = [], []

        for i in range(n):
            curr = pts[i]
            nxt  = pts[(i + 1) % n]

            # Local forward tangent; if degenerate, fall back to backward dir.
            tx = nxt[0] - curr[0]
            ty = nxt[1] - curr[1]
            m  = math.hypot(tx, ty)
            if m < 1e-9:
                prev = pts[(i - 1) % n]
                tx = curr[0] - prev[0]
                ty = curr[1] - prev[1]
                m  = math.hypot(tx, ty)
            if m < 1e-9:
                tx, ty = 1.0, 0.0
            else:
                tx, ty = tx/m, ty/m

            # Left normal (always exactly w pixels out before smoothing)
            nx, ny = -ty, tx
            left_pts.append((curr[0] + nx*w, curr[1] + ny*w))
            right_pts.append((curr[0] - nx*w, curr[1] - ny*w))

        # Light circular smoothing pass to soften harsh inside corners and
        # remove tiny loops in the visual border, without changing the
        # underlying driveable track geometry.
        def smooth_ring(pts_ring, passes=2):
            nloc = len(pts_ring)
            for _ in range(passes):
                new = []
                for i in range(nloc):
                    px, py = pts_ring[(i - 1) % nloc]
                    cx, cy = pts_ring[i]
                    nx, ny = pts_ring[(i + 1) % nloc]
                    # Simple 1D kernel: 0.25 * prev + 0.5 * curr + 0.25 * next
                    new.append((
                        (px + 2*cx + nx) * 0.25,
                        (py + 2*cy + ny) * 0.25,
                    ))
                pts_ring = new
            return pts_ring

        left_pts  = smooth_ring(left_pts)
        right_pts = smooth_ring(right_pts)

        left_pts.append(left_pts[0])
        right_pts.append(right_pts[0])
        return left_pts, right_pts

    # ── Build quad segments ───────────────────────────────────────────────────
    def _build_segments(self):
        left, right = self._compute_offset_edges()
        segs = []
        for i in range(len(left) - 1):
            segs.append((left[i], left[i+1], right[i+1], right[i]))
        self._left_edge  = left
        self._right_edge = right
        return segs

    # ── Static render ─────────────────────────────────────────────────────────
    def _render(self):
        surf = pygame.Surface((WIDTH, HEIGHT))
        surf.fill(GRASS)

        # Pass 1 — fill all track quads (no gaps, no spikes)
        for i, (lA, lB, rB, rA) in enumerate(self.segments):
            shade = 8 if (i // 6) % 2 == 0 else 0
            c = (TRACK_DARK[0]+shade, TRACK_DARK[1]+shade, TRACK_DARK[2]+shade)
            pygame.draw.polygon(surf, c, [lA, lB, rB, rA])

        def ip(pts):
            return [(int(x), int(y)) for x, y in pts]

        # Pass 2 — border lines drawn as continuous polylines (closed=True)
        # This draws ONE clean line around each edge — no per-segment gaps.
        left_ip  = ip(self._left_edge)
        right_ip = ip(self._right_edge)
        if len(left_ip) >= 2:
            pygame.draw.lines(surf, BORDER_COL, True, left_ip,  3)
            pygame.draw.lines(surf, BORDER_COL, True, right_ip, 3)

        # Checkered start/finish line
        lA, _, rB, rA = self.segments[0]
        for k in range(6):
            t0, t1 = k/6, (k+1)/6
            p0 = (int(lerp(lA[0], rA[0], t0)), int(lerp(lA[1], rA[1], t0)))
            p1 = (int(lerp(lA[0], rA[0], t1)), int(lerp(lA[1], rA[1], t1)))
            pygame.draw.line(surf, WHITE if k%2==0 else BLACK, p0, p1, 5)

        return surf

    # ── Gameplay helpers ──────────────────────────────────────────────────────
    def point_on_track(self, x, y, margin=8):
        for i in range(len(self.centerline) - 1):
            ax, ay = self.centerline[i]
            bx, by = self.centerline[i+1]
            dx, dy = bx-ax, by-ay
            seg2 = dx*dx + dy*dy
            if seg2 < 1: continue
            t = clamp(((x-ax)*dx + (y-ay)*dy) / seg2, 0.0, 1.0)
            if math.hypot(ax+t*dx-x, ay+t*dy-y) < self.half_w + margin:
                return True
        return False

    def nearest_seg_index(self, x, y):
        best, best_d = 0, float('inf')
        for i in range(len(self.centerline) - 1):
            ax, ay = self.centerline[i]
            d = math.hypot(x-ax, y-ay)
            if d < best_d:
                best_d, best = d, i
        return best

    @property
    def start_pos(self):
        return self.centerline[0]

    @property
    def start_angle(self):
        """Angle of the track centerline at start, computed from actual spline."""
        cl   = self.centerline
        look = min(4, len(cl) - 1)
        dx   = cl[look][0] - cl[0][0]
        dy   = cl[look][1] - cl[0][1]
        return math.atan2(dy, dx)

# ─────────────────────────────────────────────────────────────────────────────
#  Car
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CarState:
    x:           float
    y:           float
    angle:       float
    vx:          float = 0.0
    vy:          float = 0.0
    speed:       float = 0.0
    drift_angle: float = 0.0
    on_track:    bool  = True
    lap:         int   = 0
    lap_time:    float = 0.0
    best_lap:    float = float('inf')
    total_time:  float = 0.0
    last_seg:    int   = 0

class Car:
    ENGINE       = 620.0
    BRAKE        = 900.0
    STEER_SPD    = 2.0
    ROLL_DRAG    = 0.978
    GRASS_DRAG   = 0.86
    MAX_SPEED    = 320.0

    GRIP_FACTOR_GRIP  = 15.0
    GRIP_FACTOR_DRIFT =  0.8
    GRIP_FACTOR_GRASS =  8.0

    DRIFT_SPEED_MIN  = 80.0
    DRIFT_STEER_MIN  = 0.35
    DRIFT_SLIP_KEEP  = 0.25

    W, H = 36, 18

    def __init__(self, x, y, angle=0.0):
        self.s          = CarState(x=x, y=y, angle=angle)
        self.trail: List[Tuple[float,float]] = []
        self._drifting  = False
        self._last_thr  = 0.0
        self._last_brk  = 0.0
        self._last_steer = 0.0

    def reset(self, x, y, angle=0.0):
        self.s         = CarState(x=x, y=y, angle=angle)
        self.trail.clear()
        self._drifting = False

    def update(self, dt, throttle, brake, steer, track: Track):
        s  = self.s
        dt = min(dt, 0.05)

        spd = math.hypot(s.vx, s.vy)
        steer_rate = steer * self.STEER_SPD * dt * clamp(spd / 90, 0.12, 1.0)
        s.angle   += steer_rate

        ca, sa = math.cos(s.angle), math.sin(s.angle)
        fwd =  s.vx*ca + s.vy*sa
        lat =  s.vx*sa - s.vy*ca

        fwd += (throttle * self.ENGINE - brake * self.BRAKE) * dt

        slip_angle = abs(math.atan2(lat, max(abs(fwd), 1.0)))

        drift_trigger = (spd > self.DRIFT_SPEED_MIN and
                         abs(steer) > self.DRIFT_STEER_MIN and
                         throttle > 0.1)
        drift_sustain = slip_angle > self.DRIFT_SLIP_KEEP

        if not s.on_track:
            grip_factor = self.GRIP_FACTOR_GRASS
            self._drifting = False
        elif drift_trigger or (self._drifting and drift_sustain):
            grip_factor    = self.GRIP_FACTOR_DRIFT
            self._drifting = True
        else:
            grip_factor    = self.GRIP_FACTOR_GRIP
            self._drifting = False

        lat *= math.exp(-grip_factor * dt)

        drag = self.ROLL_DRAG if s.on_track else self.GRASS_DRAG
        fwd  = clamp(fwd * (drag ** (dt * 60)), -self.MAX_SPEED * 0.4, self.MAX_SPEED)

        ca2, sa2 = math.cos(s.angle), math.sin(s.angle)
        s.vx = ca2*fwd + sa2*lat
        s.vy = sa2*fwd - ca2*lat

        s.x = clamp(s.x + s.vx*dt, 20, WIDTH-20)
        s.y = clamp(s.y + s.vy*dt, 20, HEIGHT-20)

        s.on_track    = track.point_on_track(s.x, s.y)
        s.speed       = math.hypot(s.vx, s.vy)
        s.drift_angle = math.atan2(s.vy, s.vx) - s.angle if spd > 5 else 0.0

        if abs(s.drift_angle) > 0.15:
            self.trail.append((s.x, s.y))
        if len(self.trail) > 600:
            self.trail = self.trail[-600:]

        s.total_time += dt
        s.lap_time   += dt
        seg  = track.nearest_seg_index(s.x, s.y)
        n    = len(track.centerline) - 1
        if seg < n * 0.08 and s.last_seg > n * 0.85:
            if s.lap > 0 and s.lap_time < s.best_lap:
                s.best_lap = s.lap_time
            s.lap     += 1
            s.lap_time = 0.0
        s.last_seg = seg
        self._last_thr  = throttle
        self._last_brk  = brake
        self._last_steer = steer

    def get_corners(self):
        hw, hh = self.W/2, self.H/2
        ca, sa = math.cos(self.s.angle), math.sin(self.s.angle)
        return [
            (self.s.x + px*ca - py*sa, self.s.y + px*sa + py*ca)
            for px, py in [(-hw,-hh),(hw,-hh),(hw,hh),(-hw,hh)]
        ]

    def draw(self, surf):
        s = self.s
        n_trail = len(self.trail)
        for i in range(1, n_trail):
            a = int(200 * i / n_trail)
            tx, ty = self.trail[i]
            ts = pygame.Surface((6,6), pygame.SRCALPHA)
            pygame.draw.circle(ts, (15, 15, 15, a), (3,3), 3)
            surf.blit(ts, (int(tx)-3, int(ty)-3))

        corners = self.get_corners()
        pygame.draw.polygon(surf, (20,20,20), [(x+3,y+3) for x,y in corners])
        body_col = (230, 90, 10) if self._drifting else CAR_RED
        pygame.draw.polygon(surf, body_col, corners)

        ca, sa = math.cos(s.angle), math.sin(s.angle)
        wf = [(s.x+ca*6-sa*7,  s.y+sa*6+ca*7),
              (s.x+ca*11-sa*7, s.y+sa*11+ca*7),
              (s.x+ca*11+sa*7, s.y+sa*11-ca*7),
              (s.x+ca*6+sa*7,  s.y+sa*6-ca*7)]
        pygame.draw.polygon(surf, (180,225,255), wf)
        rs = [(s.x+ca*12-sa*5, s.y+sa*12+ca*5),
              (s.x+ca*14-sa*5, s.y+sa*14+ca*5),
              (s.x+ca*14+sa*5, s.y+sa*14-ca*5),
              (s.x+ca*12+sa*5, s.y+sa*12-ca*5)]
        pygame.draw.polygon(surf, CAR_ACCENT, rs)
        self._draw_control_arrow(surf)

    def _draw_control_arrow(self, surf):
        """Draw a small arrow showing steering direction and thrust/brake."""
        s = self.s
        thr, brk, steer = self._last_thr, self._last_brk, self._last_steer
        # Steering direction: car angle + steer offset (e.g. ±0.6 rad)
        steer_angle = s.angle + steer * 0.6
        ca, sa = math.cos(steer_angle), math.sin(steer_angle)
        # Thrust arrow: forward in steer direction, length ∝ throttle
        base_len = 28
        tip_x = s.x + ca * thr * base_len
        tip_y = s.y + sa * thr * base_len
        if thr > 0.03:
            pygame.draw.line(surf, (100, 255, 120), (s.x, s.y), (tip_x, tip_y), 2)
            # Arrowhead
            ah = 8
            ax = tip_x - ca * ah + sa * 4
            ay = tip_y - sa * ah - ca * 4
            bx = tip_x - ca * ah - sa * 4
            by = tip_y - sa * ah + ca * 4
            pygame.draw.polygon(surf, (100, 255, 120), [(tip_x, tip_y), (ax, ay), (bx, by)])
        # Brake: short arrow backward
        if brk > 0.03:
            bca, bsa = math.cos(s.angle + math.pi), math.sin(s.angle + math.pi)
            brk_len = brk * 18
            bx = s.x + bca * brk_len
            by = s.y + bsa * brk_len
            pygame.draw.line(surf, (255, 90, 90), (s.x, s.y), (bx, by), 2)
            pygame.draw.circle(surf, (255, 90, 90), (int(bx), int(by)), 3)

    def get_observation(self, track: Track):
        s = self.s
        return dict(x=s.x, y=s.y, angle=s.angle, vx=s.vx, vy=s.vy,
                    speed=s.speed, drift_angle=s.drift_angle,
                    on_track=s.on_track,
                    nearest_seg=track.nearest_seg_index(s.x, s.y),
                    lap=s.lap, lap_time=s.lap_time)

# ─────────────────────────────────────────────────────────────────────────────
#  HUD
# ─────────────────────────────────────────────────────────────────────────────

class HUD:
    def __init__(self):
        self.fn_l = pygame.font.SysFont("consolas", 26, bold=True)
        self.fn_m = pygame.font.SysFont("consolas", 19)
        self.fn_s = pygame.font.SysFont("consolas", 14)

    def draw(self, surf, car: Car, track_name: str, recording: bool, rec_frames: int):
        s = car.s

        drift_deg = math.degrees(s.drift_angle)
        drift_col = (255, 80, 30) if abs(drift_deg) > 8 else (255, 210, 50)
        rows = [
            (f"SPEED   {s.speed:5.0f} px/s",  CYAN),
            (f"LAP     {s.lap:3d}",            WHITE),
            (f"TIME    {s.lap_time:6.2f} s",   WHITE),
            (f"BEST    {s.best_lap if s.best_lap < 1e9 else 0.0:6.2f} s", ORANGE),
            (f"DRIFT   {drift_deg:+5.1f}\u00b0", drift_col),
        ]
        pw, ph, pad = 232, len(rows)*26+24, 10
        panel = pygame.Surface((pw, ph), pygame.SRCALPHA)
        panel.fill((0,0,0,190))
        pygame.draw.rect(panel, CYAN, (0,0,4,ph))
        surf.blit(panel, (10, 10))
        for i, (txt, col) in enumerate(rows):
            surf.blit(self.fn_m.render(txt, True, col), (24, 10+pad+i*26))

        tn   = self.fn_l.render(track_name, True, WHITE)
        tx   = WIDTH//2 - tn.get_width()//2
        bg   = pygame.Surface((tn.get_width()+22, tn.get_height()+10), pygame.SRCALPHA)
        bg.fill((0,0,0,165))
        surf.blit(bg, (tx-11, 7))
        surf.blit(tn, (tx, 12))

        bx, by, bw, bh = 10, HEIGHT-90, 222, 12
        ratio = clamp(s.speed/Car.MAX_SPEED, 0, 1)
        pygame.draw.rect(surf, (45,45,45), (bx,by,bw,bh), border_radius=5)
        if ratio > 0:
            bc = (int(lerp(50,255,ratio)), int(lerp(210,50,ratio)), 50)
            pygame.draw.rect(surf, bc, (bx,by,int(bw*ratio),bh), border_radius=5)
        pygame.draw.rect(surf, GREY, (bx,by,bw,bh), 1, border_radius=5)
        surf.blit(self.fn_s.render("SPEED", True, GREY), (bx, by-16))

        hints = ["W/S  Throttle · Brake", "A/D  Steer",
                 "R  Reset", "TAB  Next Track", "ESC  Menu"]
        for i, h in enumerate(hints):
            surf.blit(self.fn_s.render(h, True, (140,140,140)), (10, HEIGHT-68+i*14))

        if not s.on_track:
            w = self.fn_l.render("\u26a0  OFF TRACK", True, (255,60,60))
            surf.blit(w, (WIDTH//2 - w.get_width()//2, 50))
        elif car._drifting:
            d = self.fn_l.render("DRIFTING!", True, (255, 100, 20))
            surf.blit(d, (WIDTH//2 - d.get_width()//2, 50))

        if recording:
            dot = pygame.Surface((12,12), pygame.SRCALPHA)
            pygame.draw.circle(dot, (255,50,50,220), (6,6), 6)
            surf.blit(dot, (WIDTH-162, 14))
            surf.blit(self.fn_s.render(f"REC  {rec_frames} frames", True,(255,80,80)),
                      (WIDTH-147, 11))

# ─────────────────────────────────────────────────────────────────────────────
#  Track Selection Menu
# ─────────────────────────────────────────────────────────────────────────────

class TrackMenu:
    CARD_W, CARD_H = 190, 130
    COLS           = 5

    def __init__(self, tracks: List[Track]):
        self.tracks  = tracks
        self.sel     = 0
        self.fn_t    = pygame.font.SysFont("consolas", 48, bold=True)
        self.fn_m    = pygame.font.SysFont("consolas", 18)
        self.fn_s    = pygame.font.SysFont("consolas", 15)
        self.previews = [self._make_preview(t) for t in tracks]

    def _make_preview(self, track: Track):
        cw, ch = self.CARD_W, self.CARD_H
        surf = pygame.Surface((cw, ch))
        surf.fill((28, 28, 32))
        pts = track.centerline
        if not pts: return surf
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        pad = 12
        sx = (cw-pad*2) / max(max_x-min_x, 1)
        sy = (ch-pad*2) / max(max_y-min_y, 1)
        scale = min(sx, sy)
        cx = (max_x+min_x)/2; cy = (max_y+min_y)/2
        def tp(px, py):
            return (int((px-cx)*scale+cw/2), int((py-cy)*scale+ch/2))
        for seg in track.segments:
            lA, lB, rB, rA = seg
            pygame.draw.polygon(surf, (68,68,68), [tp(*lA), tp(*lB), tp(*rB), tp(*rA)])
        le = [tp(*p) for p in track._left_edge]
        re = [tp(*p) for p in track._right_edge]
        if len(le) >= 2:
            pygame.draw.lines(surf, BORDER_COL, True, le, 2)
            pygame.draw.lines(surf, BORDER_COL, True, re, 2)
        sx2, sy2 = tp(*track.start_pos)
        pygame.draw.circle(surf, (255, 50, 50), (sx2, sy2), 4)
        return surf

    def handle_event(self, event) -> int:
        n    = len(self.tracks)
        COLS = self.COLS
        row  = self.sel // COLS
        col  = self.sel % COLS
        n_rows = math.ceil(n / COLS)
        if event.type == pygame.KEYDOWN:
            if event.key in (pygame.K_LEFT, pygame.K_a):
                if col > 0: self.sel -= 1
            elif event.key in (pygame.K_RIGHT, pygame.K_d):
                new = self.sel + 1
                if new % COLS != 0 and new < n: self.sel = new
            elif event.key in (pygame.K_UP,):
                new = self.sel - COLS
                if new >= 0: self.sel = new
            elif event.key in (pygame.K_DOWN,):
                new = self.sel + COLS
                if new < n: self.sel = new
            elif event.key in (pygame.K_RETURN, pygame.K_SPACE):
                return self.sel
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            cw, ch = self.CARD_W, self.CARD_H
            gap    = 20
            total  = COLS * cw + (COLS-1) * gap
            ox     = WIDTH//2 - total//2
            oy     = 148
            rh     = ch + 42
            for i in range(n):
                r = i // COLS; c = i % COLS
                x = ox + c*(cw+gap); y = oy + r*(rh+gap)
                if pygame.Rect(x-2, y-2, cw+4, rh+4).collidepoint(event.pos):
                    if i == self.sel: return i
                    self.sel = i
                    break
        return -1

    def draw(self, surf):
        surf.fill((16, 16, 22))
        title = self.fn_t.render("DRIFT RACING", True, ORANGE)
        surf.blit(title, (WIDTH//2-title.get_width()//2, 28))
        sub = self.fn_s.render("← → ↑ ↓  navigate   ENTER to race   click card to select, double-click to race", True, GREY)
        surf.blit(sub, (WIDTH//2-sub.get_width()//2, 90))

        n    = len(self.tracks)
        COLS = self.COLS
        cw, ch = self.CARD_W, self.CARD_H
        gap  = 20
        total = COLS * cw + (COLS-1) * gap
        ox   = WIDTH//2 - total//2
        oy   = 148
        rh   = ch + 42   # row height per card incl. name

        for i, (track, prev) in enumerate(zip(self.tracks, self.previews)):
            r   = i // COLS;  c = i % COLS
            x   = ox + c*(cw+gap);  y = oy + r*(rh+gap)
            sel = (i == self.sel)
            border_col = ORANGE if sel else (70,70,80)
            bg_col     = (35,35,50) if sel else (22,22,28)
            bg = pygame.Surface((cw+4, rh+4), pygame.SRCALPHA)
            bg.fill((*bg_col, 230))
            surf.blit(bg, (x-2, y-2))
            pygame.draw.rect(surf, border_col, (x-2, y-2, cw+4, rh+4),
                             2 if sel else 1, border_radius=6)
            surf.blit(prev, (x, y))
            col = WHITE if sel else GREY
            nm  = self.fn_m.render(track.name, True, col)
            surf.blit(nm, (x+cw//2-nm.get_width()//2, y+ch+8))

        hint = self.fn_s.render("ENTER / SPACE to race   F11 fullscreen", True, (110,110,110))
        surf.blit(hint, (WIDTH//2-hint.get_width()//2, HEIGHT-36))

# ─────────────────────────────────────────────────────────────────────────────
#  Game
# ─────────────────────────────────────────────────────────────────────────────

class Game:
    MENU    = "menu"
    DRIVING = "driving"

    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.SCALED)
        pygame.display.set_caption("Drift Racing Sim")
        self.clock  = pygame.time.Clock()
        self.fullscreen = False

        self.all_tracks = [Track(d) for d in TRACKS]
        self.menu       = TrackMenu(self.all_tracks)
        self.track_idx  = 0
        self.track      = self.all_tracks[0]
        self.car        = Car(*self.track.start_pos, angle=self.track.start_angle)
        self.hud        = HUD()
        self.state      = self.MENU
        self.recording  = False
        self.rec_buf    = []

    def _load_track(self, idx):
        self.track_idx = idx % len(self.all_tracks)
        self.track     = self.all_tracks[self.track_idx]
        self.car.reset(*self.track.start_pos, angle=self.track.start_angle)
        self.rec_buf.clear()

    def run(self):
        while True:
            dt = self.clock.tick(FPS) / 1000.0

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit(); sys.exit()

                if self.state == self.MENU:
                    chosen = self.menu.handle_event(event)
                    if chosen >= 0:
                        self._load_track(chosen)
                        self.state = self.DRIVING

                elif self.state == self.DRIVING:
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            self.state = self.MENU
                        elif event.key == pygame.K_F11:
                            self.fullscreen = not self.fullscreen
                            pygame.display.toggle_fullscreen()
                        elif event.key == pygame.K_r:
                            self.car.reset(*self.track.start_pos,
                                           angle=self.track.start_angle)
                        elif event.key == pygame.K_TAB:
                            self._load_track(self.track_idx + 1)
                        elif event.key == pygame.K_F1:
                            self.recording = not self.recording
                            if not self.recording and self.rec_buf:
                                print(f"[REC] {len(self.rec_buf)} frames captured")

            if self.state == self.MENU:
                self.menu.draw(self.screen)

            elif self.state == self.DRIVING:
                keys = pygame.key.get_pressed()
                thr  = 1.0 if keys[pygame.K_w] else 0.0
                brk  = 1.0 if keys[pygame.K_s] else 0.0
                steer= (1.0 if keys[pygame.K_d] else 0.0) \
                      -(1.0 if keys[pygame.K_a] else 0.0)

                self.car.update(dt, thr, brk, steer, self.track)

                if self.recording:
                    self.rec_buf.append({
                        "obs":    self.car.get_observation(self.track),
                        "action": dict(throttle=thr, brake=brk, steer=steer),
                    })

                self.screen.blit(self.track.surface, (0, 0))
                self.car.draw(self.screen)
                self.hud.draw(self.screen, self.car,
                              self.track.name, self.recording, len(self.rec_buf))

            pygame.display.flip()