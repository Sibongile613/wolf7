"""Lone Wolf Labs bot v2 for Builders - Mech Royale (mechfight/1).

Changes vs v1 (scored 1.38):
- Lead time uses the REAL projectile speed from self.weapons[i].stats.speed
  (v1 hard-coded 220 m/s; the gun is 252 m/s by default -> v1 over-led ~13%).
- Aims at centre of mass (z + 4.5), scaled up 5% per cube the TARGET holds
  (v1 aimed at z + 3.8: shins).
- Own muzzle height read from self.stats.muzzle_height (v1 hard-coded 7.0;
  it grows 5% per cube you collect).
- Fire-gate: the gun only fires when the torso's ACTUAL aim is aligned with
  the desired yaw (shots leave along actual aim, not commanded aim).
- Strafe direction randomised from hello.seed (v1 flipped on a public
  tick//30 pattern expert bots can exploit).
- Target selection: prefers low-HP enemies (they die faster -> placement
  points + cubes) with a distance tiebreak, instead of always nearest.
"""
from __future__ import annotations

import array
import json
import math
import os
import random
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import mechfight as mf


def _wrap_angle(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def _dist2d(a: List[float], b: List[float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


class Terrain:
    """Ground height from the map's terrain.i16, so a weapon lying on a roof can be told from one on the ground."""

    def __init__(self, map_dir: str) -> None:
        self.ok = False
        try:
            base = os.path.join(map_dir, "v2")
            with open(os.path.join(base, "world.json"), "r", encoding="utf-8") as f:
                meta = json.load(f)["terrain"]
            self.min_x = float(meta["minX"])
            self.min_y = float(meta["minY"])
            self.step = float(meta["step"])
            self.nx = int(meta["nx"])
            self.ny = int(meta["ny"])
            data = array.array("h")
            with open(os.path.join(base, str(meta.get("file", "terrain.i16"))), "rb") as f:
                data.fromfile(f, self.nx * self.ny)
            if sys.byteorder == "big":
                data.byteswap()
            self.data = data
            self.ok = True
        except Exception:
            self.ok = False

    def ground(self, x: float, y: float) -> Optional[float]:
        if not self.ok:
            return None
        i = max(0, min(self.nx - 1, int(round((x - self.min_x) / self.step))))
        j = max(0, min(self.ny - 1, int(round((y - self.min_y) / self.step))))
        return self.data[j * self.nx + i] / 100.0


class Geo:
    """Building footprints from v2/world.json, with a 3D line-of-sight test (ray height vs building top)."""

    def __init__(self, map_dir: str) -> None:
        self.ok = False
        self.prisms: List[Tuple[float, float, float, float, float, float, List[float]]] = []
        try:
            with open(os.path.join(map_dir, "v2", "world.json"), "r", encoding="utf-8") as f:
                data = json.load(f)
            for p in data.get("prisms", []):
                base = float(p.get("base", 0.0))
                top = float(p.get("top", 10.0))
                if top - base < 1.5:
                    continue
                ring = p.get("ring", [])
                if len(ring) < 6:
                    continue
                xs, ys = ring[0::2], ring[1::2]
                self.prisms.append((min(xs), min(ys), max(xs), max(ys), base, top, ring))
            self.ok = len(self.prisms) > 0
        except Exception:
            self.ok = False

    @staticmethod
    def _inside(px: float, py: float, ring: List[float]) -> bool:
        n = len(ring) // 2
        ins = False
        j = n - 1
        for i in range(n):
            xi, yi = ring[2 * i], ring[2 * i + 1]
            xj, yj = ring[2 * j], ring[2 * j + 1]
            if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / ((yj - yi) or 1e-12) + xi):
                ins = not ins
            j = i
        return ins

    def in_footprint(self, x: float, y: float, pad: float = 3.0) -> bool:
        """True if (x, y) is inside (or within pad of) any solid footprint."""
        for (x0, y0, x1, y1, _b, _t, ring) in self.prisms:
            if x < x0 - pad or x > x1 + pad or y < y0 - pad or y > y1 + pad:
                continue
            if (self._inside(x, y, ring) or self._inside(x + pad, y, ring) or self._inside(x - pad, y, ring)
                    or self._inside(x, y + pad, ring) or self._inside(x, y - pad, ring)):
                return True
        return False

    def blocked(self, ax: float, ay: float, az: float, bx: float, by: float, bz: float) -> bool:
        """True if a straight line from A to B passes through a building (taking its height into account)."""
        lo_x, hi_x = min(ax, bx), max(ax, bx)
        lo_y, hi_y = min(ay, by), max(ay, by)
        dx, dy = bx - ax, by - ay
        for (x0, y0, x1, y1, base, top, ring) in self.prisms:
            if x1 < lo_x or x0 > hi_x or y1 < lo_y or y0 > hi_y:
                continue
            ts = [0.0, 1.0]
            n = len(ring) // 2
            for i in range(n):
                j = i - 1
                cx, cy, ex, ey = ring[2 * j], ring[2 * j + 1], ring[2 * i], ring[2 * i + 1]
                fx, fy = ex - cx, ey - cy
                d = dx * fy - dy * fx
                if abs(d) < 1e-12:
                    continue
                tt = ((cx - ax) * fy - (cy - ay) * fx) / d
                uu = ((cx - ax) * dy - (cy - ay) * dx) / d
                if 0.0 <= tt <= 1.0 and 0.0 <= uu <= 1.0:
                    ts.append(tt)
            ts.sort()
            for k in range(len(ts) - 1):
                t0, t1 = ts[k], ts[k + 1]
                if t1 - t0 < 1e-9:
                    continue
                tm = 0.5 * (t0 + t1)
                if not self._inside(ax + dx * tm, ay + dy * tm, ring):
                    continue
                za = az + (bz - az) * t0
                zb = az + (bz - az) * t1
                if max(za, zb) >= base and min(za, zb) <= top:
                    return True
        return False


class Bot:
    name = "lone-wolf-v6"

    def __init__(self, hello: mf.Hello) -> None:
        cfg = hello.get("config", {})
        loot_cfg = cfg.get("loot", {})
        self.pickup_radius: float = float(loot_cfg.get("pickup_radius", 4.0))
        self.upgrades_cfg: List[Dict[str, Any]] = cfg.get("upgrades", [])
        self.upgrade_priority: Dict[str, int] = {}
        for u in self.upgrades_cfg:
            uid = str(u.get("id", ""))
            effects = u.get("effects", [])
            score = 0
            for ef in effects:
                path = str(ef.get("path", ""))
                if "armor" in uid or "rear" in path or "max_hp" in path or "regen" in path:
                    score += 30
                if "sensor" in uid or "vision" in path or "sound" in path:
                    score += 22
                if "speed" in path or "boost" in path or "energy" in path:
                    score += 18
                if "damage" in path or "rate" in path or "cooldown" in path:
                    score += 12
            self.upgrade_priority[uid] = score
        # CHANGED: seeded RNG from hello.seed so behaviour is reproducible
        # (protocol suggests exactly this) and NOT predictable to opponents.
        self.rng = random.Random(int(hello.get("seed", 1234567)))
        self.last_damage_t: float = -999.0
        self.last_damage_bearing: Optional[float] = None
        self.laser_charging: List[bool] = [False, False]
        self.terrain = Terrain(str(hello.get("map", {}).get("dir", "/arena")))
        self.skip_pickups: Dict[int, float] = {}
        self.loot_mem: Optional[Dict[str, Any]] = None
        self.geo = Geo(str(hello.get("map", {}).get("dir", "/arena")))
        self.last_threat_t: float = -999.0
        self.last_threat_bearing: Optional[float] = None
        self.last_threat_by: Optional[int] = None
        self.cover_goal: Optional[Tuple[float, float]] = None
        self.cover_start: float = -999.0
        self.cover_until: float = -999.0
        self.cover_cd: float = -999.0
        self.hide_goal: Optional[Tuple[float, float]] = None
        self.hide_cd: float = -999.0
        self.hideouts: List[Tuple[float, float, float, int, float]] = self._build_hideouts()
        self.last_pos: Optional[Tuple[float, float]] = None
        self.stuck_ticks: int = 0

    def _pick_upgrade(self, offer: Dict[str, Any]) -> int:
        choices = offer.get("choices", [])
        if not choices:
            return 0
        best_idx = 0
        best_score = -10**9
        for idx, ch in enumerate(choices):
            uid = str(ch.get("id", ""))
            desc = str(ch.get("desc", "")).lower()
            s = self.upgrade_priority.get(uid, 0)
            if "armor" in desc or "hp" in desc or "regen" in desc:
                s += 25
            if "vision" in desc or "sensor" in desc or "sound" in desc:
                s += 15
            if "speed" in desc or "energy" in desc:
                s += 12
            if s > best_score:
                best_score = s
                best_idx = idx
        return best_idx

    def _choose_zone_target(self, obs: mf.Observation, x: float, y: float) -> Tuple[float, float]:
        zone = obs["zone"]
        t = float(obs["t"])
        shrink_start = float(zone.get("shrink_start", t))
        shrink_end = float(zone.get("shrink_end", t + 1.0))
        cur_c = zone.get("center", [0.0, 0.0])
        cur_r = float(zone.get("radius", 500.0))
        nxt_c = zone.get("next_center", cur_c)
        nxt_r = float(zone.get("next_radius", cur_r))

        if t >= shrink_start:
            if shrink_end > shrink_start:
                frac = max(0.0, min(1.0, (t - shrink_start) / (shrink_end - shrink_start)))
            else:
                frac = 1.0
            cx = cur_c[0] + (nxt_c[0] - cur_c[0]) * frac
            cy = cur_c[1] + (nxt_c[1] - cur_c[1]) * frac
            rad = cur_r + (nxt_r - cur_r) * frac
        else:
            cx, cy, rad = cur_c[0], cur_c[1], cur_r

        safe_r = max(5.0, rad * 0.72)
        dx_c = nxt_c[0] - cx
        dy_c = nxt_c[1] - cy
        dist_c = math.hypot(dx_c, dy_c)
        if dist_c > 1e-6:
            tx = cx + (dx_c / dist_c) * min(dist_c * 0.65, safe_r * 0.5)
            ty = cy + (dy_c / dist_c) * min(dist_c * 0.65, safe_r * 0.5)
        else:
            tx, ty = cx, cy

        d_me_cur = math.hypot(x - cx, y - cy)
        if d_me_cur < safe_r and math.hypot(x - nxt_c[0], y - nxt_c[1]) > nxt_r * 0.8:
            return float(nxt_c[0]), float(nxt_c[1])
        return float(tx), float(ty)

    def _weapon_fire_decision(
        self,
        slot_idx: int,
        w: Optional[mf.WeaponObs],
        dist: float,
        has_target: bool,
        energy: float,
        target_has_mortar: bool,
    ) -> bool:
        if w is None or not has_target:
            self.laser_charging[slot_idx] = False
            return False

        kind = w.get("kind")
        stats = w.get("stats", {})
        max_range = float(stats.get("range", 200.0))

        if kind == "gun":
            ammo = float(w.get("ammo", 0.0))
            reload_left = float(w.get("reload", 0.0))
            if reload_left > 0.0 or ammo <= 0.0:
                return False
            return dist <= max_range * 0.95

        if kind == "lightning":
            if energy < 12.0 or dist > min(45.0, max_range):
                return False
            return float(w.get("cooldown", 0.0)) <= 0.05

        if kind == "mortar":
            if dist < 24.0 or dist > max_range:
                return False
            return float(w.get("cooldown", 0.0)) <= 0.05

        if kind == "laser":
            charge_time = float(w.get("charge", 0.0))
            full_charge = float(stats.get("charge", 0.75))
            if dist > max_range:
                self.laser_charging[slot_idx] = False
                return False
            target_charge = full_charge * (0.85 if target_has_mortar else 0.98)
            if not self.laser_charging[slot_idx]:
                if float(w.get("cooldown", 0.0)) <= 0.05 and energy > 15.0:
                    self.laser_charging[slot_idx] = True
                    return True
                return False
            if charge_time >= target_charge:
                self.laser_charging[slot_idx] = False
                return False
            return True

        return False

    def _build_hideouts(self) -> List[Tuple[float, float, float, int, float]]:
        """Spots just outside tall buildings, scored by how many of 8 directions they are hidden from (computed once)."""
        out: List[Tuple[float, float, float, int, float]] = []
        if not self.geo.ok:
            return out
        t_end = time.perf_counter() + 6.0
        seen = set()
        dirs = [(math.cos(k * math.pi / 4.0), math.sin(k * math.pi / 4.0)) for k in range(8)]
        for (_x0, _y0, _x1, _y1, base, top, ring) in self.geo.prisms:
            if time.perf_counter() > t_end:
                break
            if top - base < 12.0:
                continue
            n = len(ring) // 2
            cx = sum(ring[0::2]) / n
            cy = sum(ring[1::2]) / n
            step = max(1, n // 10)
            for i in range(0, n, step):
                vx, vy = ring[2 * i], ring[2 * i + 1]
                dx, dy = vx - cx, vy - cy
                d = math.hypot(dx, dy) or 1.0
                px, py = vx + dx / d * 5.5, vy + dy / d * 5.5
                key = (int(px // 8.0), int(py // 8.0))
                if key in seen:
                    continue
                seen.add(key)
                g0 = self.terrain.ground(px, py)
                g = g0 if g0 is not None else base
                if self.geo.in_footprint(px, py, 3.0):
                    continue
                enc = 0
                for ux, uy in dirs:
                    if self.geo.blocked(px, py, g + 7.0, px + ux * 25.0, py + uy * 25.0, g + 4.5):
                        enc += 1
                if enc >= 3:
                    out.append((px, py, g, enc, min(top - base, 40.0)))
        return out

    def _pick_hideout(self, x: float, y: float, z: float, zcx: float, zcy: float, zr: float) -> Optional[Tuple[float, float]]:
        cands = []
        for (hx, hy, hg, enc, tall) in self.hideouts:
            d_c = math.hypot(hx - zcx, hy - zcy)
            if d_c > 0.85 * zr or d_c < 0.25 * zr:       # inside the next circle, but not at the hot centre
                continue
            d = math.hypot(hx - x, hy - y)
            if d > 260.0:
                continue
            cands.append((enc * 8.0 + tall * 0.2 - d * 0.12, hx, hy, hg))
        cands.sort(reverse=True)
        for _s, hx, hy, hg in cands[:12]:
            if not self.geo.blocked(x, y, z + 1.0, hx, hy, hg + 1.0):   # straight walkable path
                return (hx, hy)
        return None

    def _zone_circle(self, obs: mf.Observation) -> Tuple[float, float, float]:
        zone = obs["zone"]
        t = float(obs["t"])
        shrink_start = float(zone.get("shrink_start", t))
        shrink_end = float(zone.get("shrink_end", t + 1.0))
        cur_c = zone.get("center", [0.0, 0.0])
        cur_r = float(zone.get("radius", 500.0))
        nxt_c = zone.get("next_center", cur_c)
        nxt_r = float(zone.get("next_radius", cur_r))
        if t >= shrink_start:
            frac = max(0.0, min(1.0, (t - shrink_start) / max(1e-6, shrink_end - shrink_start)))
            return (cur_c[0] + (nxt_c[0] - cur_c[0]) * frac, cur_c[1] + (nxt_c[1] - cur_c[1]) * frac,
                    cur_r + (nxt_r - cur_r) * frac)
        return float(cur_c[0]), float(cur_c[1]), cur_r

    def _find_cover(self, x: float, y: float, z: float, threat: Tuple[float, float, float],
                    zc: Tuple[float, float, float]) -> Optional[Tuple[float, float]]:
        """A nearby open spot a tall building fully hides (feet to head) from the threat, reachable in a straight line."""
        tx, ty, tz = threat
        to_threat = math.atan2(ty - y, tx - x)
        best: Optional[Tuple[float, float]] = None
        best_s = -1e9
        for r in (10.0, 16.0, 24.0, 34.0, 46.0):
            for k in range(16):
                ang = 2.0 * math.pi * k / 16.0
                cx = x + math.cos(ang) * r
                cy = y + math.sin(ang) * r
                if math.hypot(cx - zc[0], cy - zc[1]) > zc[2] - 12.0:
                    continue
                g = self.terrain.ground(cx, cy)
                gz = g if g is not None else z
                if abs(gz - z) > 6.0:                       # avoid slopes/ledges we can't walk
                    continue
                if self.geo.in_footprint(cx, cy, 3.0):
                    continue
                # hidden from the threat's muzzle at both chest and head height
                if not (self.geo.blocked(tx, ty, tz + 7.0, cx, cy, gz + 4.5)
                        and self.geo.blocked(tx, ty, tz + 7.0, cx, cy, gz + 8.5)):
                    continue
                # straight walkable path (no wall in the way)
                if self.geo.blocked(x, y, z + 1.0, cx, cy, gz + 1.0):
                    continue
                toward = math.cos(ang - to_threat)          # 1 = toward the threat
                s = -0.45 * r - 14.0 * max(0.0, toward)
                if s > best_s:
                    best_s, best = s, (cx, cy)
        return best

    def _cover_step(self, obs: mf.Observation, x: float, y: float, z: float, t: float, in_zone: bool,
                    hostile_mechs: List[Dict[str, Any]], nearest_d: float) -> Tuple[Optional[Tuple[float, float]], bool, bool]:
        """Returns (move_vector or None, hold_still, dash)."""
        if not self.geo.ok:
            return None, False, False
        under_fire = (t - self.last_threat_t) < 2.5
        aimed: Optional[Dict[str, Any]] = None
        for m in hostile_mechs:
            if m.get("charging"):
                ddx, ddy = x - float(m["pos"][0]), y - float(m["pos"][1])
                if math.hypot(ddx, ddy) > 25.0 and abs(_wrap_angle(float(m["aim"][0]) - math.atan2(ddy, ddx))) < 0.12:
                    aimed = m
                    break
        if self.cover_goal is not None:
            # leave cover when it has done its job, the zone needs us, or a fight is on top of us
            d_goal = math.hypot(self.cover_goal[0] - x, self.cover_goal[1] - y)
            stuck = self.stuck_ticks >= 10 and d_goal > 2.5      # only "stuck" if we are trying to move
            if t > self.cover_until or not in_zone or nearest_d < 45.0 or stuck:
                if stuck or nearest_d < 45.0:
                    self.cover_cd = t + 3.0
                self.cover_goal = None
        if self.cover_goal is None and (under_fire or aimed is not None) and in_zone and nearest_d >= 45.0 and t >= self.cover_cd:
            # where is the threat?
            tp: Optional[Tuple[float, float, float]] = None
            if aimed is not None:
                tp = (float(aimed["pos"][0]), float(aimed["pos"][1]), float(aimed["pos"][2]))
            elif self.last_threat_by is not None:
                for m in hostile_mechs:
                    if m.get("id") == self.last_threat_by:
                        tp = (float(m["pos"][0]), float(m["pos"][1]), float(m["pos"][2]))
                        break
            if tp is None and self.last_threat_bearing is not None:
                tp = (x + math.cos(self.last_threat_bearing) * 120.0, y + math.sin(self.last_threat_bearing) * 120.0, z)
            if tp is not None:
                spot = self._find_cover(x, y, z, tp, self._zone_circle(obs))
                if spot is None:
                    self.cover_cd = t + 2.0
                else:
                    self.cover_goal = spot
                    self.cover_start = t
                    self.cover_until = t + 3.0
        if self.cover_goal is None:
            return None, False, False
        # stay hidden until we have gone ~5.5 s without being shot (health regeneration starts after 5 s), capped
        self.cover_until = min(self.cover_start + 14.0, max(self.cover_until, self.last_threat_t + 5.5))
        gx, gy = self.cover_goal
        d = math.hypot(gx - x, gy - y)
        if d > 2.5:
            return (gx - x, gy - y), False, (d > 14.0 and under_fire)
        return None, True, False

    def act(self, obs: mf.Observation) -> mf.Action:
        me = obs["self"]
        x, y, z = float(me["pos"][0]), float(me["pos"][1]), float(me["pos"][2])
        hp = float(me.get("hp", 1400.0))
        max_hp = max(1.0, float(me.get("max_hp", 1400.0)))
        hp_ratio = hp / max_hp
        energy = float(me.get("energy", 0.0))
        in_zone = bool(me.get("in_zone", True))
        t = float(obs["t"])
        my_stats = me.get("stats", {})
        # CHANGED: real muzzle height (grows with cubes); default 7.0 at spawn.
        muzzle_z = z + float(my_stats.get("muzzle_height", 7.0))

        for ev in obs.get("events", []):
            if ev.get("type") == "damaged":
                self.last_damage_t = t
                if ev.get("bearing") is not None:
                    self.last_damage_bearing = float(ev["bearing"])
                if ev.get("weapon") != "zone" and ev.get("bearing") is not None:
                    self.last_threat_t = t
                    self.last_threat_bearing = float(ev["bearing"])
                    self.last_threat_by = ev.get("by")
            elif ev.get("type") == "take_failed":
                self.skip_pickups[int(ev.get("id", -1))] = t + 30.0

        if self.last_pos is not None and math.hypot(x - self.last_pos[0], y - self.last_pos[1]) < 0.12:
            self.stuck_ticks += 1
        else:
            self.stuck_ticks = 0
        self.last_pos = (x, y)

        act: mf.Action = {"tick": obs["tick"]}

        if obs.get("offer"):
            act["pick"] = self._pick_upgrade(obs["offer"])  # type: ignore[arg-type]

        weapons = me.get("weapons", [None, None])
        gun_slot = weapons[0] if len(weapons) > 0 and weapons[0] and weapons[0].get("kind") == "gun" else None
        # CHANGED: real bullet speed for lead (upgrades change it mid-match).
        bullet_speed = 252.0
        if gun_slot is not None:
            bullet_speed = max(1.0, float(gun_slot.get("stats", {}).get("speed", 252.0)))
            ammo = float(gun_slot.get("ammo", 0.0))
            mag = float(gun_slot.get("stats", {}).get("mag", 12.0))
            reload_left = float(gun_slot.get("reload", 0.0))
            if reload_left <= 0.0 and ammo <= max(2.0, mag * 0.25):
                act["reload"] = True

        pickups = obs.get("pickups", [])
        if len(weapons) > 1 and weapons[1] is None:
            for p in pickups:
                pkind = p.get("kind")
                if pkind in ("laser", "lightning", "mortar") and _dist2d(p["pos"], [x, y]) <= self.pickup_radius:
                    act["take"] = p["id"]
                    break

        mechs = obs.get("mechs", [])
        hostile_mechs = [m for m in mechs if m.get("kind") == "mech"]
        # CHANGED: target selection scores by threat-free weakness:
        # weakest-first gives kills + cubes; distance breaks ties. House bots
        # hunt the weakest too, so killing the weak one first also removes the
        # squad's own preferred target.
        if hostile_mechs:
            def tgt_score(m: Dict[str, Any]) -> float:
                d = math.dist(m["pos"], [x, y, z])
                hp_r = float(m.get("hp", 1400.0)) / max(1.0, float(m.get("max_hp", 1400.0)))
                return d + hp_r * 90.0
            hostile_mechs.sort(key=tgt_score)
        else:
            hostile_mechs.sort(key=lambda m: math.dist(m["pos"], [x, y, z]))

        recently_hit = (t - self.last_damage_t) < 4.5
        low_hp = hp_ratio < 0.45

        # --- LOOT: get a second weapon early. Built-in bots hold a laser within ~5 s of spawn. ---
        loot_target: Optional[Dict[str, Any]] = None
        loot_dist = 0.0
        nearest_d = min((math.hypot(m["pos"][0] - x, m["pos"][1] - y) for m in hostile_mechs), default=999.0)
        enemy_close = nearest_d < 90.0
        if len(weapons) > 1 and weapons[1] is None and in_zone and not enemy_close:
            held = {w.get("kind") for w in weapons if w}
            best_s = -1e9
            for p in pickups:
                pk = p.get("kind")
                if pk not in ("laser", "lightning", "mortar") or pk in held:
                    continue
                if self.skip_pickups.get(int(p["id"]), -1.0) > t:
                    continue
                g = self.terrain.ground(float(p["pos"][0]), float(p["pos"][1]))
                if g is not None and float(p["pos"][2]) - g > 2.0:   # on a roof: not reachable on foot
                    continue
                d = _dist2d(p["pos"], [x, y])
                if d > 200.0:
                    continue
                s = {"laser": 40.0, "lightning": 25.0, "mortar": 10.0}[pk] - 0.3 * d
                if s > best_s:
                    best_s, loot_target, loot_dist = s, p, d
        if loot_target is not None:
            self.loot_mem = {"id": int(loot_target["id"]), "kind": loot_target["kind"],
                             "pos": [float(loot_target["pos"][0]), float(loot_target["pos"][1]), float(loot_target["pos"][2])],
                             "seen": t}
        elif self.loot_mem is not None:
            m = self.loot_mem
            d_mem = _dist2d(m["pos"], [x, y])
            gone = (
                len(weapons) <= 1 or weapons[1] is not None or not in_zone or enemy_close
                or t - m["seen"] > 6.0 or d_mem < 40.0 or self.skip_pickups.get(m["id"], -1.0) > t
            )
            if gone:
                self.loot_mem = None
            else:
                loot_target, loot_dist = m, d_mem   # out of sight for a moment: keep heading to where it was
        if loot_target is not None and self.stuck_ticks >= 12:
            self.skip_pickups[int(loot_target["id"])] = t + 15.0
            self.loot_mem = None
            loot_target = None

        cover_move: Optional[Tuple[float, float]] = None
        cover_hold = False
        cover_dash = False
        try:
            cover_move, cover_hold, cover_dash = self._cover_step(obs, x, y, z, t, in_zone, hostile_mechs, nearest_d)
        except Exception:
            self.cover_goal = None

        # --- GHOST: with a weapon in hand (or nothing worth fetching), stop hunting. Walk to a hideout and sit still. ---
        ghost_ok = False
        ghost_move: Optional[Tuple[float, float]] = None
        ghost_hold = False
        try:
            zn = obs["zone"]
            gnc = zn.get("next_center", zn.get("center", [0.0, 0.0]))
            gnr = float(zn.get("next_radius", zn.get("radius", 500.0)))
            ghost_ok = (self.geo.ok and len(self.hideouts) > 0 and in_zone and loot_target is None
                        and cover_move is None and not cover_hold and nearest_d >= 60.0
                        and (t - self.last_threat_t) > 4.0)
            if ghost_ok:
                if self.hide_goal is not None:
                    d_g0 = math.hypot(self.hide_goal[0] - x, self.hide_goal[1] - y)
                    zone_moved = math.hypot(self.hide_goal[0] - gnc[0], self.hide_goal[1] - gnc[1]) > 0.92 * gnr
                    stuck_g = self.stuck_ticks >= 12 and d_g0 > 2.5       # stuck only counts while we are trying to walk
                    if zone_moved or stuck_g:
                        if stuck_g:
                            self.hide_cd = t + 3.0
                        self.hide_goal = None
                if self.hide_goal is None and t >= self.hide_cd:
                    spot = self._pick_hideout(x, y, z, float(gnc[0]), float(gnc[1]), gnr)
                    if spot is None:
                        self.hide_cd = t + 2.0
                    else:
                        self.hide_goal = spot
                if self.hide_goal is not None:
                    d_g = math.hypot(self.hide_goal[0] - x, self.hide_goal[1] - y)
                    if d_g > 2.5:
                        ghost_move = (self.hide_goal[0] - x, self.hide_goal[1] - y)
                    else:
                        ghost_hold = True
                else:
                    ghost_ok = False          # nowhere to hide: normal behaviour
        except Exception:
            self.hide_goal = None
            ghost_ok, ghost_move, ghost_hold = False, None, False

        zx, zy = self._choose_zone_target(obs, x, y)
        move_x = zx - x
        move_y = zy - y

        nearest_enemy = hostile_mechs[0] if hostile_mechs else None
        desired_yaw: Optional[float] = None
        if nearest_enemy is not None and not low_hp:
            ex, ey, _ = nearest_enemy["pos"]
            edist = math.hypot(ex - x, ey - y)
            to_enemy_yaw = math.atan2(ey - y, ex - x)
            # CHANGED: randomised strafe side + random flip interval instead
            # of the public tick//30 pattern.
            if not hasattr(self, "_strafe_dir") or self.rng.random() < 0.02:
                self._strafe_dir = self.rng.choice([-1.0, 1.0])
                self._strafe_until = t + self.rng.uniform(1.5, 3.5)
            if t > getattr(self, "_strafe_until", 0.0):
                self._strafe_dir *= -1.0
                self._strafe_until = t + self.rng.uniform(1.5, 3.5)
            strafe_yaw = to_enemy_yaw + self._strafe_dir * (math.pi * 0.5)
            desired_range = 75.0 if hp_ratio > 0.6 else 115.0
            radial_pull = 0.0
            if edist < desired_range * 0.75:
                radial_pull = -0.65
            elif edist > desired_range * 1.35 and in_zone:
                radial_pull = 0.35

            blend_x = (zx - x) * (1.2 if not in_zone else 0.45) + math.cos(to_enemy_yaw) * radial_pull * 40.0 + math.cos(strafe_yaw) * 28.0
            blend_y = (zy - y) * (1.2 if not in_zone else 0.45) + math.sin(to_enemy_yaw) * radial_pull * 40.0 + math.sin(strafe_yaw) * 28.0
            move_x, move_y = blend_x, blend_y

        if loot_target is not None:
            move_x = float(loot_target["pos"][0]) - x
            move_y = float(loot_target["pos"][1]) - y

        if cover_move is not None:
            move_x, move_y = cover_move
        if ghost_move is not None:
            move_x, move_y = ghost_move

        move_norm = math.hypot(move_x, move_y)
        if move_norm > 1e-6:
            speed_scale = 0.38 if (low_hp and not recently_hit and in_zone and len(hostile_mechs) == 0) else 1.0
            act["move"] = [(move_x / move_norm) * speed_scale, (move_y / move_norm) * speed_scale]
        else:
            act["move"] = [0.0, 0.0]
        if cover_hold or ghost_hold:
            act["move"] = [0.0, 0.0]   # standing still is silent

        should_boost = (
            (not in_zone and energy > 20.0)
            or (recently_hit and low_hp and energy > 35.0)
            or (loot_target is not None and loot_dist > 30.0 and energy > 40.0)
            or (cover_dash and energy > 35.0)
        )
        act["boost"] = bool(should_boost)
        act["jump"] = bool(me.get("grounded", False) and (self.rng.random() < 0.02) and len(hostile_mechs) > 0)

        if nearest_enemy is not None and not ghost_ok:
            tpos = nearest_enemy["pos"]
            tvel = nearest_enemy.get("vel", [0.0, 0.0, 0.0])
            horiz_dist = max(1.0, math.hypot(tpos[0] - x, tpos[1] - y))
            # CHANGED: lead from the real projectile speed, not 220.
            lead_t = min(0.9, horiz_dist / bullet_speed)
            px = tpos[0] + tvel[0] * lead_t
            py = tpos[1] + tvel[1] * lead_t
            # CHANGED: centre of mass, scaled up by the target's cubes.
            t_cubes = int(nearest_enemy.get("cubes", 0))
            pz = tpos[2] + 4.5 * (1.0 + 0.05 * min(t_cubes, 10))
            yaw = math.atan2(py - y, px - x)
            pitch = math.atan2(pz - muzzle_z, horiz_dist)
            pitch = max(-0.8, min(0.8, pitch))
            act["aim"] = [yaw, pitch]
            desired_yaw = yaw

            target_weapons = nearest_enemy.get("weapons", [])
            target_has_mortar = "mortar" in target_weapons
            f0 = self._weapon_fire_decision(0, weapons[0] if len(weapons) > 0 else None, horiz_dist, True, energy, target_has_mortar)
            f1 = self._weapon_fire_decision(1, weapons[1] if len(weapons) > 1 else None, horiz_dist, True, energy, target_has_mortar)
            fire = [f0, f1]
            # CHANGED: fire-gate. Shots leave along self.aim (actual torso
            # aim), which slews at turn_rate. Hold trigger only when aligned;
            # exception: keep holding a charging laser (aim slews fine while
            # charging, charge_turn_mult is 1).
            my_aim = me.get("aim", [yaw, pitch])
            aim_err = abs(_wrap_angle(float(my_aim[0]) - yaw))
            for i, w in enumerate(weapons[:2]):
                if w is not None and w.get("kind") == "laser" and self.laser_charging[i]:
                    continue
                if aim_err > 0.05:
                    fire[i] = False
            if self.geo.ok:
                try:
                    if (self.geo.blocked(x, y, muzzle_z, float(tpos[0]), float(tpos[1]), float(tpos[2]) + 4.5)
                            and self.geo.blocked(x, y, muzzle_z, float(tpos[0]), float(tpos[1]), float(tpos[2]) + 8.5)):
                        for i, w in enumerate(weapons[:2]):
                            if w is not None and w.get("kind") != "mortar":   # shells arc over walls
                                fire[i] = False
                                if w.get("kind") == "laser":
                                    self.laser_charging[i] = False
                except Exception:
                    pass
            act["fire"] = fire
            mf.draw.line([x, y, muzzle_z], [px, py, pz], "#ff5040")
        else:
            self.laser_charging = [False, False]
            if recently_hit and self.last_damage_bearing is not None:
                act["aim"] = [_wrap_angle(self.last_damage_bearing), 0.0]
            elif self.cover_goal is not None and self.last_threat_bearing is not None:
                act["aim"] = [_wrap_angle(self.last_threat_bearing), 0.0]
            elif loot_target is not None:
                act["aim"] = [math.atan2(float(loot_target["pos"][1]) - y, float(loot_target["pos"][0]) - x), 0.0]
            elif ghost_ok and ghost_hold:
                act["aim"] = [_wrap_angle(t * 0.55), 0.0]                 # slow sweep: watch every direction
            elif ghost_ok and ghost_move is not None:
                act["aim"] = [math.atan2(ghost_move[1], ghost_move[0]), 0.0]
            else:
                act["aim"] = [math.atan2(zy - y, zx - x), 0.0]
            act["fire"] = [False, False]

        mf.draw.point([zx, zy, z], "#40c0ff", 6)
        mf.draw.label(f"v6{' COVER' if self.cover_goal else ''}{' GHOST' if ghost_ok else ''} hp {hp:.0f}/{max_hp:.0f} · z{'1' if in_zone else '0'} · e {len(hostile_mechs)}" + (f" · loot {loot_target['kind']}" if loot_target else ""))
        return act


mf.run(Bot)
