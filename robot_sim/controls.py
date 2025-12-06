from __future__ import annotations

import ast
import configparser
import json
import math
import os
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np

if TYPE_CHECKING:
    from sim import Obstacle, Robot, Target, World

CONFIG_PATH = Path(__file__).with_name("term_math.cfg")
COMPILED_JSON_PATH = Path(__file__).with_name("term_config.json")

_DEFAULT_CONTROLLER_MODE = (
    os.environ.get("ROBOT_CONTROLLER_MODE", "voronoi").strip().lower()
)
if _DEFAULT_CONTROLLER_MODE not in {"voronoi", "terms"}:
    _DEFAULT_CONTROLLER_MODE = "voronoi"

_VORONOI_GAIN = float(os.environ.get("ROBOT_VORONOI_GAIN", "1.2"))
_VORONOI_RESOLUTION = float(os.environ.get("ROBOT_VORONOI_RESOLUTION", "24.0"))
_VORONOI_DENSITY_SIGMA = float(os.environ.get("ROBOT_VORONOI_SIGMA", "220.0"))
_VORONOI_OBS_MARGIN = float(os.environ.get("ROBOT_VORONOI_OBS_MARGIN", "6.0"))
_VORONOI_LINK_GAIN = float(os.environ.get("ROBOT_VORONOI_LINK_GAIN", "1.4"))
_VORONOI_LINK_RANGE = float(os.environ.get("ROBOT_VORONOI_LINK_RANGE", "0.0"))
_VORONOI_OVERLAP_MARGIN = float(os.environ.get("ROBOT_VORONOI_OVERLAP", "20.0"))

SAFE_FUNCTIONS = {
    name: getattr(math, name)
    for name in (
        "sin",
        "cos",
        "tan",
        "asin",
        "acos",
        "atan",
        "atan2",
        "sinh",
        "cosh",
        "tanh",
        "exp",
        "log",
        "log10",
        "sqrt",
        "fabs",
        "floor",
        "ceil",
    )
}
SAFE_FUNCTIONS.update(
    {
        "abs": abs,
        "min": min,
        "max": max,
        "pow": pow,
        "clamp": lambda value, lo, hi: max(lo, min(hi, value)),
    }
)
SAFE_CONSTANTS = {
    "pi": math.pi,
    "tau": math.tau,
    "e": math.e,
}

ALLOWED_NODE_TYPES = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Constant,
    ast.Name,
    ast.Load,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Pow,
    ast.USub,
    ast.UAdd,
    ast.Mod,
    ast.FloorDiv,
    ast.Compare,
    ast.BoolOp,
    ast.And,
    ast.Or,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
    ast.Eq,
    ast.NotEq,
    ast.IfExp,
    ast.Call,
    ast.Attribute,
    ast.Subscript,
    ast.Slice,
    ast.Index,
    ast.Num,
    ast.List,
    ast.Tuple,
)
RESERVED_KEYS = {
    "type",
    "label",
    "description",
    "active",
    "weight",
    "vx",
    "vy",
    "condition",
    "potential",
}


@dataclass
class TermInfo:
    name: str
    label: str
    description: str
    weight: float
    active: bool
    kind: str
    vx_expr: str = "0.0"
    vy_expr: str = "0.0"
    condition_expr: str = ""
    potential_expr: str = ""
    constants: Dict[str, float] = field(default_factory=dict)
    _compiled_vx: Optional[Callable[[Dict[str, float]], float]] = None
    _compiled_vy: Optional[Callable[[Dict[str, float]], float]] = None
    _compiled_condition: Optional[Callable[[Dict[str, float]], float]] = None
    _compiled_potential: Optional[Callable[[Dict[str, float]], float]] = None

    def clone(self) -> TermInfo:
        return TermInfo(
            name=self.name,
            label=self.label,
            description=self.description,
            weight=self.weight,
            active=self.active,
            kind=self.kind,
            vx_expr=self.vx_expr,
            vy_expr=self.vy_expr,
            condition_expr=self.condition_expr,
            potential_expr=self.potential_expr,
            constants=dict(self.constants),
            _compiled_vx=self._compiled_vx,
            _compiled_vy=self._compiled_vy,
            _compiled_condition=self._compiled_condition,
            _compiled_potential=self._compiled_potential,
        )


_registry: Dict[str, TermInfo] = {}
_default_terms: Dict[str, TermInfo] = {}
_config_state: Dict[str, object] = {
    "status": "Using built-in terms",
    "error": None,
    "last_loaded": None,
    "path": CONFIG_PATH,
    "term_names": [],
    "runtime_error": None,
    "json_path": COMPILED_JSON_PATH,
    "last_json_saved": None,
    "last_json_loaded": None,
}

_controller_mode: str = _DEFAULT_CONTROLLER_MODE
_voronoi_samples_cache: Dict[str, object] = {
    "bounds": None,
    "resolution": None,
    "points": None,
}
_voronoi_velocity_cache_key: Optional[Tuple[object, ...]] = None
_voronoi_velocity_cache: List[np.ndarray] = []


def iter_terms() -> Iterable[Tuple[str, TermInfo]]:
    return _registry.items()


def set_weight(name: str, weight: float) -> None:
    info = _registry.get(name)
    if not info:
        return
    info.weight = max(0.0, min(2.0, weight))


def get_weight(name: str) -> float:
    info = _registry.get(name)
    return info.weight if info else 0.0


def toggle_term(name: str) -> bool:
    info = _registry.get(name)
    if not info:
        return False
    info.active = not info.active
    return info.active


def term_active(name: str) -> bool:
    info = _registry.get(name)
    return bool(info and info.active)


def set_term_active(name: str, active: bool) -> None:
    info = _registry.get(name)
    if info:
        info.active = active


def get_controller_mode() -> str:
    return _controller_mode


def set_controller_mode(mode: str) -> None:
    global _controller_mode
    normalized = mode.strip().lower()
    if normalized not in {"voronoi", "terms"}:
        return
    if normalized != _controller_mode:
        _controller_mode = normalized
        _invalidate_voronoi_cache()


def _use_voronoi_controller() -> bool:
    return _controller_mode == "voronoi"


def _invalidate_voronoi_cache() -> None:
    global _voronoi_velocity_cache_key, _voronoi_velocity_cache
    _voronoi_velocity_cache_key = None
    _voronoi_velocity_cache = []


def compute_velocity(
    robot: Robot,
    idx: int,
    robots: List[Robot],
    world: World,
    obstacles: List[Obstacle],
    targets: List[Target],
) -> np.ndarray:
    if _use_voronoi_controller():
        velocities = _get_cached_voronoi_velocities(robots, world, obstacles, targets)
        if 0 <= idx < len(velocities):
            return velocities[idx]
        return np.zeros(2)
    total = np.zeros(2)
    _config_state["runtime_error"] = None
    chain_links = _build_chain_links(robots)
    for name, term in _registry.items():
        if not term.active:
            continue
        try:
            contribution = _evaluate_term(
                term, robot, idx, robots, world, obstacles, targets, chain_links
            )
        except Exception as exc:  # pragma: no cover - runtime safety
            _config_state["runtime_error"] = f"{name}: {exc}"
            contribution = np.zeros(2)
        total += term.weight * contribution
    return total


def clip_speed(v: np.ndarray, vmax: float) -> np.ndarray:
    norm = np.linalg.norm(v)
    if norm == 0 or norm <= vmax:
        return v
    return v * (vmax / norm)


def _get_cached_voronoi_velocities(
    robots: List[Robot],
    world: World,
    obstacles: List[Obstacle],
    targets: List[Target],
) -> List[np.ndarray]:
    global _voronoi_velocity_cache_key, _voronoi_velocity_cache
    key = _make_voronoi_cache_key(robots, world, obstacles, targets)
    if key != _voronoi_velocity_cache_key:
        _voronoi_velocity_cache = _compute_voronoi_velocities(
            robots, world, obstacles, targets
        )
        _voronoi_velocity_cache_key = key
    return _voronoi_velocity_cache


def _make_voronoi_cache_key(
    robots: List[Robot],
    world: World,
    obstacles: List[Obstacle],
    targets: List[Target],
) -> Tuple[object, ...]:
    robot_key = tuple((float(robot.pos[0]), float(robot.pos[1])) for robot in robots)
    obstacle_key = tuple(
        (float(obstacle.c[0]), float(obstacle.c[1]), float(obstacle.R))
        for obstacle in obstacles
    )
    target_key = tuple(
        (float(target.pos[0]), float(target.pos[1])) for target in targets
    )
    world_key = (
        float(world.xmin),
        float(world.xmax),
        float(world.ymin),
        float(world.ymax),
    )
    return (robot_key, obstacle_key, target_key, world_key, _controller_mode)


def _compute_voronoi_velocities(
    robots: List[Robot],
    world: World,
    obstacles: List[Obstacle],
    targets: List[Target],
) -> List[np.ndarray]:
    if not robots:
        return []
    points = _ensure_sample_points(world, _VORONOI_RESOLUTION)
    if points.size == 0:
        return [np.zeros(2) for _ in robots]
    mask = _filter_points_by_obstacles(points, obstacles)
    valid_points = points[mask]
    if valid_points.size == 0:
        return [np.zeros(2) for _ in robots]
    importance = _importance_density(valid_points, targets)
    cell_weight = max(_VORONOI_RESOLUTION, 1.0) ** 2
    weights = importance * cell_weight
    robot_positions = np.array([np.array(robot.pos, dtype=float) for robot in robots])
    diffs = valid_points[:, None, :] - robot_positions[None, :, :]
    dists_sq = np.sum(diffs * diffs, axis=2)
    ownership = np.argmin(dists_sq, axis=1)
    accum = np.zeros_like(robot_positions)
    total_weight = np.zeros(len(robots))
    for idx, robot_idx in enumerate(ownership):
        w = weights[idx]
        accum[robot_idx] += valid_points[idx] * w
        total_weight[robot_idx] += w
    velocities: List[np.ndarray] = []
    for idx, pos in enumerate(robot_positions):
        if total_weight[idx] > 1e-6:
            centroid = accum[idx] / total_weight[idx]
        else:
            centroid = pos
        velocities.append(_VORONOI_GAIN * (centroid - pos))
    _apply_connectivity_links(velocities, robots, robot_positions)
    return velocities


def _apply_connectivity_links(
    velocities: List[np.ndarray], robots: List[Robot], positions: np.ndarray
) -> None:
    if len(velocities) != len(robots) or len(robots) < 2:
        return
    gain = max(_VORONOI_LINK_GAIN, 0.0)
    if gain <= 0.0:
        return
    coverage = np.array(
        [float(getattr(robot, "coverage_radius", robot.r)) for robot in robots]
    )
    base_range = max(_VORONOI_LINK_RANGE, 0.0)
    overlap = max(_VORONOI_OVERLAP_MARGIN, 0.0)
    chain_links = _build_chain_links(robots)
    neighbor_sets: List[Set[int]] = [set() for _ in robots]
    limit_maps: List[Dict[int, float]] = [dict() for _ in robots]

    def add_link(i: int, j: int) -> None:
        if i == j:
            return
        allowed = _link_limit(i, j, coverage, base_range, overlap)
        if allowed <= 0.0:
            return
        neighbor_sets[i].add(j)
        neighbor_sets[j].add(i)
        limit_maps[i][j] = allowed
        limit_maps[j][i] = allowed

    for idx, linked in chain_links.items():
        for neighbor_idx in linked:
            add_link(idx, neighbor_idx)

    _ensure_base_connectivity(
        neighbor_sets, limit_maps, coverage, base_range, overlap, positions
    )

    neighbor_lists: List[List[int]] = []
    limit_lists: List[List[float]] = []
    for idx in range(len(robots)):
        linked = sorted(neighbor_sets[idx])
        neighbor_lists.append(linked)
        limit_lists.append([limit_maps[idx][nbr] for nbr in linked])

    _project_link_constraints(velocities, positions, neighbor_lists, limit_lists, gain)



def _project_link_constraints(
    velocities: List[np.ndarray],
    positions: np.ndarray,
    neighbors: List[List[int]],
    limits: List[List[float]],
    gain: float,
) -> None:
    dt = 1.0
    eps = 1e-6
    max_iters = 8
    for _ in range(max_iters):
        any_violation = False
        for idx, linked in enumerate(neighbors):
            for entry_idx, neighbor_idx in enumerate(linked):
                allowed = limits[idx][entry_idx]
                if allowed <= 0.0:
                    continue
                rel_pos = positions[idx] - positions[neighbor_idx]
                dist = float(np.linalg.norm(rel_pos))
                if dist <= eps:
                    continue
                rel_vel = velocities[idx] - velocities[neighbor_idx]
                rel_next = rel_pos + rel_vel * dt
                next_dist = float(np.linalg.norm(rel_next))
                if next_dist <= allowed:
                    continue
                any_violation = True
                correction_dir = rel_next / (next_dist + eps)
                excess = next_dist - allowed
                impulse = correction_dir * excess * 0.5 * gain
                velocities[idx] -= impulse / dt
                velocities[neighbor_idx] += impulse / dt
        if not any_violation:
            break

def _link_limit(
    idx: int,
    neighbor_idx: int,
    coverage: np.ndarray,
    base_range: float,
    overlap: float,
) -> float:
    allowed = max(coverage[idx] + coverage[neighbor_idx] - overlap, 0.0)
    if base_range > 0.0:
        allowed = min(allowed, base_range)
    return allowed


def _ensure_base_connectivity(
    neighbor_sets: List[Set[int]],
    limit_maps: List[Dict[int, float]],
    coverage: np.ndarray,
    base_range: float,
    overlap: float,
    positions: np.ndarray,
) -> None:
    if not neighbor_sets:
        return
    n = len(neighbor_sets)
    visited = set()
    components: List[List[int]] = []
    for idx in range(n):
        if idx in visited:
            continue
        stack = [idx]
        comp: List[int] = []
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            comp.append(node)
            stack.extend(neighbor_sets[node])
        components.append(comp)
    if len(components) <= 1:
        return
    base_idx = 0 if n else None
    base_comp = None
    for comp in components:
        if base_idx is not None and base_idx in comp:
            base_comp = comp
            break
    if base_comp is None:
        base_comp = components[0]
    base_set = set(base_comp)
    for comp in components:
        if base_set.issuperset(comp):
            continue
        best_pair = None
        best_dist = float("inf")
        for node in comp:
            for anchor in base_set:
                dist = float(np.linalg.norm(positions[node] - positions[anchor]))
                if dist < best_dist:
                    best_dist = dist
                    best_pair = (node, anchor)
        if not best_pair:
            continue
        node, anchor = best_pair
        allowed = _link_limit(node, anchor, coverage, base_range, overlap)
        if allowed <= 0.0:
            continue
        neighbor_sets[node].add(anchor)
        neighbor_sets[anchor].add(node)
        limit_maps[node][anchor] = allowed
        limit_maps[anchor][node] = allowed
        base_set.update(comp)



def _importance_density(points: np.ndarray, targets: List[Target]) -> np.ndarray:
    density = np.ones(points.shape[0])
    if not targets:
        return density
    sigma_sq = max(_VORONOI_DENSITY_SIGMA, 1.0) ** 2
    inv_two_sigma = -0.5 / sigma_sq
    for target in targets:
        delta = points - np.array(target.pos, dtype=float)
        dist_sq = np.sum(delta * delta, axis=1)
        density += np.exp(dist_sq * inv_two_sigma)
    return density


def _filter_points_by_obstacles(
    points: np.ndarray, obstacles: List[Obstacle]
) -> np.ndarray:
    if not obstacles:
        return np.ones(points.shape[0], dtype=bool)
    mask = np.ones(points.shape[0], dtype=bool)
    for obstacle in obstacles:
        inflated = float(obstacle.R) + _VORONOI_OBS_MARGIN
        delta = points - np.array(obstacle.c, dtype=float)
        dist_sq = np.sum(delta * delta, axis=1)
        mask &= dist_sq >= inflated * inflated
    return mask


def _ensure_sample_points(world: World, resolution: float) -> np.ndarray:
    cache = _voronoi_samples_cache
    bounds = (
        float(world.xmin),
        float(world.xmax),
        float(world.ymin),
        float(world.ymax),
    )
    if (
        cache["points"] is not None
        and cache["bounds"] == bounds
        and cache["resolution"] == resolution
    ):
        return cache["points"]
    if resolution <= 0.0:
        return np.empty((0, 2))
    xmin, xmax, ymin, ymax = bounds
    if xmax <= xmin or ymax <= ymin:
        return np.empty((0, 2))
    xs = np.arange(xmin + 0.5 * resolution, xmax, resolution)
    ys = np.arange(ymin + 0.5 * resolution, ymax, resolution)
    if xs.size == 0:
        xs = np.array([(xmin + xmax) * 0.5])
    if ys.size == 0:
        ys = np.array([(ymin + ymax) * 0.5])
    grid_x, grid_y = np.meshgrid(xs, ys)
    points = np.stack((grid_x.ravel(), grid_y.ravel()), axis=1)
    cache["bounds"] = bounds
    cache["resolution"] = resolution
    cache["points"] = points
    return points


def numeric_grad(
    phi_fn: Callable[[np.ndarray], float], position: np.ndarray, step: float = 1e-3
) -> np.ndarray:
    dx = np.array([step, 0.0])
    dy = np.array([0.0, step])
    gx = (phi_fn(position + dx) - phi_fn(position - dx)) / (2.0 * step)
    gy = (phi_fn(position + dy) - phi_fn(position - dy)) / (2.0 * step)
    return np.array([gx, gy])


def reload_config_terms() -> Tuple[bool, str]:
    if not CONFIG_PATH.exists():
        _apply_default_terms()
        msg = f"Missing {CONFIG_PATH.name}; using defaults"
        _config_state.update(
            {
                "status": msg,
                "error": None,
                "last_loaded": None,
                "term_names": list(_registry.keys()),
            }
        )
        return False, msg
    try:
        parser = configparser.ConfigParser(interpolation=None)
        parser.optionxform = str  # preserve case
        parser.read(CONFIG_PATH, encoding="utf-8")
        terms: Dict[str, TermInfo] = {}
        for section in parser.sections():
            info = _build_term_from_section(section, parser[section])
            _compile_term(info)
            terms[info.name] = info
        _registry.clear()
        _registry.update({name: info.clone() for name, info in terms.items()})
        _export_terms_json(terms)
        now = datetime.now()
        _config_state.update(
            {
                "status": f"Loaded {len(terms)} term(s)",
                "error": None,
                "last_loaded": now,
                "term_names": list(terms.keys()),
                "runtime_error": None,
                "last_json_saved": now,
            }
        )
        return True, _config_state["status"]
    except Exception as exc:  # pragma: no cover - diagnostics
        _config_state["error"] = traceback.format_exc()
        _config_state["status"] = "Error loading term_math.cfg"
        return False, str(exc)


def reload_json_terms() -> Tuple[bool, str]:
    if not COMPILED_JSON_PATH.exists():
        return False, f"Missing {COMPILED_JSON_PATH.name}"
    try:
        data = json.loads(COMPILED_JSON_PATH.read_text(encoding="utf-8"))
        terms: Dict[str, TermInfo] = {}
        for entry in data:
            info = _term_from_json(entry)
            _compile_term(info)
            terms[info.name] = info
        _registry.clear()
        _registry.update({name: info.clone() for name, info in terms.items()})
        now = datetime.now()
        _config_state.update(
            {
                "status": f"Loaded {len(terms)} term(s) from JSON",
                "error": None,
                "last_loaded": now,
                "last_json_loaded": now,
                "term_names": list(terms.keys()),
                "runtime_error": None,
            }
        )
        return True, _config_state["status"]
    except Exception as exc:  # pragma: no cover - diagnostics
        _config_state["error"] = traceback.format_exc()
        _config_state["status"] = "Error loading term_config.json"
        return False, str(exc)


def save_entries_to_json(entries: List[Dict[str, object]]) -> Tuple[bool, str]:
    try:
        normalized: List[Dict[str, object]] = []
        names: List[str] = []
        for entry in entries:
            info = _term_from_json(entry)
            _compile_term(info)
            normalized.append(_info_to_entry(info))
            names.append(info.name)
        COMPILED_JSON_PATH.write_text(
            json.dumps(normalized, indent=2), encoding="utf-8"
        )
        now = datetime.now()
        _config_state.update(
            {
                "last_json_saved": now,
                "json_path": COMPILED_JSON_PATH,
                "status": f"Saved {len(normalized)} term(s) to JSON",
                "term_names": names,
                "error": None,
            }
        )
        return True, _config_state["status"]
    except Exception as exc:
        _config_state["error"] = str(exc)
        return False, str(exc)


def get_config_state() -> Dict[str, object]:
    return {
        "status": _config_state.get("status"),
        "error": _config_state.get("error"),
        "last_loaded": _config_state.get("last_loaded"),
        "path": _config_state.get("path"),
        "term_names": list(_config_state.get("term_names", [])),
        "runtime_error": _config_state.get("runtime_error"),
        "json_path": _config_state.get("json_path"),
        "last_json_saved": _config_state.get("last_json_saved"),
        "last_json_loaded": _config_state.get("last_json_loaded"),
    }


def _apply_default_terms() -> None:
    if not _default_terms:
        defaults = {
            "go_to": TermInfo(
                name="go_to",
                label="Go To",
                description="Attractive pull toward assigned target",
                weight=1.0,
                active=True,
                kind="single",
                vx_expr="1.2 * (goal_x - x)",
                vy_expr="1.2 * (goal_y - y)",
            ),
            "avoid_obs": TermInfo(
                name="avoid_obs",
                label="Avoid Obst",
                description="Hard wall push near obstacles",
                weight=0.9,
                active=True,
                kind="obstacle",
                vx_expr="(dist < radius + infl) * k * (dx / (dist + soft_dir)) / ((max(dist - radius, 0.0) + soft_gap)**2)",
                vy_expr="(dist < radius + infl) * k * (dy / (dist + soft_dir)) / ((max(dist - radius, 0.0) + soft_gap)**2)",
                condition_expr="dist < radius + infl",
                constants={
                    "k": 320000.0,
                    "infl": 120.0,
                    "soft_dir": 1.0,
                    "soft_gap": 6.0,
                },
            ),
            "avoid_robots": TermInfo(
                name="avoid_robots",
                label="Separation",
                description="Hard separation (inverse-square push)",
                weight=0.6,
                active=True,
                kind="robots",
                vx_expr="(dist < sum_radius + infl) * k * (dx / (dist + soft_dir)) / ((max(dist - sum_radius, 0.0) + soft_gap)**2)",
                vy_expr="(dist < sum_radius + infl) * k * (dy / (dist + soft_dir)) / ((max(dist - sum_radius, 0.0) + soft_gap)**2)",
                condition_expr="dist < sum_radius + infl",
                constants={
                    "k": 180000.0,
                    "infl": 80.0,
                    "soft_dir": 1.0,
                    "soft_gap": 6.0,
                },
            ),
            "bounds": TermInfo(
                name="bounds",
                label="Bounds",
                description="Push inward near edges",
                weight=0.6,
                active=True,
                kind="single",
                vx_expr="(dist_left < 80.0) * 260.0 * (80.0 - dist_left) / 80.0 - (dist_right < 80.0) * 260.0 * (80.0 - dist_right) / 80.0",
                vy_expr="(dist_bottom < 80.0) * 260.0 * (80.0 - dist_bottom) / 80.0 - (dist_top < 80.0) * 260.0 * (80.0 - dist_top) / 80.0",
            ),
            "phi": TermInfo(
                name="phi",
                label="Phi",
                description="Gradient descent",
                weight=0.3,
                active=False,
                kind="potential",
                potential_expr="(x - 0.0)**2 + (y - 0.0)**2",
            ),
        }
        for info in defaults.values():
            _compile_term(info)
        _default_terms.update(defaults)
    _registry.clear()
    _registry.update({name: info.clone() for name, info in _default_terms.items()})


def _export_terms_json(terms: Dict[str, TermInfo]) -> None:
    payload = [_info_to_entry(info) for info in terms.values()]
    COMPILED_JSON_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _info_to_entry(info: TermInfo) -> Dict[str, object]:
    return {
        "name": info.name,
        "label": info.label,
        "description": info.description,
        "weight": info.weight,
        "active": info.active,
        "kind": info.kind,
        "vx": info.vx_expr,
        "vy": info.vy_expr,
        "condition": info.condition_expr,
        "potential": info.potential_expr,
        "constants": info.constants,
    }


def _term_from_json(entry: Dict[str, object]) -> TermInfo:
    info = TermInfo(
        name=str(entry.get("name")),
        label=str(entry.get("label", entry.get("name", "term"))),
        description=str(entry.get("description", "")),
        weight=float(entry.get("weight", 1.0)),
        active=bool(entry.get("active", True)),
        kind=str(entry.get("kind", "single")),
        vx_expr=str(entry.get("vx", "0.0")),
        vy_expr=str(entry.get("vy", "0.0")),
        condition_expr=str(entry.get("condition", "")),
        potential_expr=str(entry.get("potential", "")),
        constants=dict(entry.get("constants", {})),
    )
    return info


def _build_term_from_section(name: str, section: configparser.SectionProxy) -> TermInfo:
    kind = section.get("type", "single").strip().lower()
    label = section.get("label", name)
    description = section.get("description", "")
    active = section.getboolean("active", fallback=True)
    weight = section.getfloat("weight", fallback=1.0)
    constants: Dict[str, float] = {}
    for key, value in section.items():
        if key in RESERVED_KEYS:
            continue
        constants[key] = _parse_constant(value)
    info = TermInfo(
        name=name,
        label=label,
        description=description,
        weight=weight,
        active=active,
        kind=kind,
        vx_expr=section.get("vx", "0.0"),
        vy_expr=section.get("vy", "0.0"),
        condition_expr=section.get("condition", ""),
        potential_expr=section.get("potential", ""),
        constants=constants,
    )
    return info


def _parse_constant(raw: str) -> float:
    try:
        return float(raw)
    except ValueError:
        return raw


def _compile_term(info: TermInfo) -> None:
    allowed_names = set(SAFE_FUNCTIONS.keys()) | set(SAFE_CONSTANTS.keys())
    allowed_names.update(
        {
            "x",
            "y",
            "idx",
            "goal_x",
            "goal_y",
            "goal_dx",
            "goal_dy",
            "goal_dist",
            "xmin",
            "xmax",
            "ymin",
            "ymax",
            "dist_left",
            "dist_right",
            "dist_top",
            "dist_bottom",
            "num_robots",
            "num_targets",
            "num_obstacles",
            "ob_x",
            "ob_y",
            "radius",
            "dx",
            "dy",
            "dist",
            "other_x",
            "other_y",
            "other_idx",
            "other_radius",
            "sum_radius",
            "coverage_radius",
            "other_coverage_radius",
            "neighbor_gate",
        }
    )
    allowed_names.update(info.constants.keys())

    if info.kind == "potential":
        info._compiled_potential = _compile_expression(
            info.potential_expr or "0.0", allowed_names
        )
        info._compiled_vx = None
        info._compiled_vy = None
        info._compiled_condition = None
        return
    info._compiled_vx = _compile_expression(info.vx_expr or "0.0", allowed_names)
    info._compiled_vy = _compile_expression(info.vy_expr or "0.0", allowed_names)
    if info.condition_expr:
        info._compiled_condition = _compile_expression(
            info.condition_expr, allowed_names
        )
    else:
        info._compiled_condition = None


def _compile_expression(
    expr: str, allowed_names: Iterable[str]
) -> Callable[[Dict[str, float]], float]:
    expression = expr.strip() or "0.0"
    tree = ast.parse(expression, mode="eval")
    for node in ast.walk(tree):
        if not isinstance(node, ALLOWED_NODE_TYPES):
            raise ValueError(f"Unsupported syntax in expression: {expression}")
        if isinstance(node, ast.Call):
            if (
                not isinstance(node.func, ast.Name)
                or node.func.id not in SAFE_FUNCTIONS
            ):
                func_label = getattr(node.func, "id", "<expr>")
                raise ValueError(f"Disallowed function '{func_label}'")
        if isinstance(node, ast.Name):
            if node.id not in allowed_names:
                raise ValueError(f"Unknown symbol '{node.id}' in expression")
    code = compile(tree, "<term_expr>", "eval")

    def evaluator(context: Dict[str, float]) -> float:
        env = dict(SAFE_FUNCTIONS)
        env.update(SAFE_CONSTANTS)
        return eval(code, env, context)

    return evaluator


def _evaluate_term(
    term: TermInfo,
    robot: Robot,
    idx: int,
    robots: List[Robot],
    world: World,
    obstacles: List[Obstacle],
    targets: List[Target],
    chain_links: Optional[Dict[int, Set[int]]],
) -> np.ndarray:
    base_ctx = _base_context(robot, idx, robots, world, obstacles, targets)
    base_ctx.update(term.constants)
    if term.kind == "single":
        vx = term._compiled_vx(base_ctx) if term._compiled_vx else 0.0
        vy = term._compiled_vy(base_ctx) if term._compiled_vy else 0.0
        return np.array([vx, vy])
    if term.kind == "obstacle":
        total = np.zeros(2)
        for obstacle in obstacles:
            ctx = base_ctx.copy()
            dx = float(robot.pos[0] - obstacle.c[0])
            dy = float(robot.pos[1] - obstacle.c[1])
            dist = math.hypot(dx, dy)
            ctx.update(
                {
                    "ob_x": float(obstacle.c[0]),
                    "ob_y": float(obstacle.c[1]),
                    "radius": float(obstacle.R),
                    "dx": dx,
                    "dy": dy,
                    "dist": dist,
                }
            )
            if term._compiled_condition and not term._compiled_condition(ctx):
                continue
            vx = term._compiled_vx(ctx) if term._compiled_vx else 0.0
            vy = term._compiled_vy(ctx) if term._compiled_vy else 0.0
            total += np.array([vx, vy])
        return total
    if term.kind == "robots":
        total = np.zeros(2)
        for j, other in enumerate(robots):
            if j == idx:
                continue
            ctx = base_ctx.copy()
            dx = float(robot.pos[0] - other.pos[0])
            dy = float(robot.pos[1] - other.pos[1])
            dist = math.hypot(dx, dy)
            ctx.update(
                {
                    "other_x": float(other.pos[0]),
                    "other_y": float(other.pos[1]),
                    "other_idx": float(j),
                    "other_radius": float(other.r),
                    "dx": dx,
                    "dy": dy,
                    "dist": dist,
                    "sum_radius": float(robot.r + other.r),
                    "coverage_radius": float(
                        getattr(robot, "coverage_radius", robot.r)
                    ),
                    "other_coverage_radius": float(
                        getattr(other, "coverage_radius", other.r)
                    ),
                    "neighbor_gate": float(
                        1.0
                        if chain_links and idx in chain_links and j in chain_links[idx]
                        else 0.0
                    ),
                }
            )
            if term._compiled_condition and not term._compiled_condition(ctx):
                continue
            vx = term._compiled_vx(ctx) if term._compiled_vx else 0.0
            vy = term._compiled_vy(ctx) if term._compiled_vy else 0.0
            total += np.array([vx, vy])
        return total
    if term.kind == "potential":
        if not term._compiled_potential:
            return np.zeros(2)

        def potential_value(point: np.ndarray) -> float:
            ctx = base_ctx.copy()
            ctx.update({"x": float(point[0]), "y": float(point[1])})
            return float(term._compiled_potential(ctx))

        grad = numeric_grad(potential_value, np.array(robot.pos, dtype=float))
        return -grad
    # fallback
    return np.zeros(2)


def _base_context(
    robot: Robot,
    idx: int,
    robots: List[Robot],
    world: World,
    obstacles: List[Obstacle],
    targets: List[Target],
) -> Dict[str, float]:
    x = float(robot.pos[0])
    y = float(robot.pos[1])
    coverage_radius = float(getattr(robot, "coverage_radius", robot.r))
    xmin = float(world.xmin)
    xmax = float(world.xmax)
    ymin = float(world.ymin)
    ymax = float(world.ymax)
    if targets:
        target = targets[idx % len(targets)]
        goal_x = float(target.pos[0])
        goal_y = float(target.pos[1])
    else:
        goal_x = x
        goal_y = y
    goal_dx = goal_x - x
    goal_dy = goal_y - y
    goal_dist = math.hypot(goal_dx, goal_dy)
    ctx = {
        "x": x,
        "y": y,
        "idx": float(idx),
        "goal_x": goal_x,
        "goal_y": goal_y,
        "goal_dx": goal_dx,
        "goal_dy": goal_dy,
        "goal_dist": goal_dist,
        "xmin": xmin,
        "xmax": xmax,
        "ymin": ymin,
        "ymax": ymax,
        "dist_left": x - xmin,
        "dist_right": xmax - x,
        "dist_bottom": y - ymin,
        "dist_top": ymax - y,
        "num_robots": float(len(robots)),
        "num_targets": float(len(targets)),
        "num_obstacles": float(len(obstacles)),
        "coverage_radius": coverage_radius,
    }
    return ctx


def _build_chain_links(robots: List[Robot]) -> Dict[int, Set[int]]:
    links: Dict[int, Set[int]] = {idx: set() for idx in range(len(robots))}
    for idx, robot in enumerate(robots):
        distances: List[Tuple[float, int]] = []
        for other_idx, other in enumerate(robots):
            if other_idx == idx:
                continue
            dist = float(np.linalg.norm(robot.pos - other.pos))
            distances.append((dist, other_idx))
        distances.sort(key=lambda pair: pair[0])
        for _, neighbor_idx in distances[:2]:
            links[idx].add(neighbor_idx)
            links[neighbor_idx].add(idx)
    return links


_apply_default_terms()
if CONFIG_PATH.exists():
    reload_config_terms()
else:
    _config_state["status"] = f"Using defaults (missing {CONFIG_PATH.name})"
