from __future__ import annotations

import ast
import configparser
import json
import math
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


def compute_velocity(
    robot: Robot,
    idx: int,
    robots: List[Robot],
    world: World,
    obstacles: List[Obstacle],
    targets: List[Target],
) -> np.ndarray:
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
                        if chain_links
                        and idx in chain_links
                        and j in chain_links[idx]
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
