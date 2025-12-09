from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

SCENE_PATH = Path(__file__).with_name("scene.json")


@dataclass
class World:
    xmin: float
    xmax: float
    ymin: float
    ymax: float


@dataclass
class Scene:
    world: World
    goal_boxes: List[Tuple[float, float, float, float]]
    obstacles: np.ndarray


def _parse_world(payload: dict) -> World:
    world = payload.get("world", {})
    return World(
        float(world.get("xmin", 40.0)),
        float(world.get("xmax", 2440.0)),
        float(world.get("ymin", 40.0)),
        float(world.get("ymax", 1540.0)),
    )


def _parse_goal_boxes(payload: dict) -> List[Tuple[float, float, float, float]]:
    boxes = []
    for item in payload.get("goal_boxes", []):
        try:
            xmin = float(item.get("xmin"))
            xmax = float(item.get("xmax"))
            ymin = float(item.get("ymin"))
            ymax = float(item.get("ymax"))
        except (TypeError, ValueError):
            continue
        boxes.append((xmin, xmax, ymin, ymax))
    return boxes


def _parse_obstacles(payload: dict) -> np.ndarray:
    obs = []
    for entry in payload.get("obstacles", []):
        center = entry.get("c") or entry.get("center")
        if not center or len(center) != 2:
            continue
        radius = float(entry.get("R", entry.get("radius", 0.0)))
        obs.append([float(center[0]), float(center[1]), radius])
    if not obs:
        return np.zeros((0, 3), dtype=np.float32)
    return np.array(obs, dtype=np.float32)


def load_scene(path: Optional[Path] = None) -> Scene:
    scene_path = path or SCENE_PATH
    if scene_path.exists():
        payload = json.loads(scene_path.read_text())
    else:
        payload = {}
    world = _parse_world(payload)
    boxes = _parse_goal_boxes(payload)
    obstacles = _parse_obstacles(payload)
    return Scene(world=world, goal_boxes=boxes, obstacles=obstacles)


def load_world() -> World:
    return load_scene().world


def coverage_box(world: World, width: float, height: float) -> Tuple[float, float, float, float]:
    world_width = world.xmax - world.xmin
    world_height = world.ymax - world.ymin
    box_w = min(width, world_width * 0.75)
    box_h = min(height, world_height * 0.75)
    cx = 0.5 * (world.xmin + world.xmax)
    cy = 0.5 * (world.ymin + world.ymax)
    return (
        cx - 0.5 * box_w,
        cx + 0.5 * box_w,
        cy - 0.5 * box_h,
        cy + 0.5 * box_h,
    )


def density_grid(box: Tuple[float, float, float, float], spacing: float) -> np.ndarray:
    xmin, xmax, ymin, ymax = box
    xs = np.arange(xmin + 0.5 * spacing, xmax, spacing)
    ys = np.arange(ymin + 0.5 * spacing, ymax, spacing)
    if xs.size == 0:
        xs = np.array([0.5 * (xmin + xmax)])
    if ys.size == 0:
        ys = np.array([0.5 * (ymin + ymax)])
    grid_x, grid_y = np.meshgrid(xs, ys)
    return np.stack((grid_x.ravel(), grid_y.ravel()), axis=1).astype(np.float32)


def sequential_spawn_position(
    world: World,
    base: np.ndarray,
    order_idx: int,
    coverage_radius: float,
) -> np.ndarray:
    center_y = 0.5 * (world.ymin + world.ymax)
    if order_idx == 0:
        offset = np.array([coverage_radius * 1.5, 0.0], dtype=np.float32)
        return np.array(base) + offset
    x = base[0] + coverage_radius * (1.4 + order_idx * 0.9)
    x = min(x, world.xmax - coverage_radius)
    direction = -1 if order_idx % 2 == 0 else 1
    band = ((order_idx // 2) % 3) + 1
    y_offset = direction * coverage_radius * (0.35 + 0.12 * band)
    y = np.clip(center_y + y_offset, world.ymin + coverage_radius, world.ymax - coverage_radius)
    return np.array([x, y], dtype=np.float32)


def cluster_spawn_position(
    world: World,
    base: np.ndarray,
    order_idx: int,
    coverage_radius: float,
    base_radius: float,
    target_point: np.ndarray,
) -> np.ndarray:
    base_vec = np.array(base, dtype=np.float32)
    direction = np.array(target_point, dtype=np.float32) - base_vec
    norm = np.linalg.norm(direction)
    if norm < 1.0e-6:
        direction = np.array([1.0, 0.0], dtype=np.float32)
    else:
        direction = direction / norm
    perp = np.array([-direction[1], direction[0]], dtype=np.float32)
    max_dist = base_radius * 0.92
    growth = 1.0 - np.exp(-0.55 * (order_idx + 1))
    radial_dist = np.clip(growth * max_dist, base_radius * 0.15, max_dist)
    lane_cycle = (order_idx % 3) - 1
    lane_width = min(coverage_radius * 0.18, base_radius * 0.25)
    lateral_offset = lane_cycle * lane_width * 0.35
    pos = base_vec + direction * radial_dist + perp * lateral_offset
    offset = pos - base_vec
    dist = np.linalg.norm(offset)
    if dist > max_dist:
        pos = base_vec + offset / (dist + 1.0e-6) * max_dist
    pos[0] = np.clip(pos[0], world.xmin + coverage_radius * 0.5, world.xmax - coverage_radius)
    pos[1] = np.clip(pos[1], world.ymin + coverage_radius, world.ymax - coverage_radius)
    return pos
