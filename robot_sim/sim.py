from __future__ import annotations

import json
from collections import deque
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import controls
import numpy as np
import pygame as pg


@dataclass
class World:
    xmin: float
    xmax: float
    ymin: float
    ymax: float


@dataclass
class Robot:
    pos: np.ndarray
    r: float = 14.0
    vmax: float = 150.0
    color: Tuple[int, int, int] = (30, 120, 255)
    coverage_radius: float = 180.0
    trail: deque = field(default_factory=lambda: deque(maxlen=160))


@dataclass
class Obstacle:
    c: np.ndarray
    R: float
    color: Tuple[int, int, int] = (210, 70, 70)


@dataclass
class Target:
    pos: np.ndarray
    color: Tuple[int, int, int] = (40, 170, 90)


@dataclass
class DragState:
    kind: str
    index: int
    grab_offset: np.ndarray


DEFAULT_SCENE_PATH = Path(__file__).with_name("scene.json")
WORLD_OUTLINE_COLOR = (25, 25, 25)
HIGHLIGHT_COLOR = (30, 30, 30)
TRAIL_COLOR = (180, 180, 180)


def _np_point(values: List[float]) -> np.ndarray:
    return np.asarray(values, dtype=float)


def load_scene(path: Path) -> Tuple[World, List[Robot], List[Obstacle], List[Target]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    world_data = data.get("world", {})
    world = World(
        float(world_data.get("xmin", 0.0)),
        float(world_data.get("xmax", 1000.0)),
        float(world_data.get("ymin", 0.0)),
        float(world_data.get("ymax", 700.0)),
    )
    robots = [
        Robot(
            pos=_np_point(entry.get("pos", [0.0, 0.0])),
            r=float(entry.get("r", 14.0)),
            vmax=float(entry.get("vmax", 150.0)),
            coverage_radius=float(entry.get("coverage_radius", 180.0)),
        )
        for entry in data.get("robots", [])
    ]
    obstacles = [
        Obstacle(c=_np_point(entry.get("c", [0.0, 0.0])), R=float(entry.get("R", 60.0)))
        for entry in data.get("obstacles", [])
    ]
    targets = [
        Target(pos=_np_point(entry.get("pos", [0.0, 0.0])))
        for entry in data.get("targets", [])
    ]
    return world, robots, obstacles, targets


def save_scene(
    path: Path,
    world: World,
    robots: List[Robot],
    obstacles: List[Obstacle],
    targets: List[Target],
) -> None:
    payload = {
        "world": {
            "xmin": world.xmin,
            "xmax": world.xmax,
            "ymin": world.ymin,
            "ymax": world.ymax,
        },
        "robots": [
            {
                "pos": robot.pos.tolist(),
                "r": robot.r,
                "vmax": robot.vmax,
                "coverage_radius": robot.coverage_radius,
            }
            for robot in robots
        ],
        "obstacles": [
            {"c": obstacle.c.tolist(), "R": obstacle.R} for obstacle in obstacles
        ],
        "targets": [{"pos": target.pos.tolist()} for target in targets],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_default_scene() -> Tuple[World, List[Robot], List[Obstacle], List[Target]]:
    if DEFAULT_SCENE_PATH.exists():
        return load_scene(DEFAULT_SCENE_PATH)
    # fallback simple layout if file missing
    world = World(40.0, 1840.0, 40.0, 1240.0)
    robots = [
        Robot(pos=_np_point([200.0, 260.0]), coverage_radius=140.0),
        Robot(pos=_np_point([240.0, 620.0]), coverage_radius=140.0),
        Robot(pos=_np_point([200.0, 980.0]), coverage_radius=140.0),
    ]
    obstacles = [
        Obstacle(c=_np_point([640.0, 360.0]), R=120.0),
        Obstacle(c=_np_point([980.0, 320.0]), R=80.0),
        Obstacle(c=_np_point([1340.0, 260.0]), R=110.0),
        Obstacle(c=_np_point([520.0, 760.0]), R=90.0),
        Obstacle(c=_np_point([960.0, 780.0]), R=130.0),
        Obstacle(c=_np_point([1380.0, 900.0]), R=70.0),
        Obstacle(c=_np_point([720.0, 1100.0]), R=120.0),
        Obstacle(c=_np_point([1150.0, 1080.0]), R=85.0),
        Obstacle(c=_np_point([1520.0, 640.0]), R=140.0),
    ]
    targets: List[Target] = []
    return world, robots, obstacles, targets


class Simulator:
    def __init__(
        self,
        world: World,
        robots: List[Robot],
        obstacles: List[Obstacle],
        targets: List[Target],
        scene_path: Optional[Path] = None,
    ) -> None:
        self.scene_path = Path(scene_path) if scene_path else None
        self.world: World = deepcopy(world)
        self.robots: List[Robot] = [self._clone_robot(robot) for robot in robots]
        self.obstacles: List[Obstacle] = [
            self._clone_obstacle(obstacle) for obstacle in obstacles
        ]
        self.targets: List[Target] = [self._clone_target(target) for target in targets]
        self.drag_state: Optional[DragState] = None
        self.selection: Optional[Tuple[str, int]] = None
        self._capture_snapshot()
        self._stamp_trails()

    def _clone_robot(self, robot: Robot) -> Robot:
        clone = Robot(
            pos=np.array(robot.pos, dtype=float),
            r=float(robot.r),
            vmax=float(robot.vmax),
            color=robot.color,
            coverage_radius=float(robot.coverage_radius),
        )
        maxlen = robot.trail.maxlen or 160
        clone.trail = deque(robot.trail, maxlen=maxlen)
        return clone

    def _clone_obstacle(self, obstacle: Obstacle) -> Obstacle:
        return Obstacle(
            c=np.array(obstacle.c, dtype=float),
            R=float(obstacle.R),
            color=obstacle.color,
        )

    def _clone_target(self, target: Target) -> Target:
        return Target(pos=np.array(target.pos, dtype=float), color=target.color)

    def reset(self) -> None:
        if self.scene_path and self.scene_path.exists():
            world, robots, obstacles, targets = load_scene(self.scene_path)
        else:
            world, robots, obstacles, targets = deepcopy(self._initial_snapshot)
        self.world = deepcopy(world)
        self.robots = [self._clone_robot(robot) for robot in robots]
        self.obstacles = [self._clone_obstacle(obstacle) for obstacle in obstacles]
        self.targets = [self._clone_target(target) for target in targets]
        self.drag_state = None
        self.selection = None
        self._capture_snapshot()
        self._stamp_trails()

    def _capture_snapshot(self) -> None:
        self._initial_snapshot = (
            deepcopy(self.world),
            [self._clone_robot(robot) for robot in self.robots],
            [self._clone_obstacle(ob) for ob in self.obstacles],
            [self._clone_target(tg) for tg in self.targets],
        )

    def _stamp_trails(self) -> None:
        for robot in self.robots:
            robot.trail.clear()
            robot.trail.append(tuple(robot.pos))

    def step(self, dt: float) -> None:
        dragging_robot = None
        if self.drag_state and self.drag_state.kind == "robot":
            dragging_robot = self.drag_state.index
        velocities: List[np.ndarray] = []
        for idx, robot in enumerate(self.robots):
            if dragging_robot is not None and idx == dragging_robot:
                velocities.append(np.zeros(2))
                continue
            velocity = controls.compute_velocity(
                robot, idx, self.robots, self.world, self.obstacles, self.targets
            )
            limited = controls.clip_speed(velocity, robot.vmax)
            velocities.append(limited)
        for idx, robot in enumerate(self.robots):
            if dragging_robot is not None and idx == dragging_robot:
                continue
            robot.pos = self._clamp_position(robot.pos + velocities[idx] * dt, robot.r)
            robot.trail.append(tuple(robot.pos))

    def _clamp_position(self, pos: np.ndarray, radius: float) -> np.ndarray:
        x = np.clip(pos[0], self.world.xmin + radius, self.world.xmax - radius)
        y = np.clip(pos[1], self.world.ymin + radius, self.world.ymax - radius)
        return np.array([x, y])

    def handle_event(self, event: pg.event.Event) -> None:
        if event.type == pg.MOUSEBUTTONDOWN and event.button == 1:
            self._begin_drag(np.array(event.pos, dtype=float))
        elif event.type == pg.MOUSEBUTTONDOWN and event.button == 3:
            self.selection = self._pick(np.array(event.pos, dtype=float))
        elif event.type == pg.MOUSEBUTTONUP and event.button == 1:
            self.drag_state = None
        elif event.type == pg.MOUSEMOTION and self.drag_state:
            self._continue_drag(np.array(event.pos, dtype=float))

    def _begin_drag(self, pointer: np.ndarray) -> None:
        pick = self._pick(pointer)
        if pick is None:
            self.drag_state = None
            return
        kind, idx = pick
        entity_pos = self._entity_position(kind, idx)
        self.drag_state = DragState(
            kind=kind, index=idx, grab_offset=pointer - entity_pos
        )
        self.selection = pick

    def _continue_drag(self, pointer: np.ndarray) -> None:
        if not self.drag_state:
            return
        target = pointer - self.drag_state.grab_offset
        kind = self.drag_state.kind
        idx = self.drag_state.index
        if kind == "robot":
            self.robots[idx].pos = self._clamp_position(target, self.robots[idx].r)
            self.robots[idx].trail.append(tuple(self.robots[idx].pos))
        elif kind == "target":
            self.targets[idx].pos = self._clamp_position(target, 4.0)
        elif kind == "obstacle":
            radius = self.obstacles[idx].R
            self.obstacles[idx].c = self._clamp_position(target, radius)

    def _entity_position(self, kind: str, idx: int) -> np.ndarray:
        if kind == "robot":
            return self.robots[idx].pos
        if kind == "target":
            return self.targets[idx].pos
        return self.obstacles[idx].c

    def _pick(self, pointer: np.ndarray) -> Optional[Tuple[str, int]]:
        for idx, robot in enumerate(self.robots):
            if np.linalg.norm(pointer - robot.pos) <= robot.r:
                return ("robot", idx)
        for idx, target in enumerate(self.targets):
            if np.linalg.norm(pointer - target.pos) <= 10.0:
                return ("target", idx)
        for idx, obstacle in enumerate(self.obstacles):
            if np.linalg.norm(pointer - obstacle.c) <= obstacle.R:
                return ("obstacle", idx)
        return None

    def draw(self, surface: pg.Surface) -> None:
        world_rect = pg.Rect(
            int(self.world.xmin),
            int(self.world.ymin),
            int(self.world.xmax - self.world.xmin),
            int(self.world.ymax - self.world.ymin),
        )
        pg.draw.rect(surface, WORLD_OUTLINE_COLOR, world_rect, 2)
        for obstacle in self.obstacles:
            center = obstacle.c.astype(int)
            pg.draw.circle(surface, obstacle.color, center, int(obstacle.R), 2)
        for target in self.targets:
            center = target.pos.astype(int)
            pg.draw.circle(surface, target.color, center, 6, 2)
            pg.draw.line(
                surface,
                target.color,
                (center[0] - 6, center[1]),
                (center[0] + 6, center[1]),
                2,
            )
            pg.draw.line(
                surface,
                target.color,
                (center[0], center[1] - 6),
                (center[0], center[1] + 6),
                2,
            )
        for robot in self.robots:
            if len(robot.trail) > 1:
                points = [tuple(map(int, point)) for point in robot.trail]
                pg.draw.lines(surface, TRAIL_COLOR, False, points, 2)
            center = robot.pos.astype(int)
            pg.draw.circle(surface, robot.color, center, int(robot.r))
        if self.selection:
            kind, idx = self.selection
            entity_pos = self._entity_position(kind, idx)
            pg.draw.circle(surface, HIGHLIGHT_COLOR, entity_pos.astype(int), 6, 1)

    def add_robot(
        self, position: np.ndarray, coverage_radius: Optional[float] = None
    ) -> None:
        robot = Robot(
            pos=self._clamp_position(position, 14.0),
            coverage_radius=float(coverage_radius) if coverage_radius else 180.0,
        )
        robot.trail.append(tuple(robot.pos))
        self.robots.append(robot)
        self.selection = ("robot", len(self.robots) - 1)

    def add_target(self, position: np.ndarray) -> None:
        target = Target(pos=self._clamp_position(position, 4.0))
        self.targets.append(target)
        self.selection = ("target", len(self.targets) - 1)

    def add_obstacle(self, position: np.ndarray, radius: float = 60.0) -> None:
        obstacle = Obstacle(c=self._clamp_position(position, radius), R=radius)
        self.obstacles.append(obstacle)
        self.selection = ("obstacle", len(self.obstacles) - 1)

    def delete_selection(self) -> None:
        if not self.selection:
            return
        kind, idx = self.selection
        if kind == "robot" and 0 <= idx < len(self.robots):
            self.robots.pop(idx)
        elif kind == "target" and 0 <= idx < len(self.targets):
            self.targets.pop(idx)
        elif kind == "obstacle" and 0 <= idx < len(self.obstacles):
            self.obstacles.pop(idx)
        self.selection = None

    def selection_summary(self) -> str:
        if not self.selection:
            return ""
        kind, idx = self.selection
        return f"{kind} #{idx + 1}"

    def pose_rows(self) -> List[Tuple[int, float, float]]:
        return [
            (idx, float(robot.pos[0]), float(robot.pos[1]))
            for idx, robot in enumerate(self.robots)
        ]

    def save_scene(self, path: Path) -> None:
        save_scene(path, self.world, self.robots, self.obstacles, self.targets)

    def set_robot_count(self, count: int) -> None:
        count = max(0, int(count))
        if count < len(self.robots):
            del self.robots[count:]
        while len(self.robots) < count:
            self._spawn_robot(len(self.robots))
        self.drag_state = None
        self._ensure_selection_valid()

    def set_target_count(self, count: int) -> None:
        count = max(0, int(count))
        if count < len(self.targets):
            del self.targets[count:]
        while len(self.targets) < count:
            self._spawn_target(len(self.targets))
        if self.drag_state and self.drag_state.kind == "target":
            self.drag_state = None
        self._ensure_selection_valid()

    def _spawn_robot(self, idx: int) -> None:
        position = self._default_robot_position(idx)
        robot = Robot(pos=self._clamp_position(position, 14.0))
        robot.trail.append(tuple(robot.pos))
        self.robots.append(robot)

    def _spawn_target(self, idx: int) -> None:
        position = self._default_target_position(idx)
        target = Target(pos=self._clamp_position(position, 4.0))
        self.targets.append(target)

    def _default_robot_position(self, idx: int) -> np.ndarray:
        width = self.world.xmax - self.world.xmin
        height = self.world.ymax - self.world.ymin
        cols = 3
        spacing_x = max(60.0, (width * 0.4) / max(1, cols - 1))
        spacing_y = 60.0
        start = np.array(
            [self.world.xmin + 0.15 * width, self.world.ymin + 0.2 * height]
        )
        offset = np.array([(idx % cols) * spacing_x, (idx // cols) * spacing_y])
        return start + offset

    def _default_target_position(self, idx: int) -> np.ndarray:
        width = self.world.xmax - self.world.xmin
        height = self.world.ymax - self.world.ymin
        cols = 2
        spacing_x = max(40.0, (width * 0.2) / max(1, cols - 1))
        spacing_y = 70.0
        start = np.array(
            [self.world.xmin + 0.65 * width, self.world.ymin + 0.25 * height]
        )
        offset = np.array(
            [(idx % cols) * spacing_x, (idx // cols) * spacing_y], dtype=float
        )
        return start + offset

    def _ensure_selection_valid(self) -> None:
        if not self.selection:
            return
        kind, idx = self.selection
        if kind == "robot" and idx >= len(self.robots):
            self.selection = None
        elif kind == "target" and idx >= len(self.targets):
            self.selection = None
        elif kind == "obstacle" and idx >= len(self.obstacles):
            self.selection = None
