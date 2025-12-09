from __future__ import annotations

import argparse
import csv
import json
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import controls
from sim import (
    SCENE_PATH,
    World,
    cluster_spawn_position,
    coverage_box,
    density_grid,
    load_scene,
    sequential_spawn_position,
)

DATA_ROOT = Path(__file__).resolve().parents[1] / "docs" / "data"
DEFAULT_DT = 0.05
DEFAULT_THRESHOLD = 0.6
DEFAULT_MAX_TIME = 400.0
DEFAULT_COVERAGE_RADIUS = 200.0
DEFAULT_SPAWN_INTERVAL = 6.0
DEFAULT_MAX_ROBOTS = 8
DEFAULT_ROBOT_SPEED = 220.0
DEFAULT_VIDEO_FPS = 12.0
BOX_WIDTH = 960.0
BOX_HEIGHT = 680.0
GRID_RESOLUTION = 25.0
BASE_RADIUS = 260.0
SPAWN_TRANSITION_STEPS = 4
REDUNDANCY_GAIN = 1.0
OBSTACLE_MARGIN = 140.0
OBSTACLE_GAIN = 340000.0
OBSTACLE_SOFT = 25.0


@dataclass
class CoverageLogEntry:
    time_s: float
    coverage_fraction: float
    active_robots: int


def ensure_run_dir() -> Path:
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    run_dir = DATA_ROOT / f"run-{datetime.now():%Y%m%d-%H%M%S}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def base_position(world: World) -> np.ndarray:
    center_y = 0.5 * (world.ymin + world.ymax)
    return np.array([world.xmin + 0.5 * BASE_RADIUS, center_y], dtype=np.float32)


def bridge_points(
    base_pos: np.ndarray,
    box: Tuple[float, float, float, float],
    spacing: float,
    lanes: int = 3,
) -> np.ndarray:
    xmin, _, ymin, ymax = box
    target_x = xmin
    xs = np.arange(base_pos[0] + spacing, target_x, spacing)
    if xs.size == 0:
        return np.empty((0, 2), dtype=np.float32)
    center_y = 0.5 * (ymin + ymax)
    lane_offsets = (
        (np.arange(lanes, dtype=float) - 0.5 * (lanes - 1)) * spacing * 0.35
    )
    points = []
    for x in xs:
        for offset in lane_offsets:
            points.append([x, center_y + offset])
    return np.array(points, dtype=np.float32)


def resolve_scene_path(scene_arg: str | None) -> Path:
    if not scene_arg:
        return SCENE_PATH
    script_dir = Path(__file__).resolve().parent
    candidate = Path(scene_arg).expanduser()
    search: List[Path] = [candidate]
    if not candidate.is_absolute():
        search.append(script_dir / candidate)
        search.append(script_dir / Path(scene_arg).name)
    for path in search:
        if path.exists():
            return path
    return candidate


def render_frame(
    ax: plt.Axes,
    world: World,
    boxes: List[Tuple[float, float, float, float]],
    base_pos: np.ndarray,
    positions: np.ndarray,
    coverage_radius: float,
) -> None:
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(world.xmin, world.xmax)
    ax.set_ylim(world.ymin, world.ymax)
    for box in boxes:
        xmin, xmax, ymin, ymax = box
        ax.add_patch(
            plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, color="#7fcdbb", alpha=0.2, linewidth=1.8)
        )
    ax.add_patch(
        plt.Rectangle((world.xmin, world.ymin), world.xmax - world.xmin, world.ymax - world.ymin, fill=False)
    )
    base_circle = plt.Circle(base_pos, BASE_RADIUS, color="green", alpha=0.1, linewidth=1.5)
    ax.add_patch(base_circle)
    ax.scatter([base_pos[0]], [base_pos[1]], color="green", marker="s", s=80, label="Base")
    for idx in range(1, positions.shape[0]):
        ax.add_patch(
            plt.Circle(positions[idx], coverage_radius, color="tab:blue", alpha=0.05)
        )
        ax.scatter([positions[idx, 0]], [positions[idx, 1]], color="tab:blue", s=20)
        ax.text(positions[idx, 0] + 8, positions[idx, 1] + 8, f"R{idx}", fontsize=7)


def plot_final(
    run_dir: Path,
    world: World,
    boxes: List[tuple[float, float, float, float]],
    positions: np.ndarray,
    coverage_radius: float,
    base_pos: np.ndarray,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.set_title("Final coverage state")
    render_frame(ax, world, boxes, base_pos, positions, coverage_radius)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(run_dir / "final_map.png", dpi=200)
    plt.close(fig)


def plot_coverage_history(run_dir: Path, entries: List[CoverageLogEntry]) -> None:
    if not entries:
        return
    times = np.array([entry.time_s for entry in entries], dtype=float)
    coverage = np.array([entry.coverage_fraction for entry in entries], dtype=float)
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    ax.plot(times, coverage, color="tab:blue", linewidth=2.0)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Coverage fraction")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.25)
    ax.set_title("Coverage vs time")
    fig.tight_layout()
    fig.savefig(run_dir / "coverage_history.png", dpi=200)
    plt.close(fig)


def write_log(run_dir: Path, entries: List[CoverageLogEntry]) -> None:
    path = run_dir / "coverage_log.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["time_s", "coverage_fraction", "active_robots"])
        for entry in entries:
            writer.writerow([f"{entry.time_s:.2f}", f"{entry.coverage_fraction:.4f}", entry.active_robots])


def save_summary(run_dir: Path, payload: dict) -> None:
    (run_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_video(
    run_dir: Path,
    world: World,
    boxes: List[Tuple[float, float, float, float]],
    base_pos: np.ndarray,
    frames: List[np.ndarray],
    coverage_radius: float,
    fps: float,
    path: Path,
) -> None:
    if not frames or fps <= 0.0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tempdir:
        tempdir_path = Path(tempdir)
        for idx, positions in enumerate(frames):
            fig, ax = plt.subplots(figsize=(9, 5.5))
            ax.set_title("Swarm deployment")
            render_frame(ax, world, boxes, base_pos, positions, coverage_radius)
            ax.set_xticks([])
            ax.set_yticks([])
            fig.tight_layout()
            frame_path = tempdir_path / f"frame_{idx:05d}.png"
            fig.savefig(frame_path, dpi=144)
            plt.close(fig)
        cmd = [
            "ffmpeg",
            "-y",
            "-framerate",
            f"{fps}",
            "-i",
            str(tempdir_path / "frame_%05d.png"),
            "-pix_fmt",
            "yuv420p",
            str(path),
        ]
        subprocess.run(cmd, check=True)


def add_spawn_transition(
    frame_states: List[np.ndarray],
    previous: np.ndarray,
    updated: np.ndarray,
) -> None:
    for step in range(1, SPAWN_TRANSITION_STEPS + 1):
        alpha = step / float(SPAWN_TRANSITION_STEPS + 1)
        interp = previous + (updated - previous) * alpha
        frame_states.append(interp.copy())


def auto_run(
    args: argparse.Namespace,
) -> tuple[
    List[CoverageLogEntry],
    dict,
    np.ndarray,
    List[np.ndarray],
    World,
    List[Tuple[float, float, float, float]],
    np.ndarray,
]:
    scene_path = resolve_scene_path(args.scene)
    scene = load_scene(scene_path)
    world = scene.world
    boxes = scene.goal_boxes or [coverage_box(world, BOX_WIDTH, BOX_HEIGHT)]
    base_pos = base_position(world)
    centers = np.array([
        [0.5 * (box[0] + box[1]), 0.5 * (box[2] + box[3])] for box in boxes
    ])
    coverage_center = np.mean(centers, axis=0).astype(np.float32)
    box_min = np.array([min(box[0] for box in boxes), min(box[2] for box in boxes)], dtype=np.float32)
    box_max = np.array([max(box[1] for box in boxes), max(box[3] for box in boxes)], dtype=np.float32)
    grid_chunks = [density_grid(box, args.resolution) for box in boxes]
    grid = np.vstack(grid_chunks)
    spacing = max(args.coverage_radius * 0.8, args.resolution)
    bridge = bridge_points(base_pos, boxes[0], spacing)
    if bridge.size:
        grid = np.vstack([grid, bridge])
    obstacles = scene.obstacles
    if obstacles.size:
        obstacle_centers = jnp.array(obstacles[:, :2], dtype=jnp.float32)
        obstacle_radii = jnp.array(obstacles[:, 2], dtype=jnp.float32)
    else:
        obstacle_centers = jnp.zeros((0, 2), dtype=jnp.float32)
        obstacle_radii = jnp.zeros((0,), dtype=jnp.float32)
    params = controls.SimParams(
        coverage_grid=jnp.array(grid, dtype=jnp.float32),
        world_min=jnp.array([world.xmin, world.ymin], dtype=jnp.float32),
        world_max=jnp.array([world.xmax, world.ymax], dtype=jnp.float32),
        base_position=jnp.array(base_pos, dtype=jnp.float32),
        box_min=jnp.array(box_min, dtype=jnp.float32),
        box_max=jnp.array(box_max, dtype=jnp.float32),
        obstacle_centers=obstacle_centers,
        obstacle_radii=obstacle_radii,
        coverage_radius=float(args.coverage_radius),
        body_radius=float(args.coverage_radius) * 0.08,
        dt=float(args.dt),
        coverage_gain=1.15,
        link_gain=0.85,
        robot_speed=float(args.robot_speed),
        desired_gap=float(args.coverage_radius) * 0.95,
        max_gap=float(args.coverage_radius) * 1.9,
        box_gain=0.9,
        redundancy_gain=REDUNDANCY_GAIN,
        obstacle_margin=OBSTACLE_MARGIN,
        obstacle_gain=OBSTACLE_GAIN,
        obstacle_soft=OBSTACLE_SOFT,
    )
    total_slots = args.max_robots + 1
    state = {
        "positions": jnp.tile(params.base_position, (total_slots, 1)),
        "active": jnp.zeros((total_slots,), dtype=bool).at[0].set(True),
    }
    entries: List[CoverageLogEntry] = []
    deploy_count = 0
    last_spawn = -args.spawn_interval
    sim_time = 0.0
    coverage_reached = None
    per_robot = np.zeros(total_slots)
    if args.spawn_all:
        for spawn_idx in range(args.max_robots):
            slot = spawn_idx + 1
            spawn_pos = cluster_spawn_position(
                world,
                base_pos,
                spawn_idx,
                args.coverage_radius,
                BASE_RADIUS,
                coverage_center,
            )
            state = {
                "positions": state["positions"].at[slot].set(jnp.array(spawn_pos, dtype=jnp.float32)),
                "active": state["active"].at[slot].set(True),
            }
        deploy_count = args.max_robots
    frame_states: List[np.ndarray] = []
    frame_interval = 1.0 / max(args.video_fps, 1e-6) if args.video_fps > 0.0 else None
    next_frame_time = frame_interval if frame_interval is not None else None
    if frame_interval is not None:
        frame_states.append(np.array(jax.device_get(state["positions"]), copy=True))
    while sim_time <= args.max_time:
        if (
            not args.spawn_all
            and deploy_count < args.max_robots
            and (sim_time - last_spawn) >= args.spawn_interval
        ):
            prev_positions_np = None
            if frame_interval is not None:
                prev_positions_np = np.array(jax.device_get(state["positions"]), copy=True)
            slot = deploy_count + 1
            spawn_idx = deploy_count
            spawn_pos = sequential_spawn_position(
                world, base_pos, spawn_idx, args.coverage_radius
            )
            state = {
                "positions": state["positions"].at[slot].set(jnp.array(spawn_pos, dtype=jnp.float32)),
                "active": state["active"].at[slot].set(True),
            }
            if prev_positions_np is not None:
                updated_positions = prev_positions_np.copy()
                updated_positions[slot] = spawn_pos
                add_spawn_transition(frame_states, prev_positions_np, updated_positions)
            deploy_count += 1
            last_spawn = sim_time
        state = controls.step_state(state, params)
        total_cov, per_robot_frac = controls.coverage_metrics(state, params)
        total_cov = float(jax.device_get(total_cov))
        per_robot = np.array(jax.device_get(per_robot_frac))
        entries.append(CoverageLogEntry(sim_time, total_cov, deploy_count))
        if coverage_reached is None and total_cov >= args.threshold:
            coverage_reached = sim_time
        if frame_interval is not None and next_frame_time is not None:
            while sim_time + 1e-9 >= next_frame_time:
                frame_states.append(np.array(jax.device_get(state["positions"]), copy=True))
                next_frame_time += frame_interval
        sim_time += args.dt
    final_positions = np.array(jax.device_get(state["positions"]))
    if frame_interval is not None:
        frame_states.append(final_positions.copy())
    world_area = (world.xmax - world.xmin) * (world.ymax - world.ymin)
    box_area = sum((b[1] - b[0]) * (b[3] - b[2]) for b in boxes)
    summary = {
        "threshold": args.threshold,
        "coverage_radius_px": args.coverage_radius,
        "grid_resolution_px": args.resolution,
        "spawn_interval_s": args.spawn_interval,
        "dt_s": args.dt,
        "max_time_s": args.max_time,
        "robots_used": deploy_count,
        "coverage_time_s": coverage_reached,
        "final_coverage": entries[-1].coverage_fraction if entries else 0.0,
        "coverage_boxes": [
            {
                "xmin": b[0],
                "xmax": b[1],
                "ymin": b[2],
                "ymax": b[3],
                "area_px2": (b[1] - b[0]) * (b[3] - b[2]),
            }
            for b in boxes
        ],
        "per_robot_area_fraction": per_robot.tolist(),
        "per_robot_area_px2": (per_robot * box_area).tolist(),
        "final_positions": final_positions.tolist(),
        "world_area_px2": world_area,
        "scene_path": str(scene_path),
    }
    return entries, summary, final_positions, frame_states, world, boxes, base_pos


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="JAX-based multi-robot coverage simulator")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--coverage-radius", type=float, default=DEFAULT_COVERAGE_RADIUS)
    parser.add_argument("--dt", type=float, default=DEFAULT_DT)
    parser.add_argument("--spawn-interval", type=float, default=DEFAULT_SPAWN_INTERVAL)
    parser.add_argument("--max-time", type=float, default=DEFAULT_MAX_TIME)
    parser.add_argument("--resolution", type=float, default=GRID_RESOLUTION)
    parser.add_argument("--max-robots", type=int, default=DEFAULT_MAX_ROBOTS)
    parser.add_argument("--robot-speed", type=float, default=DEFAULT_ROBOT_SPEED)
    parser.add_argument(
        "--spawn-all",
        action="store_true",
        help="Spawn all robots at t=0 instead of sequential deployment",
    )
    parser.add_argument("--video-fps", type=float, default=DEFAULT_VIDEO_FPS)
    parser.add_argument(
        "--video-path",
        type=str,
        default="",
        help="Optional mp4 output path (default: run directory)",
    )
    parser.add_argument(
        "--scene",
        type=str,
        default=str(SCENE_PATH),
        help="Path to scene JSON with world/obstacle/goal definitions",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = ensure_run_dir()
    entries, summary, final_positions, frames, world, boxes, base_pos = auto_run(args)
    write_log(run_dir, entries)
    save_summary(run_dir, summary)
    plot_final(run_dir, world, boxes, final_positions, args.coverage_radius, base_pos)
    plot_coverage_history(run_dir, entries)
    video_path = Path(args.video_path) if args.video_path else run_dir / "coverage.mp4"
    if args.video_fps > 0.0:
        write_video(run_dir, world, boxes, base_pos, frames, args.coverage_radius, args.video_fps, video_path)
    if summary["coverage_time_s"] is None:
        print("Coverage threshold not reached within allotted time.")
    else:
        print(
            f"Coverage threshold reached in {summary['coverage_time_s']:.1f} s using {summary['robots_used']} robots."
        )
    per_robot = summary["per_robot_area_fraction"]
    if per_robot:
        print("Per-robot coverage fractions (including base slot):")
        for idx, frac in enumerate(per_robot):
            tag = "Base" if idx == 0 else f"R{idx}"
            print(f"  {tag:<5} {frac * 100:5.2f}% of box")
    print(f"Final coverage fraction: {summary['final_coverage']:.3f}")
    print(f"Data saved to: {run_dir}")
    print(f"Coverage map saved to: {run_dir / 'final_map.png'}")
    if args.video_fps > 0.0:
        print(f"Video saved to: {video_path}")


if __name__ == "__main__":
    main()
