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
from sim import World, coverage_box, density_grid, load_world, spawn_position

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


def render_frame(
    ax: plt.Axes,
    world: World,
    box: Tuple[float, float, float, float],
    base_pos: np.ndarray,
    positions: np.ndarray,
    coverage_radius: float,
) -> None:
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(world.xmin, world.xmax)
    ax.set_ylim(world.ymin, world.ymax)
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
    box: tuple[float, float, float, float],
    positions: np.ndarray,
    coverage_radius: float,
    base_pos: np.ndarray,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.set_title("Final coverage state (JAX core)")
    render_frame(ax, world, box, base_pos, positions, coverage_radius)
    xmin, xmax, ymin, ymax = box
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(run_dir / "final_map.png", dpi=200)
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
    box: Tuple[float, float, float, float],
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
            render_frame(ax, world, box, base_pos, positions, coverage_radius)
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


def auto_run(
    args: argparse.Namespace,
) -> tuple[List[CoverageLogEntry], dict, np.ndarray, List[np.ndarray], World, Tuple[float, float, float, float], np.ndarray]:
    world = load_world()
    box = coverage_box(world, BOX_WIDTH, BOX_HEIGHT)
    base_pos = base_position(world)
    grid = density_grid(box, args.resolution)
    spacing = max(args.coverage_radius * 0.8, args.resolution)
    bridge = bridge_points(base_pos, box, spacing)
    if bridge.size:
        grid = np.vstack([grid, bridge])
    params = controls.SimParams(
        coverage_grid=jnp.array(grid, dtype=jnp.float32),
        world_min=jnp.array([world.xmin, world.ymin], dtype=jnp.float32),
        world_max=jnp.array([world.xmax, world.ymax], dtype=jnp.float32),
        base_position=jnp.array(base_pos, dtype=jnp.float32),
        coverage_radius=float(args.coverage_radius),
        body_radius=float(args.coverage_radius) * 0.08,
        dt=float(args.dt),
        coverage_gain=1.15,
        link_gain=0.85,
        robot_speed=float(args.robot_speed),
        desired_gap=float(args.coverage_radius) * 0.95,
        max_gap=float(args.coverage_radius) * 1.9,
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
    frame_states: List[np.ndarray] = []
    frame_interval = 1.0 / max(args.video_fps, 1e-6) if args.video_fps > 0.0 else None
    next_frame_time = 0.0
    if frame_interval is not None:
        frame_states.append(np.array(jax.device_get(state["positions"])))
        next_frame_time = frame_interval
    while sim_time <= args.max_time:
        if deploy_count < args.max_robots and (sim_time - last_spawn) >= args.spawn_interval:
            slot = deploy_count + 1
            spawn_idx = deploy_count
            spawn_pos = spawn_position(world, base_pos, spawn_idx, args.coverage_radius)
            state = {
                "positions": state["positions"].at[slot].set(jnp.array(spawn_pos, dtype=jnp.float32)),
                "active": state["active"].at[slot].set(True),
            }
            deploy_count += 1
            last_spawn = sim_time
        state = controls.step_state(state, params)
        total_cov, per_robot_frac = controls.coverage_metrics(state, params)
        total_cov = float(jax.device_get(total_cov))
        per_robot = np.array(jax.device_get(per_robot_frac))
        entries.append(CoverageLogEntry(sim_time, total_cov, deploy_count))
        if coverage_reached is None and total_cov >= args.threshold:
            coverage_reached = sim_time
        if frame_interval is not None and sim_time >= next_frame_time:
            frame_states.append(np.array(jax.device_get(state["positions"])))
            next_frame_time += frame_interval
        sim_time += args.dt
    final_positions = np.array(jax.device_get(state["positions"]))
    world_area = (world.xmax - world.xmin) * (world.ymax - world.ymin)
    box_area = (box[1] - box[0]) * (box[3] - box[2])
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
        "coverage_box": {
            "xmin": box[0],
            "xmax": box[1],
            "ymin": box[2],
            "ymax": box[3],
            "area_px2": box_area,
        },
        "per_robot_area_fraction": per_robot.tolist(),
        "per_robot_area_px2": (per_robot * box_area).tolist(),
        "final_positions": final_positions.tolist(),
        "world_area_px2": world_area,
    }
    return entries, summary, final_positions, frame_states, world, box, base_pos


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
    parser.add_argument("--video-fps", type=float, default=DEFAULT_VIDEO_FPS)
    parser.add_argument(
        "--video-path",
        type=str,
        default="",
        help="Optional mp4 output path (default: run directory)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = ensure_run_dir()
    entries, summary, final_positions, frames, world, box, base_pos = auto_run(args)
    write_log(run_dir, entries)
    save_summary(run_dir, summary)
    plot_final(run_dir, world, box, final_positions, args.coverage_radius, base_pos)
    video_path = Path(args.video_path) if args.video_path else run_dir / "coverage.mp4"
    if args.video_fps > 0.0:
        write_video(run_dir, world, box, base_pos, frames, args.coverage_radius, args.video_fps, video_path)
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
