from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sim import Robot, Simulator, World, load_default_scene

DATA_ROOT = Path(__file__).resolve().parents[1] / "docs" / "data"
DEFAULT_DT = 0.05
DEFAULT_THRESHOLD = 0.65
DEFAULT_MAX_TIME = 600.0
DEFAULT_COVERAGE_RADIUS = 220.0
DEFAULT_SPAWN_INTERVAL = 4.0
HOLD_DURATION = 2.0
GRID_RESOLUTION = 20.0
MIN_RUNTIME_BEFORE_CHECK = 5.0
MOTION_SPEED_THRESHOLD = 3.0
MOTION_HOLD_DURATION = 6.0
BASE_RADIUS = 260.0


@dataclass
class CoverageLogEntry:
    time_s: float
    coverage_fraction: float
    num_robots: int


def estimate_coverage_fraction(
    world: World, robots: List[Robot], resolution: float
) -> float:
    if not robots:
        return 0.0
    xs = np.arange(world.xmin, world.xmax + resolution, resolution)
    ys = np.arange(world.ymin, world.ymax + resolution, resolution)
    grid_x, grid_y = np.meshgrid(xs, ys)
    points = np.stack((grid_x.ravel(), grid_y.ravel()), axis=1)
    covered = np.zeros(points.shape[0], dtype=bool)
    for robot in robots:
        radii_sq = robot.coverage_radius**2
        deltas = points - robot.pos
        dist_sq = np.sum(deltas * deltas, axis=1)
        covered |= dist_sq <= radii_sq
    return float(np.mean(covered))


def spawn_position(world: World, idx: int, coverage_radius: float) -> np.ndarray:
    stage_depth = coverage_radius * 1.25
    x = world.xmin + stage_depth
    lane_height = coverage_radius * 1.05
    lanes = max(1, int((world.ymax - world.ymin) // lane_height))
    row = idx % lanes
    direction = -1 if row % 2 == 0 else 1
    steps = (row // 2) + 0.5
    center_y = 0.5 * (world.ymin + world.ymax)
    y = center_y + direction * steps * lane_height
    y = np.clip(y, world.ymin + coverage_radius, world.ymax - coverage_radius)
    return np.array([x, y], dtype=float)


def spawn_robot(sim: Simulator, idx: int, coverage_radius: float) -> None:
    radius = max(coverage_radius, BASE_RADIUS) if idx == 0 else coverage_radius
    sim.add_robot(spawn_position(sim.world, idx, coverage_radius), radius)


def ensure_data_dir() -> Path:
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    run_dir = DATA_ROOT / f"run-{datetime.now():%Y%m%d-%H%M%S}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def write_log(run_dir: Path, entries: List[CoverageLogEntry]) -> None:
    path = run_dir / "coverage_log.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["time_s", "coverage_fraction", "num_robots"])
        for entry in entries:
            writer.writerow(
                [
                    f"{entry.time_s:.3f}",
                    f"{entry.coverage_fraction:.4f}",
                    entry.num_robots,
                ]
            )


def save_summary(run_dir: Path, summary: dict) -> None:
    path = run_dir / "summary.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)


def plot_timeseries(
    run_dir: Path, entries: List[CoverageLogEntry], threshold: float
) -> None:
    times = [entry.time_s for entry in entries]
    coverage = [entry.coverage_fraction for entry in entries]
    robot_counts = [entry.num_robots for entry in entries]
    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax1.plot(times, coverage, color="tab:blue", label="Coverage fraction")
    ax1.axhline(
        threshold, color="tab:blue", linestyle="--", alpha=0.4, label="Threshold"
    )
    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel("Coverage fraction", color="tab:blue")
    ax1.set_ylim(0.0, 1.05)
    ax2 = ax1.twinx()
    ax2.step(times, robot_counts, color="tab:orange", where="post", label="# Robots")
    ax2.set_ylabel("Robots deployed", color="tab:orange")
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="lower right")
    fig.tight_layout()
    fig.savefig(run_dir / "coverage_plot.png", dpi=200)
    plt.close(fig)


def plot_final_map(run_dir: Path, sim: Simulator, base_center: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_title("Final coverage map")
    ax.set_xlim(sim.world.xmin, sim.world.xmax)
    ax.set_ylim(sim.world.ymin, sim.world.ymax)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [px]")
    ax.set_ylabel("y [px]")
    for obstacle in sim.obstacles:
        circle = plt.Circle(
            obstacle.c, obstacle.R, color="#cc4444", fill=False, linewidth=2
        )
        ax.add_patch(circle)
    base_circle = plt.Circle(
        base_center, BASE_RADIUS, color="green", alpha=0.12, linewidth=2
    )
    ax.add_patch(base_circle)
    ax.scatter([base_center[0]], [base_center[1]], color="green", marker="s", s=80)
    for robot in sim.robots:
        coverage_circle = plt.Circle(
            robot.pos,
            robot.coverage_radius,
            color="tab:blue",
            alpha=0.08,
            linewidth=1.0,
        )
        ax.add_patch(coverage_circle)
        body = plt.Circle(robot.pos, robot.r, color="tab:blue")
        ax.add_patch(body)
    rectangle = plt.Rectangle(
        (sim.world.xmin, sim.world.ymin),
        sim.world.xmax - sim.world.xmin,
        sim.world.ymax - sim.world.ymin,
        fill=False,
        linewidth=2,
        edgecolor="black",
    )
    ax.add_patch(rectangle)
    fig.tight_layout()
    fig.savefig(run_dir / "final_map.png", dpi=200)
    plt.close(fig)


def auto_run(
    threshold: float,
    coverage_radius: float,
    dt: float,
    spawn_interval: float,
    max_time: float,
    grid_resolution: float,
    max_robots: Optional[int],
) -> Tuple[List[CoverageLogEntry], dict, Simulator]:
    world, _, obstacles, targets = load_default_scene()
    base_center = np.array(
        [
            world.xmin + BASE_RADIUS * 0.5,
            0.5 * (world.ymin + world.ymax),
        ],
        dtype=float,
    )
    sim = Simulator(world, [], obstacles, targets)
    entries: List[CoverageLogEntry] = []
    sim_time = 0.0
    last_spawn_time = -spawn_interval
    coverage_fraction = 0.0
    coverage_reached_at: float | None = None
    prev_positions: Optional[List[np.ndarray]] = None
    motion_still_since: float | None = None
    settled_time: float | None = None
    while sim_time <= max_time:
        can_spawn = False
        if max_robots is None:
            can_spawn = coverage_fraction < threshold
        else:
            can_spawn = len(sim.robots) < max_robots
        if can_spawn and (sim_time - last_spawn_time) >= spawn_interval:
            spawn_robot(sim, len(sim.robots), coverage_radius)
            last_spawn_time = sim_time
        sim.step(dt)
        sim_time += dt
        coverage_fraction = estimate_coverage_fraction(
            sim.world, sim.robots, grid_resolution
        )
        entries.append(CoverageLogEntry(sim_time, coverage_fraction, len(sim.robots)))
        if coverage_fraction >= threshold and sim_time >= MIN_RUNTIME_BEFORE_CHECK:
            if coverage_reached_at is None:
                coverage_reached_at = sim_time

        avg_speed = None
        if prev_positions and len(prev_positions) == len(sim.robots) and sim.robots:
            disps = [
                float(np.linalg.norm(robot.pos - prev_positions[idx]))
                for idx, robot in enumerate(sim.robots)
            ]
            avg_speed = (sum(disps) / max(len(disps), 1)) / dt

        prev_positions = [np.array(robot.pos, dtype=float) for robot in sim.robots]

        settle_ready = (
            coverage_fraction >= threshold
            and sim_time >= MIN_RUNTIME_BEFORE_CHECK
        )

        if avg_speed is not None and settle_ready:
            if avg_speed < MOTION_SPEED_THRESHOLD:
                if motion_still_since is None:
                    motion_still_since = sim_time
                elif sim_time - motion_still_since >= MOTION_HOLD_DURATION:
                    settled_time = sim_time
                    break
            else:
                motion_still_since = None
    summary = {
        "threshold": threshold,
        "coverage_radius_px": coverage_radius,
        "grid_resolution_px": grid_resolution,
        "spawn_interval_s": spawn_interval,
        "dt_s": dt,
        "max_time_s": max_time,
        "robots_used": len(sim.robots),
        "coverage_time_s": coverage_reached_at,
        "final_coverage": entries[-1].coverage_fraction if entries else 0.0,
        "max_robots": max_robots,
        "settled_time_s": settled_time,
        "base_center": base_center.tolist(),
    }
    return entries, summary, sim


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Automatically deploy robots until coverage threshold is met."
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="Coverage fraction target (0-1)",
    )
    parser.add_argument(
        "--coverage-radius",
        type=float,
        default=DEFAULT_COVERAGE_RADIUS,
        help="Coverage radius per robot (px)",
    )
    parser.add_argument(
        "--dt", type=float, default=DEFAULT_DT, help="Simulation timestep (s)"
    )
    parser.add_argument(
        "--spawn-interval",
        type=float,
        default=DEFAULT_SPAWN_INTERVAL,
        help="Seconds between robot spawns",
    )
    parser.add_argument(
        "--max-time",
        type=float,
        default=DEFAULT_MAX_TIME,
        help="Maximum simulation time (s)",
    )
    parser.add_argument(
        "--resolution",
        type=float,
        default=GRID_RESOLUTION,
        help="Grid resolution for coverage estimation (px)",
    )
    parser.add_argument(
        "--max-robots",
        type=int,
        default=None,
        help="Maximum number of robots to spawn (default unlimited)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = ensure_data_dir()
    entries, summary, sim = auto_run(
        threshold=args.threshold,
        coverage_radius=args.coverage_radius,
        dt=args.dt,
        spawn_interval=args.spawn_interval,
        max_time=args.max_time,
        grid_resolution=args.resolution,
        max_robots=args.max_robots,
    )
    write_log(run_dir, entries)
    save_summary(run_dir, summary)
    plot_timeseries(run_dir, entries, args.threshold)
    plot_final_map(run_dir, sim, np.array(summary["base_center"]))
    coverage_time = summary["coverage_time_s"]
    settled_time = summary.get("settled_time_s")
    if coverage_time is None:
        print("Coverage threshold not achieved before settlement/timeout.")
    else:
        print(
            f"Coverage threshold reached in {coverage_time:.2f} s using {summary['robots_used']} robots."
        )
    if settled_time is not None:
        print(f"Swarm motion settled after {settled_time:.2f} s.")
    else:
        print("Swarm did not settle before max-time limit.")
    print(f"Data saved to: {run_dir}")


if __name__ == "__main__":
    main()
