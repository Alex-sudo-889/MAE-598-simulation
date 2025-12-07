from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp


class SimParams(NamedTuple):
    coverage_grid: jnp.ndarray
    world_min: jnp.ndarray
    world_max: jnp.ndarray
    base_position: jnp.ndarray
    coverage_radius: float
    body_radius: float
    dt: float
    coverage_gain: float
    link_gain: float
    robot_speed: float
    desired_gap: float
    max_gap: float


def _compute_centroids(positions: jnp.ndarray, active: jnp.ndarray, params: SimParams) -> jnp.ndarray:
    points = params.coverage_grid
    diffs = points[:, None, :] - positions[None, :, :]
    dist_sq = jnp.sum(diffs * diffs, axis=-1)
    dist_sq = jnp.where(active[None, :], dist_sq, 1.0e12)
    assignment = jnp.argmin(dist_sq, axis=1)
    ownership = jax.nn.one_hot(assignment, positions.shape[0], dtype=positions.dtype)
    counts = jnp.sum(ownership, axis=0, keepdims=True)
    sums = ownership.T @ points
    centroids = sums / jnp.where(counts.T <= 0.0, 1.0, counts.T)
    active_counts = counts.T > 0.5
    centroids = jnp.where(active_counts, centroids, positions)
    return centroids


def _coverage_velocities(positions: jnp.ndarray, active: jnp.ndarray, params: SimParams) -> jnp.ndarray:
    centroids = _compute_centroids(positions, active, params)
    drive = centroids - positions
    return params.coverage_gain * drive


def _link_velocities(positions: jnp.ndarray, params: SimParams) -> jnp.ndarray:
    parent = jnp.concatenate([positions[:1], positions[:-1]], axis=0)
    delta = positions - parent
    dist = jnp.linalg.norm(delta, axis=1, keepdims=True)
    direction = jnp.where(dist > 1.0e-6, delta / dist, jnp.zeros_like(delta))
    gap_error = dist - params.desired_gap
    link = -direction * gap_error * params.link_gain
    link = link.at[0].set(jnp.zeros(2, dtype=positions.dtype))
    return link


def _limit_speed(velocities: jnp.ndarray, params: SimParams) -> jnp.ndarray:
    speed = jnp.linalg.norm(velocities, axis=1, keepdims=True)
    factor = jnp.where(
        speed > params.robot_speed,
        params.robot_speed / (speed + 1.0e-6),
        1.0,
    )
    return velocities * factor


def _clamp_world(positions: jnp.ndarray, params: SimParams) -> jnp.ndarray:
    lo = params.world_min + params.body_radius
    hi = params.world_max - params.body_radius
    clamped = jnp.clip(positions, lo, hi)
    return clamped.at[0].set(params.base_position)


def _enforce_chain(positions: jnp.ndarray, params: SimParams) -> jnp.ndarray:
    def body(i, pos):
        parent = pos[i - 1]
        child = pos[i]
        delta = child - parent
        dist = jnp.linalg.norm(delta)
        clipped = jnp.where(
            dist > params.max_gap,
            parent + delta / (dist + 1.0e-6) * params.max_gap,
            child,
        )
        return pos.at[i].set(clipped)

    return jax.lax.fori_loop(1, positions.shape[0], body, positions)


def _integrate(state: dict, params: SimParams) -> dict:
    pos = state["positions"]
    active = state["active"]
    coverage_term = _coverage_velocities(pos, active, params)
    link_term = _link_velocities(pos, params)
    velocities = coverage_term + link_term
    velocities = jnp.where(active[:, None], velocities, 0.0)
    velocities = velocities.at[0].set(jnp.zeros(2, dtype=velocities.dtype))
    velocities = _limit_speed(velocities, params)
    new_pos = pos + velocities * params.dt
    new_pos = _clamp_world(new_pos, params)
    new_pos = _enforce_chain(new_pos, params)
    return {"positions": new_pos, "active": active}


step_state = jax.jit(_integrate)


@jax.jit
def coverage_metrics(state: dict, params: SimParams) -> tuple[jnp.ndarray, jnp.ndarray]:
    pos = state["positions"]
    active = state["active"]
    points = params.coverage_grid
    diffs = points[:, None, :] - pos[None, :, :]
    dist_sq = jnp.sum(diffs * diffs, axis=-1)
    within = dist_sq <= (params.coverage_radius ** 2)
    mask = jnp.logical_and(within, active[None, :])
    covered = jnp.any(mask, axis=1)
    total = jnp.mean(covered.astype(jnp.float32))
    per_robot = jnp.sum(mask.astype(jnp.float32), axis=0) / points.shape[0]
    return total, per_robot
