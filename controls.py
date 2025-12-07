from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp


class SimParams(NamedTuple):
    coverage_grid: jnp.ndarray
    world_min: jnp.ndarray
    world_max: jnp.ndarray
    base_position: jnp.ndarray
    box_min: jnp.ndarray
    box_max: jnp.ndarray
    coverage_radius: float
    body_radius: float
    dt: float
    coverage_gain: float
    link_gain: float
    robot_speed: float
    desired_gap: float
    max_gap: float
    box_gain: float
    redundancy_gain: float


def _pairwise_distances(positions: jnp.ndarray, params: SimParams) -> tuple[jnp.ndarray, jnp.ndarray]:
    points = params.coverage_grid
    diffs = points[:, None, :] - positions[None, :, :]
    dist_sq = jnp.sum(diffs * diffs, axis=-1)
    return points, dist_sq


def _coverage_stats(
    positions: jnp.ndarray, active: jnp.ndarray, params: SimParams
) -> tuple[jnp.ndarray, jnp.ndarray]:
    points, dist_sq = _pairwise_distances(positions, params)
    mask = jnp.logical_and(dist_sq <= (params.coverage_radius ** 2), active[None, :])
    masked_dist = jnp.where(active[None, :], dist_sq, 1.0e12)
    assignment = jnp.argmin(masked_dist, axis=1)
    ownership = jax.nn.one_hot(assignment, positions.shape[0], dtype=positions.dtype)
    counts = jnp.sum(ownership, axis=0, keepdims=True)
    sums = ownership.T @ points
    centroids = sums / jnp.where(counts.T <= 0.0, 1.0, counts.T)
    active_counts = counts.T > 0.5
    centroids = jnp.where(active_counts, centroids, positions)
    return centroids, mask


def _redundancy_weights(mask: jnp.ndarray, params: SimParams) -> jnp.ndarray:
    mask_f = mask.astype(jnp.float32)
    coverage_hits = jnp.sum(mask_f, axis=0)
    share_counts = jnp.sum(mask_f, axis=1, keepdims=True)
    normalized = jnp.where(share_counts > 0.5, mask_f / share_counts, 0.0)
    unique_hits = jnp.sum(normalized, axis=0)
    ratio = jnp.where(coverage_hits > 0.5, unique_hits / (coverage_hits + 1.0e-6), 1.0)
    redundancy = 1.0 - ratio
    weights = 1.0 + params.redundancy_gain * redundancy
    weights = weights.at[0].set(0.0)
    return weights


def _coverage_velocities(positions: jnp.ndarray, active: jnp.ndarray, params: SimParams) -> jnp.ndarray:
    centroids, mask = _coverage_stats(positions, active, params)
    drive = centroids - positions
    weights = _redundancy_weights(mask, params)[:, None]
    return params.coverage_gain * drive * weights


def _link_velocities(positions: jnp.ndarray, params: SimParams) -> jnp.ndarray:
    parent = jnp.concatenate([positions[:1], positions[:-1]], axis=0)
    delta = positions - parent
    dist = jnp.linalg.norm(delta, axis=1, keepdims=True)
    direction = jnp.where(dist > 1.0e-6, delta / dist, jnp.zeros_like(delta))
    gap_error = dist - params.desired_gap
    link = -direction * gap_error * params.link_gain
    link = link.at[0].set(jnp.zeros(2, dtype=positions.dtype))
    return link


def _box_velocities(positions: jnp.ndarray, params: SimParams) -> jnp.ndarray:
    clip_lo = params.box_min
    clip_hi = params.box_max
    clipped = jnp.clip(positions, clip_lo, clip_hi)
    return (clipped - positions) * params.box_gain


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
    box_term = _box_velocities(pos, params)
    velocities = coverage_term + link_term + box_term
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
    points, dist_sq = _pairwise_distances(pos, params)
    within = dist_sq <= (params.coverage_radius ** 2)
    mask = jnp.logical_and(within, active[None, :])
    covered = jnp.any(mask, axis=1)
    total = jnp.mean(covered.astype(jnp.float32))
    per_robot = jnp.sum(mask.astype(jnp.float32), axis=0) / points.shape[0]
    return total, per_robot
