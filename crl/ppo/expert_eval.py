"""Windowed off-policy expert value-agreement evaluation.

Protocol and rationale: ``LOGGING_CONTRACT.md``. In one line:
sample states from an expert's greedy trajectories on a task, snapshot the ALE
emulator state at each (O(1) via ``cloneState``), then measure the ONE-SIDED
shortfall of a policy's windowed return relative to the expert's, from those
states. It answers "can the policy match the expert's value on the expert's own
states?" -- a cheap diagnostic reported ALONGSIDE the on-policy no-op anchor
(never instead of it; see the doc's Limitations).

Both generation and evaluation use a PREPROCESSING-ONLY env
(:meth:`AtariTask.make_snapshot_env`) and a self-managed 4-frame stack, so the
snapshot/restore stays inside one pipeline and the frame stack (which lives in
``FrameStackObservation``, not the ALE core) is reconstructed exactly.
"""

from __future__ import annotations

import pickle
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch

from crl.policies.base import Policy

HORIZON_GRID = (15, 20, 25, 30, 40, 60)


@dataclass
class ExpertState:
    ale_state: object          # ALEState (picklable in ale_py); restores in O(1)
    stacked_obs: np.ndarray    # (frame_stack, 84, 84) uint8 -- obs the expert acted on
    v_win: float               # expert windowed discounted (sign-clipped) value V*_win
    raw_win: float             # expert windowed RAW score (game points)
    n_win: int                 # window length used = min(H_i, steps_to_term); policy rolls this many
    steps_to_term: int
    category: str              # "early" | "mid" | "near_terminal"


@dataclass
class ExpertStateSet:
    game: str
    task_id: int
    horizon: int               # H_i (agent-steps), adaptive per game
    gamma: float
    frame_stack: int
    clip_rewards: bool
    states: list               # list[ExpertState]
    meta: dict                 # generation settings + diagnostics


def _clip(r: float, clip: bool) -> float:
    return float(np.sign(r)) if clip else float(r)


def _greedy_actions(policy: Policy, stacks: np.ndarray, task_id: int,
                    device: torch.device) -> np.ndarray:
    """Batched greedy (argmax) actions for a bank of stacked uint8 observations."""
    obs = torch.as_tensor(stacks, device=device)          # (B, fs, 84, 84) uint8
    return policy.dist(obs, task_id).logits.argmax(dim=-1).to("cpu").numpy()


# --------------------------------------------------------------------------- #
# Precompute the expert state set (once per task; reusable across checkpoints)
# --------------------------------------------------------------------------- #

@torch.no_grad()
def build_expert_state_set(
    expert: Policy,
    task,
    device: torch.device,
    *,
    expert_task_id: int = 0,
    grid: tuple[int, ...] = HORIZON_GRID,
    tau: float = 0.5,
    consequential_thresh: float = 1.0,
    v_floor: float = 0.1,
    n_source_starts: int = 20,
    states_per_task: int = 5000,
    seed: int = 100_000,
    rng: np.random.Generator | None = None,
) -> ExpertStateSet:
    """Roll ``expert`` greedily from ``n_source_starts`` distinct no-op starts,
    snapshot every visited state, pick the adaptive horizon ``H_i``, keep the
    CONSEQUENTIAL windows, and sample up to ``states_per_task`` of them.

    A window is CONSEQUENTIAL iff the expert's RAW window score >=
    ``consequential_thresh`` (the expert scored non-trivially) AND its discounted
    sign-clipped value ``V*_win >= v_floor`` (so the relative-shortfall denominator
    is bounded away from 0 -- a raw-positive window can still be net-negative in
    discounted-clipped value, e.g. +1 early then several -1). ``H_i`` = smallest H
    in ``grid`` for which >= ``tau`` of expert windows are consequential (else the
    largest H, with a warning in ``meta``).
    """
    rng = rng or np.random.default_rng(seed)
    gamma = float(task.gamma)
    fs = int(task.frame_stack)
    clip = bool(getattr(task, "clip_rewards", True))
    max_gen = int(getattr(task, "max_steps", 0) or 10_000)

    # ---- roll the expert; record (ale_state, stacked_obs, raw_reward) per step ---
    trajectories: list[list[tuple]] = []
    ended_terminal: list[bool] = []                        # true terminal vs max_gen cut
    for k in range(n_source_starts):
        env = task.make_snapshot_env(clip_rewards=False)   # raw rewards; single frame
        frame, _ = env.reset(seed=seed + k)                # noop_max gives start diversity
        ale = env.unwrapped.ale
        stack = deque([np.asarray(frame, np.uint8)] * fs, maxlen=fs)  # FrameStack reset semantics
        traj, done, steps, last_term = [], False, 0, False
        while not done and steps < max_gen:
            obs = np.stack(stack, axis=0)                  # (fs, 84, 84) oldest..newest
            a = int(_greedy_actions(expert, obs[None], expert_task_id, device)[0])
            st = ale.cloneState()                          # snapshot BEFORE stepping -> state s_t
            frame, r, term, trunc, _ = env.step(a)
            traj.append((st, obs, float(r)))               # obs the expert acted on + reward r_t
            stack.append(np.asarray(frame, np.uint8))
            last_term = bool(term)
            done, steps = (term or trunc), steps + 1
        env.close()
        if traj:
            trajectories.append(traj)
            ended_terminal.append(last_term)

    rewards = [[r for _, _, r in tr] for tr in trajectories]

    # ---- windowed expert values, for any (trajectory, start index t) -----------
    def window(tr_i: int, t: int, H: int) -> tuple[float, float, int]:
        rs = rewards[tr_i]
        n = min(H, len(rs) - t)
        raw = float(sum(rs[t:t + n]))
        v = float(sum((gamma ** j) * _clip(rs[t + j], clip) for j in range(n)))
        return v, raw, n

    def consequential(v: float, raw: float) -> bool:
        # expert scored (raw) AND discounted-clipped value bounded away from 0, so
        # the relative-shortfall denominator V*_win is safe (see the MAJOR fix).
        return raw >= consequential_thresh and v >= v_floor

    # ---- adaptive horizon H_i: smallest H with >= tau consequential windows ----
    def consequential_frac(H: int) -> float:
        hit = tot = 0
        for i, rs in enumerate(rewards):
            for t in range(len(rs)):
                tot += 1
                v, raw, _ = window(i, t, H)
                if consequential(v, raw):
                    hit += 1
        return (hit / tot) if tot else 0.0

    fracs = {H: consequential_frac(H) for H in grid}
    H_i = next((H for H in sorted(grid) if fracs[H] >= tau), max(grid))
    warn = None if fracs[H_i] >= tau else f"no H reached tau={tau}; using max grid {H_i}"

    # ---- collect consequential candidates at H_i, then sample -------------------
    cands: list[ExpertState] = []
    for i, tr in enumerate(trajectories):
        T = len(tr)
        for t in range(T):
            v, raw, n = window(i, t, H_i)
            if not consequential(v, raw):
                continue
            stt = T - t
            # near_terminal only when the trajectory TRULY terminated (not cut by the
            # max_gen cap), so the diagnostic label is not spuriously applied at a cut.
            cat = ("near_terminal" if (ended_terminal[i] and stt <= H_i)
                   else "early" if t < 20 else "mid")
            cands.append(ExpertState(tr[t][0], tr[t][1], v, raw, n, stt, cat))

    if len(cands) > states_per_task:
        pick = rng.choice(len(cands), size=states_per_task, replace=False)
        cands = [cands[i] for i in pick]

    meta = {
        "n_source_starts": n_source_starts, "consequential_thresh": consequential_thresh,
        "tau": tau, "consequential_frac_by_H": fracs, "warning": warn,
        "n_candidates_kept": len(cands),
        "traj_lengths": [len(tr) for tr in trajectories],
        "category_counts": {c: sum(s.category == c for s in cands)
                            for c in ("early", "mid", "near_terminal")},
    }
    return ExpertStateSet(task.game, expert_task_id, H_i, gamma, fs, clip, cands, meta)


def save_state_set(state_set: ExpertStateSet, path: str) -> None:
    with open(path, "wb") as f:
        pickle.dump(state_set, f)


def load_state_set(path: str) -> ExpertStateSet:
    with open(path, "rb") as f:
        return pickle.load(f)


# --------------------------------------------------------------------------- #
# Evaluate a policy against a precomputed expert state set
# --------------------------------------------------------------------------- #

@torch.no_grad()
def evaluate_expert_agreement(
    policy: Policy,
    task,
    state_set: ExpertStateSet,
    device: torch.device,
    task_id: int,
    *,
    n_envs: int = 8,
) -> dict:
    """One-sided windowed value agreement of ``policy`` (head ``task_id``) with the
    expert on ``state_set``. Restores each expert state (O(1)), rolls the policy
    GREEDILY for that state's ``n_win`` steps (= the SAME window length the expert's
    V*_win covered, so the two returns are strictly equal-horizon; capped at
    terminal), and returns the mean one-sided shortfall
    ``mean_s [V*_win(s) - Vpi_win(s)]_+`` and its relative form. Lower is better;
    0 == matches/beats the expert on every kept state."""
    H, gamma, fs, clip = (state_set.horizon, state_set.gamma,
                          state_set.frame_stack, state_set.clip_rewards)
    states = state_set.states
    if not states:
        return {"agreement_gap": float("nan"), "relative_gap": float("nan"),
                "n_states": 0, "horizon": H}

    envs = [task.make_snapshot_env(clip_rewards=False) for _ in range(n_envs)]
    ales = [e.unwrapped.ale for e in envs]
    for e in envs:
        e.reset(seed=0)                                    # init ALE before restore

    sfs: list[float] = []
    sfs_rel: list[float] = []
    per_cat: dict[str, list[float]] = {"early": [], "mid": [], "near_terminal": []}
    try:
        for c0 in range(0, len(states), n_envs):
            chunk = states[c0:c0 + n_envs]
            B = len(chunk)
            stacks = np.zeros((B, fs, 84, 84), dtype=np.uint8)
            vpi = np.zeros(B)
            gpow = np.ones(B)
            done = np.zeros(B, dtype=bool)
            wlen = np.array([es.n_win for es in chunk])   # per-state equal-horizon length
            for j, es in enumerate(chunk):
                ales[j].restoreState(es.ale_state)
                try:
                    envs[j].lives = ales[j].lives()        # sync stale life counter
                except Exception:
                    pass
                stacks[j] = es.stacked_obs
            for step in range(int(wlen.max()) if B else 0):
                if done.all():
                    break
                acts = _greedy_actions(policy, stacks, task_id, device)
                for j in range(B):
                    if done[j]:
                        continue
                    if step >= wlen[j]:                    # reached expert window length -> stop
                        done[j] = True
                        continue
                    frame, r, term, trunc, _ = envs[j].step(int(acts[j]))
                    vpi[j] += gpow[j] * _clip(r, clip)
                    gpow[j] *= gamma
                    stacks[j] = np.concatenate(
                        [stacks[j][1:], np.asarray(frame, np.uint8)[None]], axis=0)
                    if term or trunc:
                        done[j] = True
            for j, es in enumerate(chunk):
                gap = max(0.0, es.v_win - float(vpi[j]))
                rel = gap / es.v_win if es.v_win > 0 else 0.0
                sfs.append(gap)
                sfs_rel.append(rel)
                per_cat[es.category].append(rel)
    finally:
        for e in envs:
            e.close()

    return {
        "agreement_gap": float(np.mean(sfs)),
        "relative_gap": float(np.mean(sfs_rel)),
        "n_states": len(states),
        "horizon": H,
        "relative_gap_by_category": {c: (float(np.mean(v)) if v else float("nan"))
                                     for c, v in per_cat.items()},
    }
