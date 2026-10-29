"""Reusable PPO core, specialized into Local and Global trainers.

``PPOTrainer`` is the *entire* standard PPO optimizer: it turns a set of collected
rollout streams into gradient steps via the clipped surrogate, a standard
value-MSE critic loss, an entropy bonus, GAE advantages, and global grad-norm
clipping. Nothing continual lives here.

Two specializations:

* :class:`LocalTrainer` -- standard PPO on the current task (one stream, actor
  coefficient 1). No constraint, no past-task rollouts.
* :class:`GlobalTrainer` -- the SAME PPO core, but the actor receives the
  continual-learning term: past tasks enter with weight ``omega_i`` and the
  current task enters with coefficient ``mu * 2 * shortfall`` (the differentiated
  one-sided squared shortfall, eqs 30-32). It owns the replay-free fresh
  past-task rollouts, the Monte-Carlo shortfall estimate, and the ``mu`` dual
  update. The critic / GAE / value loss / entropy are inherited unchanged --
  only the ACTOR is constrained.

The only thing that differs between the two is the per-stream **actor
coefficient**; the optimization routine (:meth:`PPOTrainer.optimize_batches`) is
shared verbatim.
"""

from __future__ import annotations

import time
from typing import Callable

import torch
import torch.nn as nn

from crl.config import PPOConfig
from crl.duals.controllers import DualController
from crl.envs.base import Task
from crl.policies.base import Policy
from crl.ppo.collector import RolloutBatch, RolloutCollector
from crl.ppo.evaluate import evaluate_value_and_score

# Callback invoked once per PPO iteration: (phase_type, current_task) -> None.
ProbeHook = Callable[[str, int], None]


class PPOTrainer:
    """Standard clipped-PPO optimizer shared by the local and global phases."""

    def __init__(self, cfg: PPOConfig, device: torch.device, logger, log_every: int) -> None:
        self.ppo = cfg
        self.device = device
        self.logger = logger
        self.log_every = max(1, log_every)

    def _new_optimizer(self, policy: Policy) -> torch.optim.Optimizer:
        # Adam over the trainable params (whole model normally; only the heads
        # when the trunk is frozen for the head-only consolidation probe).
        params = [p for p in policy.parameters() if p.requires_grad]
        return torch.optim.Adam(params, lr=self.ppo.lr, eps=1e-5)

    def _stop_score(self, policy: Policy, task: Task, iters_done: int) -> float | None:
        """Greedy score for the early-stop check on this iteration, or None if
        this iteration is not a check point (or the task has no threshold)."""
        cfg = self.ppo
        thr = float(getattr(task, "threshold", float("inf")))
        if thr == float("inf") or cfg.stop_eval_every <= 0:
            return None
        if iters_done < cfg.min_iters or iters_done % cfg.stop_eval_every != 0:
            return None
        _, score, _, _ = evaluate_value_and_score(
            policy, task, cfg.stop_eval_episodes, cfg.n_envs, self.device,
            seed=cfg.eval_seed, greedy=True,
        )
        return score

    def optimize_batches(
        self,
        policy: Policy,
        optimizer: torch.optim.Optimizer,
        streams: list[RolloutBatch],
        actor_coeffs: list[float],
        bc: tuple | None = None,
        kl_stop: bool = True,
    ) -> dict[str, float]:
        """One PPO update (``ppo_epochs`` x minibatches) over the given streams.

        Each stream contributes ``actor_coeff * clipped_surrogate`` to the actor
        loss and a standard ``vf_coef * value_MSE`` to the critic loss, plus the
        standard entropy bonus. All streams share the minibatch schedule (equal
        size ``N = n_envs * n_steps``).

        A stream collected against an external (frozen specialist) critic sets
        ``batch.external_critic``; its value loss is skipped, since its returns
        belong to that specialist rather than to the policy being fitted.
        """
        cfg = self.ppo
        n = streams[0].obs.shape[0]
        mb_size = max(1, n // cfg.num_minibatches)
        # Normalize the actor coefficients so the actor-loss scale stays O(1)
        # regardless of the dual multiplier's magnitude (PPO-Lagrangian style).
        # The dual coefficient mu*2*shortfall is unbounded; without this, a large
        # mu makes the actor gradient dominate the shared global grad-norm clip
        # and STARVES the shared critic's value-loss gradient, breaking GAE
        # advantages so the actor can no longer improve. Dividing every actor
        # coefficient by their sum is a positive rescaling of the ascent
        # direction -- it preserves the relative past/current weighting (and thus
        # the primal-dual fixed point), only bounding the step magnitude. For the
        # local phase (coeffs == [1.0]) this is a no-op.
        coeff_sum = float(sum(actor_coeffs))
        if coeff_sum > 0:
            actor_coeffs = [c / coeff_sum for c in actor_coeffs]
        pg_acc = v_acc = ent_acc = kl_acc = clip_acc = 0.0
        count = 0
        for _ in range(cfg.ppo_epochs):
            perm = torch.randperm(n, device=self.device)
            ep_kl = 0.0
            ep_n = 0
            for start in range(0, n, mb_size):
                idx = perm[start : start + mb_size]
                total_loss = torch.zeros((), device=self.device)
                n_streams = len(streams)
                for si, (coeff, batch) in enumerate(zip(actor_coeffs, streams)):
                    # A stream whose advantages came from a frozen specialist's
                    # critic carries THAT critic's returns, which are not a valid
                    # regression target for the global critic. Skip its value loss
                    # (and the critic forward pass) rather than dragging the global
                    # critic toward the specialist's value function.
                    fit_critic = not batch.external_critic
                    if fit_critic:
                        dist, value = policy.dist_value(batch.obs[idx], batch.task_id)
                    else:
                        dist, value = policy.dist(batch.obs[idx], batch.task_id), None
                    new_logp = dist.log_prob(batch.actions[idx])
                    logratio = new_logp - batch.logprobs[idx]
                    ratio = logratio.exp()
                    adv = batch.advantages[idx]
                    if cfg.normalize_advantage:
                        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                    pg_loss = torch.max(
                        -adv * ratio,
                        -adv * torch.clamp(ratio, 1 - cfg.clip_ratio, 1 + cfg.clip_ratio),
                    ).mean()
                    entropy = dist.entropy().mean()
                    total_loss = (
                        total_loss
                        + coeff * pg_loss
                        - cfg.ent_coef * entropy
                    )
                    v_loss = torch.zeros((), device=self.device)
                    if fit_critic:
                        v_loss = 0.5 * (value - batch.returns[idx]).pow(2).mean()
                        total_loss = total_loss + cfg.vf_coef * v_loss
                    # EXPERIMENT 2: behavioral-cloning term on the CURRENT task
                    # (the last stream): coef * KL(pi_local || pi_global).
                    if bc is not None and si == n_streams - 1:
                        local_policy, bc_coef = bc
                        with torch.no_grad():
                            pl = local_policy.dist(batch.obs[idx], batch.task_id)
                        total_loss = total_loss + bc_coef * torch.distributions.kl_divergence(
                            pl, dist).mean()
                    with torch.no_grad():
                        pg_acc += float(pg_loss)
                        v_acc += float(v_loss)
                        ent_acc += float(entropy)
                        mb_kl = float((ratio - 1 - logratio).mean())
                        kl_acc += mb_kl
                        ep_kl += mb_kl
                        ep_n += 1
                        clip_acc += float(
                            ((ratio - 1.0).abs() > cfg.clip_ratio).float().mean()
                        )
                        count += 1
                optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
                optimizer.step()
            # KL early-stop: stop optimizing this batch once the mean policy shift
            # for the epoch exceeds target_kl (prevents the runaway actor blow-up
            # observed on Boxing). Off when target_kl == 0 or kl_stop is False -- the
            # GLOBAL consolidation phase runs FULL epochs (kl_stop=False) so the
            # min-max retention update is not throttled; only the LOCAL phase uses it.
            if kl_stop and cfg.target_kl and ep_n and ep_kl / ep_n > cfg.target_kl:
                break
        c = max(1, count)
        return {
            "pg_loss": pg_acc / c,
            "v_loss": v_acc / c,
            "entropy": ent_acc / c,
            "approx_kl": kl_acc / c,
            "clipfrac": clip_acc / c,
        }


class LocalTrainer(PPOTrainer):
    """Standard PPO on the current task (the min-max *local* player)."""

    def train(
        self,
        policy: Policy,
        task: Task,
        num_iters: int,
        seed: int,
        current_task: int,
        phase_type: str = "local",
        probe: ProbeHook | None = None,
        clog=None,
        ent_coef_override: float | None = None,
        stall_at: int = 0,
        stall_thr: float = 0.0,
    ) -> dict:
        optimizer = self._new_optimizer(policy)
        # Temporarily boost exploration for a retry attempt (restored in finally).
        _saved_ent = self.ppo.ent_coef
        if ent_coef_override is not None:
            self.ppo.ent_coef = float(ent_coef_override)
        collector = RolloutCollector(
            task, self.ppo.n_envs, self.ppo.n_steps, self.device, seed,
            async_mode=self.ppo.async_envs,
        )
        thr = float(getattr(task, "threshold", float("inf")))
        met = 0
        stalled = False
        t0, early, gscore = time.time(), False, None
        best_score, best_state = float("-inf"), None
        try:
            for it in range(num_iters):
                batch = collector.collect(policy, self.ppo.gae_lambda)
                stats = self.optimize_batches(policy, optimizer, [batch], [1.0])
                # Continual-backprop generate-and-test fires AFTER the optimizer step
                # (Dohare 2024). No-op for policies without cbp_step (all but CbpNet).
                if hasattr(policy, "cbp_step"):
                    policy.cbp_step(optimizer)
                if probe is not None:
                    probe(phase_type, current_task)
                gscore = self._stop_score(policy, task, it + 1)
                # Within-phase FWT learning curve (contract): the training policy's
                # greedy score on its OWN task, on the eval_every cadence -- decoupled
                # from the threshold-gated stop check, but REUSING its score when they
                # coincide so there is no double eval. Without this there is no p_i(t)
                # and forward transfer cannot be recovered from a finished run.
                if (clog is not None and self.ppo.eval_every
                        and (it + 1) % self.ppo.eval_every == 0):
                    fwt = gscore
                    if fwt is None:
                        _, fwt, _, _ = evaluate_value_and_score(
                            policy, task, self.ppo.stop_eval_episodes, self.ppo.n_envs,
                            self.device, seed=self.ppo.eval_seed, greedy=True)
                    clog.eval(task_idx=current_task - 1, phase=phase_type, it=it + 1,
                              evaluated_on=current_task - 1,
                              evaluated_on_task=task.spec.name, raw=float(fwt),
                              episodes=self.ppo.stop_eval_episodes, greedy=True, seen=True)
                if gscore is not None:
                    met = met + 1 if gscore >= thr else 0
                    if self.ppo.select_best_local and gscore > best_score:
                        best_score = gscore   # snapshot the best-greedy model (salvage the peak)
                        best_state = {k: v.detach().clone()
                                      for k, v in policy.state_dict().items()}
                # Stall guard: explicit greedy eval at the checkpoint; if the specialist
                # is still not learning, abort this attempt so the caller can retry fresh.
                if stall_at and (it + 1) == stall_at:
                    _, sg, _, _ = evaluate_value_and_score(
                        policy, task, self.ppo.stop_eval_episodes, self.ppo.n_envs,
                        self.device, seed=self.ppo.eval_seed, greedy=True)
                    peak = max(sg, best_score) if best_state is not None else sg
                    if peak < stall_thr:
                        stalled = True
                        # Snapshot this (poor) specialist so best_greedy is a real number
                        # and the caller can compare/keep it across retry attempts.
                        if best_state is None or sg > best_score:
                            best_score = sg
                            best_state = {k: v.detach().clone()
                                          for k, v in policy.state_dict().items()}
                        print(f"[{phase_type} k={current_task}] STALL at it={it+1} "
                              f"greedy={sg:.3f} (peak {peak:.3f}) < {stall_thr} -> abort for retry")
                        break
                if it % self.log_every == 0 or gscore is not None:
                    ep = batch.ep_returns
                    ep_mean = sum(ep) / len(ep) if ep else float("nan")
                    self.logger.log(
                        {
                            "phase": phase_type, "task": current_task, "step": it,
                            "ep_return_clipped": ep_mean, "greedy_score": gscore,
                            **stats,
                        }
                    )
                    gs = f" greedy={gscore:.1f}/{thr:.0f}" if gscore is not None else ""
                    print(
                        f"[{phase_type} k={current_task}] it={it:4d} "
                        f"epR(clip)={ep_mean:.3f} pg={stats['pg_loss']:.3f} "
                        f"v={stats['v_loss']:.3f} ent={stats['entropy']:.3f}{gs}"
                    )
                if met >= self.ppo.patience:
                    early = True
                    print(f"[{phase_type} k={current_task}] EARLY STOP it={it+1} "
                          f"greedy={gscore:.1f} >= thr={thr:.0f}")
                    break
        finally:
            collector.close()
            self.ppo.ent_coef = _saved_ent
        # Keep the best-greedy snapshot if the phase ended worse than its peak
        # (e.g. a late divergence). No-op when select_best_local is off or the final
        # already is the best.
        final_g = gscore if gscore is not None else float("-inf")
        if best_state is not None and best_score > final_g:
            policy.load_state_dict(best_state)
            print(f"[{phase_type} k={current_task}] kept BEST model greedy={best_score:.1f} "
                  f"(final was {gscore})")
        return {"iters_run": it + 1, "early_stopped": early, "stalled": stalled,
                "wall_s": round(time.time() - t0, 1), "final_greedy": gscore,
                "best_greedy": (best_score if best_state is not None else gscore)}


class GlobalTrainer(PPOTrainer):
    """PPO consolidation with the actor-only continual-learning constraint.

    Maximizes the past-task objective ``sum_i omega_i V_i`` while the single
    multiplier ``mu`` enforces the current-task floor ``V_k^G >= V_k^L`` via the
    one-sided squared shortfall (eqs 11-14, 30-32). Fresh rollouts in every task's
    environment each iteration (replay-free).
    """

    def _actor_pg(self, policy: Policy, batch: RolloutBatch) -> torch.Tensor:
        """Clipped-surrogate actor loss on one full stream (for grad diagnostics)."""
        cfg = self.ppo
        dist, _ = policy.dist_value(batch.obs, batch.task_id)
        ratio = (dist.log_prob(batch.actions) - batch.logprobs).exp()
        adv = batch.advantages
        if cfg.normalize_advantage:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        return torch.max(
            -adv * ratio,
            -adv * torch.clamp(ratio, 1 - cfg.clip_ratio, 1 + cfg.clip_ratio),
        ).mean()

    def _grad_decomposition(self, policy, past_batches, cur_batch, omega, coeff_k):
        """Decompose the actor update into the current-task constraint gradient
        g_new = grad[coeff_k * pg_k] and EACH past-task gradient g_i =
        grad[omega_i * pg_i] separately (coeffs normalized as in the real update).

        The aggregate cos(g_new, sum_i g_i) can read ~0 while individual tasks
        strongly align/conflict and cancel in the sum, so we return the PER-TASK
        alignment cos(g_new, g_i) as well. Returns
        (||g_new||, ||g_old||, ||g_new||/||g_old||, cos_aggregate, per_task) where
        per_task[i] = {grad_norm_i, grad_cos_new_i} aligned to past_batches."""
        params = [p for p in policy.parameters() if p.requires_grad]
        coeff_sum = float(sum(omega[: len(past_batches)]) + coeff_k)
        scale = (1.0 / coeff_sum) if coeff_sum > 0 else 1.0

        def flat(loss):
            gs = torch.autograd.grad(loss, params, allow_unused=True, retain_graph=False)
            return torch.cat([(g if g is not None else torch.zeros_like(p)).reshape(-1)
                              for g, p in zip(gs, params)])

        def cosine(a, b):
            d = a.norm() * b.norm()
            return float(torch.dot(a, b) / d) if float(d) > 0 else float("nan")

        g_new = flat(coeff_k * scale * self._actor_pg(policy, cur_batch))
        per_task = []                       # aligned to past_batches order
        g_old = torch.zeros_like(g_new)
        for w, b in zip(omega, past_batches):
            g_i = flat(w * scale * self._actor_pg(policy, b))
            per_task.append({"grad_norm_i": float(g_i.norm()),
                             "grad_cos_new_i": cosine(g_new, g_i)})
            g_old = g_old + g_i
        nn_new, nn_old = float(g_new.norm()), float(g_old.norm())
        ratio = nn_new / nn_old if nn_old > 0 else float("inf")
        return nn_new, nn_old, ratio, cosine(g_new, g_old), per_task

    def _log_diagnostics(self, policy, task_k, past_tasks, ref_current, mu, shortfall,
                         constraint, eps, coeff_k, v_k_g, past_batches, cur_batch,
                         omega, current_task, step, local_policy=None):
        """Emit one rich 'global_diag' row (see PPOConfig.diagnostics)."""
        cfg = self.ppo
        # EXPERIMENT 2: behavioral gap between the frozen local and the global on
        # current-task states. If the value gap -> 0 while this KL stays large,
        # the scalar value constraint is satisfied by a behaviorally-different
        # policy (the value constraint is too weak).
        kl_lg = tv_lg = float("nan")
        if local_policy is not None:
            with torch.no_grad():
                pl = local_policy.dist(cur_batch.obs, cur_batch.task_id)
                pg = policy.dist(cur_batch.obs, cur_batch.task_id)
                kl_lg = float(torch.distributions.kl_divergence(pl, pg).mean())
                tv_lg = float(0.5 * (pl.probs - pg.probs).abs().sum(-1).mean())
        # current-task greedy performance (trajectory point) + fresh stochastic V_k^G
        _, cur_score, _, _ = evaluate_value_and_score(
            policy, task_k, cfg.diag_score_episodes, cfg.n_envs, self.device,
            seed=cfg.eval_seed, greedy=True)
        # each past task individually: value V_i^G (stochastic) + greedy score
        past = []
        for t in past_tasks:
            vi, _, _, _ = evaluate_value_and_score(
                policy, t, cfg.constraint_episodes, cfg.n_envs, self.device, greedy=False)
            _, si, _, _ = evaluate_value_and_score(
                policy, t, cfg.diag_score_episodes, cfg.n_envs, self.device,
                seed=cfg.eval_seed, greedy=True)
            past.append({"name": t.spec.name, "task_id": t.spec.task_id,
                         "V_i_global": vi, "greedy_score": si})
        nn_new, nn_old, ratio, cos, per_task = self._grad_decomposition(
            policy, past_batches, cur_batch, omega, coeff_k)
        for pd, gd in zip(past, per_task):  # attach per-task grad alignment
            pd.update(gd)
        self.logger.log({
            "phase": "global_diag", "task": current_task, "step": step,
            "cur_name": task_k.spec.name,
            "mu": mu, "coeff_k": coeff_k,
            "V_k_local": ref_current, "V_k_global": v_k_g,
            "V_gap": ref_current - v_k_g,           # V_L - V_G (oscillation)
            "F_G": constraint, "eps": eps,
            "constraint_active": bool(shortfall > 0.0),
            "cur_greedy_score": cur_score,          # current-task perf trajectory
            "grad_norm_new": nn_new, "grad_norm_old": nn_old,
            "grad_ratio_new_over_old": ratio,       # <1 => old-task objective wins
            "grad_cos_new_old": cos,                # alignment / conflict
            "kl_local_global": kl_lg,               # behavioral gap (value too weak?)
            "tv_local_global": tv_lg,
            "past": past,                           # per-old-task value + score
        })
        print(f"[diag k={current_task}] it={step:4d} Vgap={ref_current-v_k_g:+.3f} "
              f"mu={mu:.2f} F_G={constraint:.4f} |g_new|/|g_old|={ratio:.3f} "
              f"cos={cos:+.3f} cur={cur_score:.1f}")

    def _retention_check(self, policy, past_tasks, task_k, refs, frac, iters_done):
        """V5 all-tasks early-stop probe. Only a check iteration returns a result:
        (current_task_greedy_score_or_None, all_ok), where all_ok is True iff EVERY
        seen task (past + current) scores >= frac * its LOCAL reference greedy."""
        cfg = self.ppo
        if (iters_done < cfg.min_iters or cfg.stop_eval_every <= 0
                or iters_done % cfg.stop_eval_every != 0):
            return None, False
        seen = list(past_tasks) + [task_k]
        all_ok, cur = True, float("nan")
        scores = []
        for t, ref in zip(seen, refs):
            _, sc, _, _ = evaluate_value_and_score(
                policy, t, cfg.stop_eval_episodes, cfg.n_envs, self.device,
                seed=cfg.eval_seed, greedy=True, max_ep_steps=cfg.eval_max_ep_steps)
            scores.append(sc / ref if ref and ref > 0 else sc)
            if t is task_k:
                cur = sc
            if ref is None or ref <= 0 or sc < frac * ref:
                all_ok = False
        # Retention of each PAST task, cached for past_task_sampling="worst".
        # Free: this loop already evaluated every seen task.
        self._past_retention = scores[:len(past_tasks)]
        return cur, all_ok

    def _windowed_value_gap(self, global_policy, local_policy, task_k, task_idx, H,
                            n_envs, seed, greedy):
        """Windowed-bootstrap (Vk_L, Vk_G), mirroring the MW SAC windowed_shortfall
        in the PPO/actor-critic setting: the frozen LOCAL critic values BOTH
        endpoints, the GLOBAL policy rolls the H-step reward.
            Vk_L = V_L(S_0);  Vk_G = sum_{t<H} g^t r_t + g^H V_L(S_H).
        Per-env means over one reset batch. Envs that terminate mid-window stop
        accumulating reward and get their bootstrap zeroed (same convention as MC)."""
        gamma = task_k.gamma
        clip = bool(getattr(task_k, "clip_rewards", False))
        venv = task_k.make_vector_env(n_envs, clip_rewards=False)  # raw rewards
        try:
            obs, _ = venv.reset(seed=seed)
            obs = torch.as_tensor(obs, device=self.device)
            with torch.no_grad():
                v0 = local_policy.value(obs, task_idx).detach().cpu()  # V_L(S_0) per env
            disc = torch.zeros(n_envs)
            gpow = torch.ones(n_envs)
            done_mask = torch.zeros(n_envs, dtype=torch.bool)
            last_obs = obs
            for _ in range(H):
                with torch.no_grad():
                    logits = global_policy.dist(obs, task_idx).logits
                    action = (logits.argmax(dim=-1) if greedy
                              else torch.distributions.Categorical(logits=logits).sample())
                obs_np, reward, term, trunc, _ = venv.step(action.to("cpu").numpy())
                reward_t = torch.as_tensor(reward, dtype=torch.float32)
                train_r = torch.sign(reward_t) if clip else reward_t
                active = ~done_mask
                disc[active] += gpow[active] * train_r[active]
                gpow[active] *= gamma
                newly = active & torch.as_tensor((term | trunc))
                done_mask |= newly
                obs = last_obs = torch.as_tensor(obs_np, device=self.device)  # advance state for next global action
                if bool(done_mask.all()):
                    break
            with torch.no_grad():
                vH = local_policy.value(last_obs, task_idx).detach().cpu()  # V_L(S_H)
            boot = (gamma ** H) * vH
            boot[done_mask] = 0.0
            vk_g = disc + boot
            return float(v0.mean()), float(vk_g.mean())
        finally:
            venv.close()

    def train(
        self,
        global_policy: Policy,
        task_k: Task,
        past_tasks: list[Task],
        ref_current: float,
        mu_ctrl: DualController,
        omega: list[float],
        eps: float,
        num_iters: int,
        seed: int,
        current_task: int,
        probe: ProbeHook | None = None,
        local_policy: Policy | None = None,
        retention_refs: list[float] | None = None,
        retention_frac: float = 0.7,
        past_experts: list[Policy] | None = None,
        clog=None,
    ) -> dict:
        """Consolidate the global policy (Part A -- experts NOT stored).

        Maximizes the past-task objective ``sum_{i<k} omega_i V_i^{pi_phi}``
        (past weights are ``omega[:len(past_tasks)]``) while a single multiplier
        ``mu`` enforces the current-task floor via the one-sided squared hinge
        ``F_k = [V_k^L - V_k^G]_+^2 <= eps`` (objective doc eqs 26, 27). The
        current task's actor coefficient is the differentiated shortfall
        ``2 mu [V_k^L - V_k^G]_+`` (eqs 38, 40); ``mu`` follows the projected
        dual ascent ``mu <- [mu + eta (F_k - eps)]_+`` (eq 47). ``V_k^{L,G}`` are
        fresh on-policy Monte-Carlo estimates. Past-task rollouts are always
        fresh (replay-free); ``past_task_sampling`` decides how many per
        iteration -- ``all`` rolls every past environment (O(k)), ``sample``
        draws ONE uniformly and rescales it by ``k-1`` (unbiased, O(1)). The
        50-task GridWorld runs use ``sample``; the Atari runs use ``all``.

        ``ppo.value_mode='residual'`` switches consolidation to the Bellman-residual
        form. Every task's advantage becomes ``A_t = G_t - V_L_i(s_t)``: the
        FULL-EPISODE return minus that task's own frozen specialist critic, used
        purely as a BASELINE. Because a state-dependent baseline is unbiased no
        matter how wrong it is, the specialist cannot skew the gradient; only a
        bootstrap could, which is why ``global_gae_lambda`` must be 1 and the
        rollout must cover whole episodes.

        The constraint scalar then costs nothing extra. By telescoping,
        ``-mean(A_0)`` over start states equals ``V_k^{pi_L} - V_k^{pi_phi}``
        exactly, so the same rollout yields both the gradient and the shortfall.
        There is no window, no second estimator, and the global critic plays no
        part at all (its value loss is gated off for every expert-scored stream;
        see :meth:`optimize_batches`)."""
        optimizer = self._new_optimizer(global_policy)
        cfg = self.ppo
        # ---- Bellman-residual consolidation -------------------------------
        # Every task's advantage becomes A_t = G_t - V_L_i(s_t): the full-episode
        # return minus that task's own frozen specialist critic, used purely as a
        # BASELINE. A state-dependent baseline is unbiased however wrong it is, so
        # the specialist cannot skew the gradient -- unlike a bootstrap, which can.
        residual = cfg.value_mode == "residual"
        if residual:
            if local_policy is None and cfg.residual_baseline == "expert":
                raise ValueError("value_mode='residual' needs the frozen specialist")
            if (cfg.residual_baseline == "expert" and past_tasks
                    and (past_experts is None
                         or len(past_experts) != len(past_tasks))):
                raise ValueError(
                    "value_mode='residual' needs one frozen expert per past task; got "
                    f"{0 if past_experts is None else len(past_experts)} for "
                    f"{len(past_tasks)} past tasks")
            lam = cfg.global_gae_lambda or cfg.gae_lambda
            if lam != 1.0 and cfg.residual_scalar != "mc":
                raise ValueError(
                    "lambda<1 makes the specialist a bootstrap, so mean(G_0) is no "
                    "longer the value gap. Set residual_scalar='mc' to measure the "
                    "constraint separately, or global_gae_lambda=1.0.")
        self._past_retention: list[float] = []
        roll_envs = cfg.global_n_envs or cfg.n_envs
        # "global" baseline -> pass no external critic, so the collector uses the
        # policy's own and optimize_batches trains it as usual.
        use_expert = residual and cfg.residual_baseline == "expert"
        cur_critic = local_policy if use_expert else None
        past_critics = (list(past_experts) if use_expert and past_tasks
                        else [None] * len(past_tasks))
        # Whole episodes from the start distribution, so t=0 really is a start
        # state and the episode closes inside the window (roll_steps > the cap).
        roll_steps = cfg.global_n_steps or cfg.n_steps
        roll_lambda = cfg.global_gae_lambda or cfg.gae_lambda
        # One persistent collector per task (current + all past): replay-free
        # fresh rollouts in the old environments.
        past_collectors = [
            RolloutCollector(t, roll_envs, roll_steps, self.device, seed + 101 + i,
                             async_mode=cfg.async_envs)
            for i, t in enumerate(past_tasks)
        ]
        cur_collector = RolloutCollector(
            task_k, roll_envs, roll_steps, self.device, seed + 7,
            async_mode=cfg.async_envs,
        )
        mu = mu_ctrl.value
        # Start-state returns pooled since the last dual update. Each iteration
        # already collects them for free, and the constraint needs a precise
        # number: at 64 envs one rollout has SD ~0.006, five pooled ~0.003, which
        # is what makes a budget of sqrt(eps)=0.01 measurable rather than noise.
        start_returns: list[float] = []
        shortfall = 0.0
        constraint = 0.0
        v_k_g = float("nan")
        s_k_g = float("nan")  # global's raw (undiscounted) score, same rollouts as V_k_g
        thr = float(getattr(task_k, "threshold", float("inf")))
        met = 0
        t0, early = time.time(), False
        try:
            for it in range(num_iters):
                if cfg.past_task_sampling == "worst" and past_collectors:
                    # Spend the budget where the policy is losing. Scores come
                    # from the retention check, which already evaluates every seen
                    # task, so selection is free. Before the first check there are
                    # none, so fall back to uniform.
                    n_draw = max(1, min(cfg.past_task_sample_n, len(past_collectors)))
                    if len(self._past_retention) == len(past_collectors):
                        order = sorted(range(len(past_collectors)),
                                       key=lambda j: self._past_retention[j])
                        picks = order[:n_draw]
                    else:
                        picks = torch.randperm(len(past_collectors))[:n_draw].tolist()
                    # By default NO (k-1)/n rescaling: picking the worst is a biased
                    # choice on purpose, so pretending it estimates the full sum
                    # would be a lie about what the gradient is. With
                    # ppo.past_full_rescale the n drawn tasks are instead scaled to
                    # STAND IN for all k-1 past tasks, so "the past" keeps a weight
                    # that does not decay as 1/k (anti-starvation).
                    past_batches = [past_collectors[j].collect(
                        global_policy, roll_lambda, critic=past_critics[j],
                        reset=residual) for j in picks]
                    _rescale = (len(past_collectors) / len(picks)
                                if cfg.past_full_rescale and picks else 1.0)
                    past_coeffs = [omega[j] * _rescale for j in picks]
                elif cfg.past_task_sampling == "sample" and past_collectors:
                    # Minibatch over past tasks: draw n at random per iteration and
                    # rescale each by (k-1)/n, so the estimate of the full past-task
                    # sum stays unbiased at any n. Larger n trades compute for
                    # variance, which matters because a single task standing in for
                    # forty-nine is a violent estimator.
                    n_draw = max(1, min(cfg.past_task_sample_n, len(past_collectors)))
                    picks = torch.randperm(len(past_collectors))[:n_draw].tolist()
                    scale = len(past_collectors) / n_draw
                    past_batches = [past_collectors[j].collect(
                        global_policy, roll_lambda, critic=past_critics[j],
                        reset=residual) for j in picks]
                    past_coeffs = [omega[j] * scale for j in picks]
                else:
                    past_batches = [c.collect(global_policy, roll_lambda,
                                              critic=past_critics[i], reset=residual)
                                    for i, c in enumerate(past_collectors)]
                    past_coeffs = list(omega[: len(past_tasks)])
                cur_batch = cur_collector.collect(global_policy, roll_lambda,
                                                  critic=cur_critic, reset=residual)
                if residual:
                    start_returns.append(cur_batch.start_return)

                # Refresh V_k^G and mu every constraint_every iters (slow dual
                # timescale; between refreshes the coeff below is held fixed).
                # V_k^G is a fresh on-policy Monte-Carlo estimate; the one-sided
                # squared hinge F_k = [V_k^L - V_k^G]_+^2 (objective doc eq 26) is
                # compared against the tolerance eps and mu follows the projected
                # dual ascent mu <- [mu + eta (F_k - eps)]_+ (eq 47).
                if it % max(1, cfg.constraint_every) == 0:
                    # MC estimate (always computed): V_k^G on the training-reward scale.
                    _t = time.perf_counter()
                    v_k_g, s_k_g, _, _ = evaluate_value_and_score(
                        global_policy, task_k, cfg.constraint_episodes,
                        cfg.n_envs, self.device, seed=cfg.eval_seed,
                        greedy=cfg.constraint_greedy,
                        noop_enumerate=cfg.eval_noop_enumerate,
                        max_ep_steps=cfg.eval_max_ep_steps)
                    t_mc_s = time.perf_counter() - _t  # wall time of the MC estimator
                    # Windowed-bootstrap estimate (always computed when a local critic
                    # is available) -> the boot-vs-MC bias diagnostic.
                    vk_l_boot = vk_g_boot = float("nan")
                    t_boot_s = float("nan")
                    if local_policy is not None:
                        _t = time.perf_counter()
                        vk_l_boot, vk_g_boot = self._windowed_value_gap(
                            global_policy, local_policy, task_k, task_k.spec.task_id,
                            cfg.window_H, cfg.n_envs, cfg.eval_seed, cfg.constraint_greedy)
                        t_boot_s = time.perf_counter() - _t  # wall time of the H-step bootstrap estimator
                    # The constraint uses whichever estimator value_mode selects.
                    if residual and cfg.residual_scalar == "mc":
                        # Gradient is bootstrapped, constraint is not. mean(G_0)
                        # would read only (1-gamma^H) of the gap once the window
                        # stops covering whole episodes, so measure it separately.
                        v_k_start = float("nan")
                        shortfall = max(0.0, ref_current - v_k_g)
                        start_returns.clear()
                    elif residual:
                        # The gap between two Monte-Carlo values, no critic in
                        # either. ref_current is the specialist's measured return,
                        # taken once when it was frozen. cur_batch.start_return is
                        # the global policy's own return from start states, free
                        # from the rollout already collected for the gradient.
                        # Exact, unattenuated, and one estimator instead of two.
                        v_k_start = sum(start_returns) / len(start_returns)
                        shortfall = max(0.0, ref_current - v_k_start)
                        start_returns.clear()
                    elif cfg.value_mode == "bootstrap" and local_policy is not None:
                        shortfall = max(0.0, vk_l_boot - vk_g_boot)  # bootstrap V_L(S_0) - V_G
                    else:
                        shortfall = max(0.0, ref_current - v_k_g)   # MC [V_k^L - V_k^G]_+
                    constraint = shortfall * shortfall          # F_k (eq 26)
                    mu = mu_ctrl.update(constraint, eps)        # eq 47 (threshold eps)
                    _mu_refreshed = True
                    self.logger.log({
                        "phase": "boot_vs_mc", "task": current_task, "step": it,
                        "value_mode": cfg.value_mode, "window_H": cfg.window_H,
                        "V_k_ref_local_mc": ref_current, "V_k_global_mc": v_k_g,
                        "V_k_ref_local_boot": vk_l_boot, "V_k_global_boot": vk_g_boot,
                        "boot_minus_mc": (vk_g_boot - v_k_g),  # global-value estimator bias
                        "R_residual": shortfall if shortfall > 0 else
                                      (ref_current - v_k_start if residual else float("nan")),
                        "V_G_start_return": v_k_start if residual else float("nan"),
                        "n_start_pooled": len(start_returns) + 1 if residual else 0,
                        "mc_gap": ref_current - v_k_g,  # the identity says these agree
                        "shortfall": shortfall, "mu": mu,
                        "t_mc_s": t_mc_s, "t_boot_s": t_boot_s,  # per-check estimator wall time (the compute-saving)
                        "mc_env_steps": cfg.constraint_episodes * cfg.eval_max_ep_steps,  # hw-independent upper bound
                        "boot_env_steps": cfg.n_envs * cfg.window_H,
                    })
                else:
                    _mu_refreshed = False
                coeff_k = mu * 2.0 * shortfall  # differentiated hinge, eqs 38/40
                # Anti-starvation cap: guarantee the past collectively keeps at
                # least a fraction f of the (post-normalisation) update by bounding
                # the current-task coefficient at past_total*(1-f)/f. When the
                # constraint is satisfied c_k~0 and the past gets ~everything; when
                # it persists the current task can take at most (1-f) of the step.
                if cfg.past_fixed_fraction > 0.0 and past_coeffs:
                    _f = cfg.past_fixed_fraction
                    _past_total = sum(past_coeffs)
                    coeff_k = min(coeff_k, _past_total * (1.0 - _f) / _f)
                if clog is not None and _mu_refreshed:
                    # Dual record (contract §3). Global has only mu; the per-past
                    # lambda + past-shortfall + grad_share are local/diagnostic-only
                    # -> null here (a field a method cannot fill is null).
                    clog.dual(task_idx=current_task - 1, it=it, mu=float(mu), lam=None,
                              shortfall_current=float(shortfall), shortfall_past=None,
                              coeff_current=float(coeff_k), grad_share_past=None)

                if cfg.diagnostics and it % max(1, cfg.diag_every) == 0:
                    self._log_diagnostics(
                        global_policy, task_k, past_tasks, ref_current, mu, shortfall,
                        constraint, eps, coeff_k, v_k_g, past_batches, cur_batch,
                        omega, current_task, it, local_policy)

                streams = past_batches + [cur_batch]
                coeffs = past_coeffs + [coeff_k]
                bc = ((local_policy, self.ppo.global_bc_coef)
                      if local_policy is not None and self.ppo.global_bc_coef > 0 else None)
                stats = self.optimize_batches(global_policy, optimizer, streams, coeffs,
                                              bc=bc, kl_stop=False)  # full consolidation

                if probe is not None:
                    probe("global", current_task)
                # Early stop. Default: current game reached its greedy threshold.
                # V5 (retention_refs given): stop ONLY when EVERY seen task is >=
                # retention_frac of its local reference; else run to the cap.
                if retention_refs is not None:
                    gscore, all_ok = self._retention_check(
                        global_policy, past_tasks, task_k, retention_refs,
                        retention_frac, it + 1)
                    if gscore is not None:
                        met = met + 1 if all_ok else 0
                else:
                    gscore = self._stop_score(global_policy, task_k, it + 1)
                    if gscore is not None:
                        met = met + 1 if gscore >= thr else 0
                if it % self.log_every == 0 or gscore is not None:
                    self.logger.log(
                        {
                            "phase": "global", "task": current_task, "step": it,
                            "F_G": constraint, "mu": mu, "V_k_global": v_k_g,
                            "V_k_ref_local": ref_current, "coeff_k": coeff_k,
                            "score_k_global": s_k_g,
                            "greedy_score": gscore, **stats,
                        }
                    )
                    gs = f" greedy={gscore:.1f}/{thr:.0f}" if gscore is not None else ""
                    print(
                        f"[global k={current_task}] it={it:4d} "
                        f"Vk_G={v_k_g:.3f} refL={ref_current:.3f} "
                        f"F_G={constraint:.5f} mu={mu:.3f} pg={stats['pg_loss']:.3f}{gs}"
                    )
                if met >= self.ppo.patience:
                    early = True
                    reason = ("all seen tasks >= retention frac"
                              if retention_refs is not None
                              else f"greedy={gscore:.1f} >= thr={thr:.0f}")
                    print(f"[global k={current_task}] EARLY STOP it={it+1} ({reason})")
                    break
        finally:
            for c in past_collectors:
                c.close()
            cur_collector.close()
        return {"iters_run": it + 1, "early_stopped": early,
                "wall_s": round(time.time() - t0, 1), "final_greedy": gscore}
