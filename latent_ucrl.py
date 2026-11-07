"""L-UCRL: optimism over the components of a latent MDP (Kwon et al., 2021).

uv run python latent_ucrl.py --env-id LMDP/DirichletRandoms-Z4-v0

EM with the smoothed E-step of Algorithm 3 and maximum-likelihood M-steps
fits Z kernels and ρ. Each update plans QMDP on the estimated kernels with
the reward bonus range·H·min(1, β/√n(z,s,a)), β = c·√(5·S·ln(K/η)). The
initial law is known, so there is no initial-state bonus.
"""

from dataclasses import dataclass

import gymnasium as gym
import time

import numpy as np
import tyro

from common import (
    LatentModel,
    QMDPPolicy,
    LoggingArgs,
    solve_qmdp,
    draw_run_seeds,
    true_model,
    validate_run,
    Recorder,
    RandomPolicy,
    evaluate,
    run_episode,
)


@dataclass
class Args(LoggingArgs):
    env_id: str = "LMDP/DirichletRandoms-Z4-v0"
    """tabular latent MDP"""
    seed: int = 1
    """base seed of the run"""
    total_episodes: int = 4000
    """training episodes K, including warmup"""
    warmup_episodes: int = 100
    """uniform-action episodes before the first EM fit"""
    full_em_until: int = 1000
    """refit full EM on all episodes up to this episode; online EM afterwards"""
    update_every: int = 10
    """episodes between updates, replanning and evaluations"""
    bonus_scale: float = 0.001
    """multiplier c of the confidence radius β"""
    confidence_eta: float = 0.05
    """failure probability η in ln(K/η)"""
    belief_alpha: float = 0.001
    """E-step smoothing α of Algorithm 3; 0 <= α < 1/(2S)"""
    em_restarts: int = 20
    """random restarts of full EM; the best final likelihood is kept"""
    em_max_iters: int = 100
    """maximum EM iterations"""
    em_tol: float = 1e-5
    """stop when no restart's mean log-likelihood changes by more than this"""
    eval_episodes: int = 100
    """evaluation episodes per component"""


class LatentUCRL:
    def __init__(self, args, num_latents, rewards, horizon, rng):
        self.args, self.rewards, self.horizon, self.rng = args, rewards, horizon, rng
        num_states, num_actions = rewards.shape[:2]
        if not 0 <= args.belief_alpha < 1 / (2 * num_states):
            raise ValueError("belief_alpha must be in [0, 1/(2S))")
        self.model = LatentModel(
            num_latents,
            num_states,
            num_actions,
            prior_alpha=0.0,
            smoothing=args.belief_alpha,
        )
        log_term = np.log(args.total_episodes / args.confidence_eta)
        self.beta = args.bonus_scale * np.sqrt(5 * num_states * log_term)

    def fit(self, episode_counts):
        return self.model.fit(
            episode_counts,
            self.rng,
            restarts=self.args.em_restarts,
            max_iters=self.args.em_max_iters,
            tol=self.args.em_tol,
            init_concentration=1.0,
        )

    def update(self, counts):
        belief = self.model.posterior(counts[None])[0]
        self.model.add(counts, belief)
        return belief

    def plan(self):
        transitions = self.model.transitions()
        visits = np.maximum(self.model.counts.sum(axis=-1), 1.0)
        bonus = np.ptp(self.rewards) * self.horizon
        bonus = bonus * np.minimum(1.0, self.beta / np.sqrt(visits))
        expected = np.einsum("zsap,sap->zsa", transitions, self.rewards) + bonus
        q = solve_qmdp(transitions, expected, self.horizon)
        return QMDPPolicy(q, transitions, self.model.mixture())


def main(args: Args):
    """Warmup, full EM until ``full_em_until``, then online EM.

    Every ``update_every`` episodes the agent refits (full EM on all episodes)
    or folds the new episodes in (online E-step and count update), plans and
    is evaluated. The uniform warmup policy is evaluated once.
    """
    validate_run(args)
    if not 0 <= args.full_em_until <= args.total_episodes:
        raise ValueError("Require 0 <= full_em_until <= total_episodes")
    if not 1 <= args.warmup_episodes < args.total_episodes or args.update_every < 1:
        raise ValueError(
            "Warmup must leave training episodes and update_every must be positive"
        )
    if (
        args.em_restarts < 1
        or args.em_max_iters < 1
        or not np.isfinite(args.em_tol)
        or args.em_tol <= 0
    ):
        raise ValueError(
            "EM restarts/iters must be positive and em_tol finite/positive"
        )
    seeds = draw_run_seeds(args.seed, ("environment", "agent", "evaluation"))
    env = gym.make(args.env_id)
    model = true_model(env)
    horizon = env.spec.max_episode_steps
    num_latents, num_states, num_actions = model["transitions"].shape[:3]
    rng = np.random.default_rng(seeds["agent"])
    eval_rng = np.random.default_rng(seeds["evaluation"])
    agent = LatentUCRL(args, num_latents, model["rewards"], horizon, rng)
    recorder = Recorder(args, seeds, "L-UCRL")

    def evaluate_policy(policy, episode):
        row = evaluate(
            policy, model, horizon=horizon, episodes=args.eval_episodes, rng=eval_rng
        )
        recorder.evaluation(row, first_episode=episode + 1, step=episode * horizon)

    def log_beliefs(beliefs, episode):
        entropy = -np.sum(beliefs * np.log(np.maximum(beliefs, 1e-300)), axis=1)
        recorder.log(
            {
                "belief/entropy": float(entropy.mean()),
                "belief/max": float(beliefs.max(axis=1).mean()),
            },
            step=episode * horizon,
        )

    evaluate_policy(RandomPolicy(num_actions, eval_rng), 0)
    history, pending = [], []
    warmup = RandomPolicy(num_actions, rng)
    policy = warmup
    for episode in range(1, args.total_episodes + 1):
        seed = seeds["environment"] if episode == 1 else None
        counts, total = run_episode(env, policy, horizon, seed)
        recorder.log({"train/episodic_return": total}, step=episode * horizon)
        history.append(counts)
        pending.append(counts)
        since_warmup = episode - args.warmup_episodes
        if since_warmup < 0 or (
            since_warmup % args.update_every and episode != args.total_episodes
        ):
            continue
        started = time.perf_counter()
        if episode <= args.full_em_until:
            beliefs = agent.fit(np.stack(history))[-len(pending) :]
        else:
            beliefs = np.stack([agent.update(counts) for counts in pending])
        pending.clear()
        fitted = time.perf_counter()
        log_beliefs(beliefs, episode)
        policy = agent.plan()
        recorder.log(
            {
                "inference/fitting_seconds": fitted - started,
                "planner/seconds": time.perf_counter() - fitted,
            },
            step=episode * horizon,
        )
        evaluate_policy(policy, episode)
    env.close()
    return recorder.finish()


if __name__ == "__main__":
    main(tyro.cli(Args))
