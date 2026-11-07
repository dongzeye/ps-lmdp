"""Pooled UCBVI-CH (Azar et al., 2017), ignoring the latent structure.

uv run python ucbvi.py --env-id LMDP/DirichletRandoms-Z4-v0

Before every episode, back up optimistic values with the Chernoff-Hoeffding
bonus 7·H·L/√n, L = ln(5·S·A·H·K/δ), keep Q non-increasing across episodes,
evaluate the greedy policy and act with it. Rewards R(s,a,s') are known and
in [0, 1]; R(s,a) uses the empirical successor frequencies.
"""

from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import tyro

from common import draw_run_seeds
from common import (
    ActionTable,
    Recorder,
    LoggingArgs,
    evaluate,
    greedy_table,
    run_episode,
    true_model,
    validate_run,
)


@dataclass
class Args(LoggingArgs):
    env_id: str = "LMDP/DirichletRandoms-Z4-v0"
    """tabular latent MDP"""
    seed: int = 1
    """base seed of the run"""
    total_episodes: int = 4000
    """training episodes K"""
    delta: float = 0.05
    """confidence parameter δ"""
    bonus_multiplier: float = 1.0
    """multiplier of the bonus 7·H·L/√n"""
    eval_episodes: int = 100
    """evaluation episodes per component"""


def optimistic_backup(counts, rewards, previous_q, horizon, log_term, multiplier):
    """UCBVI-CH values (H,S,A); unvisited pairs keep the maximum H."""
    visits = counts.sum(axis=-1)
    visited = visits > 0
    safe = np.where(visited, visits, 1)
    expected_reward = np.sum(counts * rewards, axis=-1) / safe
    bonus = multiplier * 7.0 * horizon * log_term / np.sqrt(safe)
    q = np.empty_like(previous_q)
    values = np.zeros(counts.shape[0])
    for step in reversed(range(horizon)):
        target = expected_reward + counts @ values / safe + bonus
        target = np.minimum(np.minimum(previous_q[step], horizon), target)
        q[step] = np.where(visited, target, horizon)
        values = q[step].max(axis=-1)
    return q


def main(args: Args):
    validate_run(args)
    if (
        not 0 < args.delta < 1
        or not np.isfinite(args.bonus_multiplier)
        or args.bonus_multiplier < 0
    ):
        raise ValueError(
            "delta must be in (0, 1) and bonus_multiplier finite/nonnegative"
        )
    seeds = draw_run_seeds(args.seed, ("environment", "agent", "evaluation"))
    env = gym.make(args.env_id)
    model = true_model(env)
    horizon = env.spec.max_episode_steps
    rewards = model["rewards"]
    if rewards.min() < 0 or rewards.max() > 1:
        raise ValueError("UCBVI-CH assumes rewards in [0, 1]")
    num_states, num_actions = rewards.shape[:2]
    log_term = np.log(
        5 * num_states * num_actions * horizon * args.total_episodes / args.delta
    )
    rng = np.random.default_rng(seeds["agent"])
    eval_rng = np.random.default_rng(seeds["evaluation"])
    recorder = Recorder(args, seeds, "UCBVI")
    counts = np.zeros(rewards.shape)
    q = np.full((horizon, num_states, num_actions), float(horizon))
    for episode in range(1, args.total_episodes + 1):
        q = optimistic_backup(
            counts, rewards, q, horizon, log_term, args.bonus_multiplier
        )
        policy = ActionTable(greedy_table(q, rng))
        row = evaluate(
            policy, model, horizon=horizon, episodes=args.eval_episodes, rng=eval_rng
        )
        recorder.evaluation(row, first_episode=episode, step=(episode - 1) * horizon)
        seed = seeds["environment"] if episode == 1 else None
        episode_counts, total = run_episode(env, policy, horizon, seed)
        counts += episode_counts
        recorder.log({"train/episodic_return": total}, step=episode * horizon)
    env.close()
    return recorder.finish()


if __name__ == "__main__":
    main(tyro.cli(Args))
