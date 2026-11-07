"""Pooled PSRL: posterior sampling of one MDP, ignoring the latent structure.

uv run python psrl.py --env-id LMDP/DirichletRandoms-Z4-v0

Before every episode, sample each transition row from Dir(counts + α), plan
the finite-horizon greedy policy, evaluate it and act with it.
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
    normalize_rows,
    run_episode,
    solve_qmdp,
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
    """training episodes"""
    prior_alpha: float = 1.0
    """Dirichlet concentration of each transition row"""
    eval_episodes: int = 100
    """evaluation episodes per component"""


def main(args: Args):
    validate_run(args)
    if not np.isfinite(args.prior_alpha) or args.prior_alpha <= 0:
        raise ValueError("prior_alpha must be finite and positive")
    seeds = draw_run_seeds(args.seed, ("environment", "agent", "evaluation"))
    env = gym.make(args.env_id)
    model = true_model(env)
    horizon = env.spec.max_episode_steps
    rewards = model["rewards"]
    rng = np.random.default_rng(seeds["agent"])
    eval_rng = np.random.default_rng(seeds["evaluation"])
    recorder = Recorder(args, seeds, "PSRL")
    counts = np.zeros(rewards.shape)
    for episode in range(1, args.total_episodes + 1):
        transitions = normalize_rows(rng.gamma(counts + args.prior_alpha))
        expected = np.einsum("sap,sap->sa", transitions, rewards)
        q = solve_qmdp(transitions[None], expected[None], horizon)[:, 0]
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
