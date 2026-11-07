"""Shared tabular likelihoods, planning, evaluation and run recording.

True transition kernels are confined to the evaluation driver. Algorithms
receive known rewards, declared prior parameters and unlabeled transitions.
"""

from dataclasses import asdict, dataclass
import csv
import hashlib
from numbers import Real
from pathlib import Path
import subprocess
import shutil
import time

import gymnasium as gym
import numpy as np
from scipy.special import logsumexp

import environments  # noqa: F401 -- register only tabular benchmarks


def draw_run_seeds(base_seed: int, names: tuple[str, ...]) -> dict[str, int]:
    """Draw 32-bit seeds with NumPy's post-setup seed first."""
    if "numpy" in names or len(set(names)) != len(names):
        raise ValueError("Seed names must be unique and exclude 'numpy'")
    rng = np.random.default_rng(base_seed)
    values = rng.integers(0, 2**32, size=len(names) + 1, dtype=np.uint32)
    seeds = dict(zip(("numpy", *names), map(int, values), strict=True))
    return seeds


def true_model(env: gym.Env) -> dict[str, np.ndarray]:
    """Transitions (Z,S,A,S), shared rewards (S,A,S), initial laws and ρ."""
    components = env.unwrapped.envs
    rewards = np.stack([component.rewards for component in components])
    initial = np.stack([component.init_state_probs for component in components])
    if not ((rewards == rewards[0]).all() and (initial == initial[0]).all()):
        raise ValueError("Tabular methods assume shared rewards and initial law")
    return {
        "transitions": np.stack([c.transition_probs for c in components]),
        "rewards": rewards[0],
        "initial": initial,
        "task_probs": env.unwrapped.task_probs,
    }


def normalize_rows(counts: np.ndarray) -> np.ndarray:
    """Normalize the last axis; an empty row becomes uniform."""
    totals = counts.sum(axis=-1, keepdims=True)
    uniform = np.full_like(counts, 1.0 / counts.shape[-1])
    return np.where(totals > 0, counts / np.where(totals > 0, totals, 1.0), uniform)


def solve_qmdp(transitions, rewards, horizon):
    """Finite-horizon Q-values of each component: (H,Z,S,A).

    ``transitions`` is (Z,S,A,S) and ``rewards`` the expected reward (Z,S,A).
    """
    q = np.empty((horizon, *rewards.shape))
    values = np.zeros(transitions.shape[:2])
    for step in reversed(range(horizon)):
        q[step] = rewards + np.einsum("zsap,zp->zsa", transitions, values)
        values = q[step].max(axis=-1)
    return q


def greedy_table(q, rng):
    """Greedy action per (step, state), with ties broken uniformly: (H,S)."""
    noise = rng.random(q.shape)
    return np.where(q == q.max(axis=-1, keepdims=True), noise, -1.0).argmax(axis=-1)


class QMDPPolicy:
    """Act greedily on belief-averaged Q-values; filter the belief by Bayes."""

    def __init__(self, q, transitions, mixture):
        self.q = q
        self.log_transitions = np.log(np.maximum(transitions, 1e-300))
        self.log_prior = np.log(np.maximum(mixture, 1e-32))

    def start(self, states):
        self.log_belief = np.tile(self.log_prior, (len(states), 1))
        self.step = 0

    def belief(self):
        log_belief = self.log_belief - logsumexp(self.log_belief, axis=1, keepdims=True)
        return np.exp(log_belief)

    def act(self, states):
        values = np.einsum("zba,bz->ba", self.q[self.step][:, states], self.belief())
        return values.argmax(axis=1)

    def observe(self, states, actions, next_states):
        self.log_belief += self.log_transitions[:, states, actions, next_states].T
        self.log_belief -= self.log_belief.max(axis=1, keepdims=True)
        self.step += 1


class ActionTable:
    """A Markov policy given as one action per (step, state)."""

    def __init__(self, table):
        self.table = table

    def start(self, states):
        self.step = 0

    def act(self, states):
        return self.table[self.step, states]

    def observe(self, states, actions, next_states):
        self.step += 1


class RandomPolicy:
    """Uniform actions from the supplied generator."""

    def __init__(self, num_actions, rng):
        self.num_actions, self.rng = num_actions, rng

    def start(self, states):
        pass

    def act(self, states):
        return self.rng.integers(self.num_actions, size=len(states))

    def observe(self, states, actions, next_states):
        pass


def evaluate(policy, model, *, horizon, episodes, rng) -> dict:
    """Run ``episodes`` episodes on every component; ρ-weighted mixture stats."""
    transitions, rewards = model["transitions"], model["rewards"]
    num_latents, num_states = transitions.shape[:2]
    cumulative = np.cumsum(transitions, axis=-1)
    returns = np.zeros((num_latents, episodes))
    for latent in range(num_latents):
        states = rng.choice(num_states, size=episodes, p=model["initial"][latent])
        policy.start(states)
        for _ in range(horizon):
            actions = policy.act(states)
            draws = rng.random(episodes)
            rows = cumulative[latent, states, actions]
            successors = np.minimum((rows < draws[:, None]).sum(axis=1), num_states - 1)
            returns[latent] += rewards[states, actions, successors]
            policy.observe(states, actions, successors)
            states = successors
    probs = model["task_probs"]
    means, variances = returns.mean(axis=1), returns.var(axis=1, ddof=1)
    mean = float(probs @ means)
    row = {
        "eval/return_mean": mean,
        "eval/return_std": float(np.sqrt(probs @ (variances + (means - mean) ** 2))),
    }
    for latent in range(num_latents):
        row[f"eval/z{latent}/return_mean"] = float(means[latent])
        row[f"eval/z{latent}/return_std"] = float(np.sqrt(variances[latent]))
    return row


class LatentModel:
    """Belief-weighted transition counts of Z components and their mixture.

    ``prior_alpha`` is the Dirichlet concentration added to the counts to form
    the transition estimate (0 gives the maximum-likelihood estimate).
    ``smoothing`` is the affine smoothing α of the L-UCRL E-step, whose
    factors are α + (1 − 2αS)·T; ``None`` uses the model likelihood.
    """

    def __init__(self, num_latents, num_states, num_actions, *, prior_alpha, smoothing):
        self.prior_alpha, self.smoothing = prior_alpha, smoothing
        shape = (num_latents, num_states, num_actions, num_states)
        self.counts = np.zeros(shape)
        self.mass = np.ones(num_latents)

    def transitions(self):
        return normalize_rows(self.counts + self.prior_alpha)

    def mixture(self):
        return self.mass / self.mass.sum()

    def log_factors(self, transitions):
        if self.smoothing is None:
            return np.log(np.maximum(transitions, 1e-300))
        states = transitions.shape[-1]
        smoothed = self.smoothing + (1 - 2 * self.smoothing * states) * transitions
        return np.log(np.maximum(smoothed, 1e-300))

    def _log_joint(self, flat, transitions, mixture):
        """Per-episode, per-restart, per-component log scores: (L,R,Z)."""
        restarts, num_latents = mixture.shape
        factors = self.log_factors(transitions).reshape(restarts * num_latents, -1)
        scores = (flat @ factors.T).reshape(len(flat), restarts, num_latents)
        return scores + np.log(np.maximum(mixture, 1e-32))

    def posterior(self, episode_counts):
        """E-step of each episode under the current estimate: (L,Z)."""
        flat = episode_counts.reshape(len(episode_counts), -1)
        scores = self._log_joint(flat, self.transitions()[None], self.mixture()[None])
        scores = scores[:, 0]
        return np.exp(scores - logsumexp(scores, axis=1, keepdims=True))

    def fit(self, episode_counts, rng, *, restarts, max_iters, tol, init_concentration):
        """Full EM on all episodes; keep the best restart. Returns beliefs (L,Z)."""
        num_latents = len(self.mass)
        flat = episode_counts.reshape(len(episode_counts), -1).astype(float)
        num_states, num_actions = self.counts.shape[1:3]
        transitions = rng.dirichlet(
            np.full(num_states, init_concentration),
            size=(restarts, num_latents, num_states, num_actions),
        )
        mixture = np.full((restarts, num_latents), 1.0 / num_latents)
        previous = None
        for _ in range(max_iters):
            joint = self._log_joint(flat, transitions, mixture)
            total = logsumexp(joint, axis=-1)
            beliefs = np.exp(joint - total[..., None])
            mixture = beliefs.mean(axis=0)
            weighted = beliefs.reshape(len(flat), -1).T @ flat
            transitions = normalize_rows(
                weighted.reshape(transitions.shape) + self.prior_alpha
            )
            current = total.mean(axis=0)
            if previous is not None and np.max(np.abs(current - previous)) < tol:
                break
            previous = current
        joint = self._log_joint(flat, transitions, mixture)
        best = int(np.argmax(logsumexp(joint, axis=-1).mean(axis=0)))
        scores = joint[:, best]
        beliefs = np.exp(scores - logsumexp(scores, axis=1, keepdims=True))
        self.counts = (beliefs.T @ flat).reshape(self.counts.shape)
        self.mass = beliefs.sum(axis=0)
        return beliefs

    def add(self, episode_counts, belief):
        self.counts += belief[:, None, None, None] * episode_counts
        self.mass += belief


def run_episode(env, policy, horizon, seed=None):
    """One training episode; returns its transition counts and return."""
    state, info = env.reset(seed=seed)
    latent = info["latent_idx"]
    num_states, num_actions = env.observation_space.n, env.action_space.n
    counts = np.zeros((num_states, num_actions, num_states), dtype=np.int32)
    total = 0.0
    policy.start(np.array([state]))
    for step in range(horizon):
        action = int(policy.act(np.array([state]))[0])
        successor, reward, terminated, truncated, info = env.step(action)
        # The label is checked here and never reaches the learner.
        assert info["latent_idx"] == latent and not terminated
        assert truncated == (step == horizon - 1)
        counts[state, action, successor] += 1
        total += reward
        policy.observe(np.array([state]), np.array([action]), np.array([successor]))
        state = successor
    return counts, total


def validate_run(args) -> None:
    """Validate common scientific arguments before creating run files."""
    if args.seed < 0 or args.total_episodes < 1 or args.eval_episodes < 2:
        raise ValueError(
            "seed must be nonnegative; episodes positive; eval_episodes >= 2"
        )
    if args.output is not None and args.output.suffix != ".csv":
        raise ValueError("output must be a CSV path")


@dataclass
class LoggingArgs:
    """Local CSV output and run identity."""

    output: Path | None = None
    """CSV output; default outputs/<method>-<environment>-s<seed>.csv."""
    run_id: str = ""
    """Run name; default derived from method, environment and run index."""


def source_revision() -> dict[str, str]:
    """Record the actual checkout revision, or unknown for a source archive."""
    root = Path(__file__).resolve().parent
    if shutil.which("git") is None:
        return {"source_commit": "unknown", "source_dirty": "unknown"}
    head = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    status = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "source_commit": head.stdout.strip() if head.returncode == 0 else "unknown",
        "source_dirty": (
            str(bool(status.stdout.strip())) if status.returncode == 0 else "unknown"
        ),
    }


class Recorder:
    """Write scalar metrics to CSV and resolved settings/seeds to plain text."""

    def __init__(self, args, seeds: dict[str, int], method: str):
        self.rows = []
        env_name = args.env_id.split("/")[-1]
        run_id = args.run_id or f"{method.replace(' ', '_')}-{env_name}-s{args.seed}"
        self.identity = {
            "method": method,
            "env_id": args.env_id,
            "seed": args.seed,
            "run_id": run_id,
        }
        self.path = args.output or Path("outputs") / f"{run_id}.csv"
        settings_path = self.path.with_suffix(".txt")
        if self.path.exists() or settings_path.exists():
            raise FileExistsError(f"Run output already exists: {self.path}")
        config = {**asdict(args), **self.identity, **source_revision()}
        config["top_level_seed_order"] = tuple(seeds)
        config.update({f"seed/{name}": value for name, value in seeds.items()})
        asset = environments.DATA / f"{env_name.removesuffix('-v0')}.npz"
        config["benchmark_sha256"] = hashlib.sha256(asset.read_bytes()).hexdigest()
        text = "\n".join(f"{name} = {value}" for name, value in config.items()) + "\n"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.path.open("x", newline="")
        with settings_path.open("x") as settings:
            settings.write(text)
        self.writer = csv.DictWriter(
            self.file,
            fieldnames=(
                *self.identity,
                "train/step",
                "first_episode",
                "metric",
                "value",
            ),
        )
        self.writer.writeheader()
        print(text, end="", flush=True)
        self.start = time.perf_counter()

    def _record(self, row: dict, *, step: int, first_episode: int | None = None):
        values = {key: value for key, value in row.items() if isinstance(value, Real)}
        for key, value in values.items():
            self.writer.writerow(
                {
                    **self.identity,
                    "train/step": step,
                    "first_episode": first_episode,
                    "metric": key,
                    "value": value,
                }
            )
        self.file.flush()

    def evaluation(self, row: dict, *, first_episode: int, step: int):
        row = {**row, "eval/first_episode": first_episode, "train/step": step}
        self.rows.append(row)
        self._record(row, step=step, first_episode=first_episode)
        print(
            f"episode={first_episode - 1} return={row['eval/return_mean']:.3f} "
            f"time={time.perf_counter() - self.start:.1f}s",
            flush=True,
        )

    def log(self, row: dict, *, step: int):
        self._record(row, step=step)

    def finish(self):
        self.file.close()
        return self.rows
