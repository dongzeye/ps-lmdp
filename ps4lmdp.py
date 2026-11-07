"""PS4LMDP: static-latent NUTS posterior sampling with finite-horizon QMDP.

uv run --extra ps4 python ps4lmdp.py --env-id LMDP/DirichletRandoms-Z2-v0

The observation map and shared initial law are known. The latent mixture
and each component's transitions are learned from unlabeled trajectories.
"""

from dataclasses import dataclass
import time

import gymnasium as gym
import numpy as np
from scipy.special import logsumexp
import tyro

from common import LoggingArgs, Recorder, draw_run_seeds, validate_run


@dataclass
class Args(LoggingArgs):
    env_id: str = "LMDP/DirichletRandoms-Z2-v0"
    seed: int = 1
    """Run index and base seed."""
    total_episodes: int = 4000
    warmup_episodes: int = 100
    update_every: int = 10
    eval_episodes: int = 100
    """Evaluation episodes per component."""
    prior_alpha: float = 1.0
    """Positive scalar Dirichlet concentration per successor."""
    mixture_prior_alpha: float = 1.0
    mcmc_num_warmup: int = 200
    mcmc_num_samples: int = 200
    mcmc_num_chains: int = 1
    warm_start: bool = True
    jax_clear_caches_every: int = 200


def static_lmdp_model(
    obs,
    actions,
    n_states,
    n_actions,
    n_obs,
    init_state_probs,
    t_alphas=1.0,
    rho_alphas=1.0,
):
    """Marginalize one episode-static latent; infer transitions and mixture."""
    import jax.numpy as jnp
    from jax.scipy.special import logsumexp as jax_logsumexp
    import numpyro
    import numpyro.distributions as dist

    num_latents = n_states // n_obs
    alphas = jnp.broadcast_to(
        jnp.asarray(t_alphas, dtype=float),
        (num_latents, n_obs, n_actions, n_obs),
    )
    transition_kernel = numpyro.sample("transition_kernel", dist.Dirichlet(alphas))
    initial = jnp.asarray(init_state_probs).reshape(num_latents, n_obs)
    rho = numpyro.sample("rho", dist.Dirichlet(jnp.full(num_latents, rho_alphas)))
    initial = rho[:, None] * initial.sum(axis=0)[None, :]
    initial_prob = initial[:, obs[:, 0]]
    transition_log = jnp.log(transition_kernel[:, obs[:, :-1], actions, obs[:, 1:]])
    latent_log_likelihood = jnp.log(initial_prob) + jnp.sum(transition_log, axis=-1)
    numpyro.factor("obs_loglik", jnp.sum(jax_logsumexp(latent_log_likelihood, axis=0)))


@dataclass(frozen=True)
class KnownModel:
    hidden_states: int
    observations: int
    actions: int
    initial: np.ndarray
    rewards: np.ndarray
    mixture: np.ndarray


def known_lmdp_model(env: gym.Env) -> KnownModel:
    """Read shared known rewards/initial law; represent unknown rho uniformly."""
    components = env.unwrapped.envs
    states, actions = env.observation_space.n, env.action_space.n
    initial = np.asarray(components[0].init_state_probs, dtype=float)
    rewards = np.asarray(components[0].rewards, dtype=float)
    if any(
        not np.allclose(initial, c.init_state_probs)
        or not np.allclose(rewards, c.rewards)
        for c in components[1:]
    ):
        raise ValueError("PS4LMDP requires shared known initial law and rewards")
    mixture = np.full(len(components), 1.0 / len(components))
    hidden = len(components) * states
    expanded_rewards = np.empty((hidden, actions, hidden))
    for source in range(hidden):
        expanded_rewards[source] = rewards[source % states][
            :, np.arange(hidden) % states
        ]
    return KnownModel(
        hidden, states, actions, np.kron(mixture, initial), expanded_rewards, mixture
    )


@dataclass(frozen=True)
class Episode:
    observations: np.ndarray
    actions: np.ndarray


class NUTSPosterior:
    def __init__(
        self,
        known: KnownModel,
        rng: np.random.Generator,
        *,
        alphas: float | np.ndarray,
        num_warmup: int,
        num_samples: int,
        num_chains: int,
        warm_start: bool,
        mixture_prior_alpha: float,
    ):
        self.known, self.rng, self.alphas = known, rng, alphas
        self.num_warmup, self.num_samples, self.num_chains = (
            num_warmup,
            num_samples,
            num_chains,
        )
        self.warm_start, self.mixture_prior_alpha = warm_start, mixture_prior_alpha
        self.episodes: list[Episode] = []
        self.previous: dict[str, np.ndarray] | None = None
        self.seconds = 0.0
        self.divergences = 0
        self.accept_prob_mean = float("nan")
        self.num_steps_mean = float("nan")

    def add(self, episode: Episode) -> None:
        obs = np.asarray(episode.observations, dtype=np.int32)
        actions = np.asarray(episode.actions, dtype=np.int32)
        if (
            obs.ndim != 1
            or actions.ndim != 1
            or len(obs) != len(actions) + 1
            or (self.episodes and len(actions) != len(self.episodes[0].actions))
            or (obs < 0).any()
            or (obs >= self.known.observations).any()
            or (actions < 0).any()
            or (actions >= self.known.actions).any()
        ):
            raise ValueError("Invalid fixed-horizon observed trajectory")
        self.episodes.append(Episode(obs, actions))

    def draw(self) -> dict[str, np.ndarray]:
        if not self.episodes:
            raise ValueError("NUTS needs observed trajectories")
        import jax

        jax.config.update("jax_platform_name", "cpu")
        jax.config.update("jax_enable_x64", True)
        from numpyro.infer import MCMC, NUTS
        from numpyro.infer.initialization import init_to_value

        started = time.perf_counter()
        values = self.previous if self.warm_start else None
        kernel = (
            NUTS(static_lmdp_model, init_strategy=init_to_value(values=values))
            if values is not None
            else NUTS(static_lmdp_model)
        )
        sampler = MCMC(
            kernel,
            num_warmup=self.num_warmup,
            num_samples=self.num_samples,
            num_chains=self.num_chains,
            chain_method="sequential",
            progress_bar=False,
        )
        sampler.run(
            jax.random.PRNGKey(int(self.rng.integers(0, 2**31))),
            obs=np.stack([item.observations for item in self.episodes]),
            actions=np.stack([item.actions for item in self.episodes]),
            n_states=self.known.hidden_states,
            n_actions=self.known.actions,
            n_obs=self.known.observations,
            init_state_probs=self.known.initial,
            t_alphas=self.alphas,
            rho_alphas=self.mixture_prior_alpha,
            extra_fields=("diverging", "accept_prob", "num_steps"),
        )
        samples = sampler.get_samples()
        jax.block_until_ready(samples)
        self.previous = {key: np.asarray(value[-1]) for key, value in samples.items()}
        if any(not np.isfinite(value).all() for value in self.previous.values()):
            raise FloatingPointError("NUTS returned a nonfinite posterior sample")
        diagnostics = {
            key: np.asarray(value) for key, value in sampler.get_extra_fields().items()
        }
        self.divergences = int(np.count_nonzero(diagnostics["diverging"]))
        self.accept_prob_mean = float(np.mean(diagnostics["accept_prob"]))
        self.num_steps_mean = float(np.mean(diagnostics["num_steps"]))
        self.seconds = time.perf_counter() - started
        return self.previous


@dataclass(frozen=True)
class SampledModel:
    transitions: np.ndarray
    observations: np.ndarray
    initial: np.ndarray
    rewards: np.ndarray

    @property
    def num_hidden_states(self) -> int:
        return self.transitions.shape[0]


def expand_transition(transition: np.ndarray) -> np.ndarray:
    """Embed component kernels as a block-diagonal hidden-state transition."""
    num_latents, states, actions, _ = transition.shape
    expanded = np.zeros((num_latents * states, actions, num_latents * states))
    for latent in range(num_latents):
        block = slice(latent * states, (latent + 1) * states)
        expanded[block, :, block] = transition[latent]
    return expanded


def qmdp_backward(
    transitions: np.ndarray, rewards: np.ndarray, horizon: int
) -> np.ndarray:
    """Preserve the dense hidden-state arithmetic used by the PS4 baseline."""
    states, actions, _ = transitions.shape
    q = np.empty((horizon, states, actions), dtype=np.float64)
    values = np.zeros(states, dtype=np.float64)
    for step in range(horizon - 1, -1, -1):
        future = transitions @ values
        q[step] = rewards + future
        values = q[step].max(axis=1)
    return q


class QMDPPolicy:
    """Causal sampled-model filtering and finite-horizon action selection."""

    def __init__(self, model: SampledModel, q_values: np.ndarray):
        self.model, self.q_values = model, q_values
        with np.errstate(divide="ignore"):
            self.log_transitions = np.log(model.transitions)
            self.log_observations = np.log(model.observations)
            self.log_initial = np.log(model.initial)
        self.reset()

    def reset(self) -> None:
        self.log_belief = self.log_initial.copy()
        self.previous_action = None
        self.timestep = 0

    def __call__(self, observation: int) -> int:
        if self.previous_action is not None:
            self.log_belief = logsumexp(
                self.log_belief[:, None]
                + self.log_transitions[:, self.previous_action, :],
                axis=0,
            )
        posterior = self.log_belief + self.log_observations[:, int(observation)]
        normalizer = logsumexp(posterior)
        if not np.isfinite(normalizer):
            raise ValueError("Observation is impossible under the sampled model")
        self.log_belief = posterior - normalizer
        horizon_index = min(self.timestep, len(self.q_values) - 1)
        action = int(np.argmax(np.exp(self.log_belief) @ self.q_values[horizon_index]))
        self.previous_action = action
        self.timestep += 1
        return action


class PS4LMDP:
    def __init__(self, known: KnownModel, posterior: NUTSPosterior):
        self.known, self.posterior, self.policy = known, posterior, None
        self.inference_seconds = 0.0
        self.inference_total = 0.0
        self.planning_seconds = 0.0

    def observe(self, observations: np.ndarray, actions: np.ndarray) -> None:
        self.posterior.add(Episode(observations, actions))

    def replan(self, horizon: int) -> None:
        started = time.perf_counter()
        sample = self.posterior.draw()
        states = self.known.observations
        latents = self.known.hidden_states // states
        initial = self.known.initial.reshape(latents, states)
        model = SampledModel(
            expand_transition(sample["transition_kernel"]),
            np.tile(np.eye(states), (latents, 1)),
            (sample["rho"][:, None] * initial.sum(axis=0)[None, :]).reshape(-1),
            self.known.rewards,
        )
        self.inference_seconds = time.perf_counter() - started
        self.inference_total += self.inference_seconds
        planned = time.perf_counter()
        rewards = np.sum(model.transitions * model.rewards, axis=-1)
        self.policy = QMDPPolicy(
            model, qmdp_backward(model.transitions, rewards, horizon)
        )
        self.planning_seconds = time.perf_counter() - planned

    def reset(self) -> None:
        if self.policy is not None:
            self.policy.reset()

    def __call__(self, observation: int) -> int:
        if self.policy is None:
            raise RuntimeError("PS4LMDP needs a successful planning step")
        return self.policy(observation)


def evaluate_tabular(
    eval_env: gym.Env,
    act,
    *,
    eval_eps_per_latent: int,
    task_probs: np.ndarray,
    horizon: int,
    on_episode_start,
) -> dict:
    """Keep latest PS4's sequential episode order and component RNG ownership."""
    if eval_eps_per_latent < 2 or horizon < 1:
        raise ValueError("Need a positive horizon and at least two episodes per latent")
    components = eval_env.unwrapped.envs
    probs = np.asarray(task_probs, dtype=float)
    if probs.shape != (len(components),) or not np.isclose(probs.sum(), 1.0):
        raise ValueError("Evaluation weights differ from the component count")
    returns = np.empty((len(components), eval_eps_per_latent), dtype=float)
    for index, component in enumerate(components):
        for episode in range(eval_eps_per_latent):
            on_episode_start()
            observation, info = component.reset()
            label = info.get("latent_idx", -1)
            total = 0.0
            for step in range(horizon):
                observation, reward, terminated, truncated, info = component.step(
                    act(observation)
                )
                total += float(reward)
                if (terminated or truncated) and step != horizon - 1:
                    raise RuntimeError("Evaluation component ended before the horizon")
                if info.get("latent_idx", label) != label:
                    raise RuntimeError("Latent label changed during evaluation")
            returns[index, episode] = total
    means = returns.mean(axis=1)
    variances = returns.var(axis=1, ddof=1)
    mean = float(probs @ means)
    metrics = {
        "eval/return_mean": mean,
        "eval/return_std": float(np.sqrt(probs @ (variances + (means - mean) ** 2))),
        "eval/return_se": float(
            np.sqrt(np.sum(probs**2 * variances / eval_eps_per_latent))
        ),
        "eval/worst_component_return": float(means.min()),
        "eval/eval_eps_per_latent": eval_eps_per_latent,
        "eval/task_probs": probs.tolist(),
        "component_returns": {f"z{i}": float(value) for i, value in enumerate(means)},
    }
    for index in range(len(components)):
        prefix = f"eval/z{index}"
        metrics.update(
            {
                f"{prefix}/return_mean": float(means[index]),
                f"{prefix}/return_std": float(np.sqrt(variances[index])),
                f"{prefix}/return_se": float(
                    np.sqrt(variances[index] / eval_eps_per_latent)
                ),
                f"{prefix}/task_prob": float(probs[index]),
            }
        )
    return metrics


def main(args: Args):
    validate_run(args)
    if not 1 <= args.warmup_episodes < args.total_episodes or args.update_every < 1:
        raise ValueError(
            "Warmup must leave training episodes; update_every must be positive"
        )
    if (
        min(
            args.mcmc_num_warmup,
            args.mcmc_num_samples,
            args.mcmc_num_chains,
            args.jax_clear_caches_every,
        )
        < 1
    ):
        raise ValueError("NUTS budgets and cache-clear interval must be positive")
    if not np.isfinite(args.mixture_prior_alpha) or args.mixture_prior_alpha <= 0:
        raise ValueError("mixture_prior_alpha must be finite and positive")
    if not np.isfinite(args.prior_alpha) or args.prior_alpha <= 0:
        raise ValueError("prior_alpha must be finite and positive")
    import jax

    seeds = draw_run_seeds(
        args.seed, ("environment", "evaluation", "posterior", "warmup_actions")
    )
    train_env = gym.make(args.env_id)
    eval_env = gym.make(args.env_id)
    horizon = train_env.spec.max_episode_steps
    train_env.unwrapped.seed_all(seeds["environment"])
    eval_env.unwrapped.seed_all(seeds["evaluation"])
    known = known_lmdp_model(train_env)
    posterior = NUTSPosterior(
        known,
        np.random.default_rng(seeds["posterior"]),
        alphas=args.prior_alpha,
        num_warmup=args.mcmc_num_warmup,
        num_samples=args.mcmc_num_samples,
        num_chains=args.mcmc_num_chains,
        warm_start=args.warm_start,
        mixture_prior_alpha=args.mixture_prior_alpha,
    )
    agent = PS4LMDP(known, posterior)
    warmup_rng = np.random.default_rng(seeds["warmup_actions"])
    recorder = Recorder(args, seeds, "PS4LMDP")
    mixture = eval_env.unwrapped.task_probs
    for episode in range(1, args.total_episodes + 1):
        observation, info = train_env.reset()
        label = info["latent_idx"]
        observations, actions = [int(observation)], []
        total = 0.0
        agent.reset()
        for step in range(horizon):
            action = (
                int(warmup_rng.integers(train_env.action_space.n))
                if episode <= args.warmup_episodes
                else agent(int(observation))
            )
            observation, reward, terminated, truncated, info = train_env.step(action)
            if (
                info["latent_idx"] != label
                or terminated
                or truncated != (step == horizon - 1)
            ):
                raise RuntimeError(
                    "Benchmark violated the episode-static, fixed-horizon protocol"
                )
            actions.append(action)
            observations.append(int(observation))
            total += float(reward)
        agent.observe(np.asarray(observations), np.asarray(actions))
        row = {"train/episode": episode, "train/episodic_return": total}
        if episode == args.warmup_episodes or (
            episode > args.warmup_episodes
            and (episode - args.warmup_episodes) % args.update_every == 0
        ):
            agent.replan(horizon)
            row.update(
                {
                    "inference/seconds": agent.inference_seconds,
                    "inference/cumulative_seconds": agent.inference_total,
                    "inference/nuts_seconds": posterior.seconds,
                    "planner/qmdp_seconds": agent.planning_seconds,
                    "posterior/divergences": posterior.divergences,
                    "posterior/accept_prob_mean": posterior.accept_prob_mean,
                    "posterior/num_steps_mean": posterior.num_steps_mean,
                }
            )
            metrics = evaluate_tabular(
                eval_env,
                agent,
                eval_eps_per_latent=args.eval_episodes,
                task_probs=mixture,
                horizon=horizon,
                on_episode_start=agent.reset,
            )
            recorder.evaluation(
                metrics, first_episode=episode + 1, step=episode * horizon
            )
        if episode % args.jax_clear_caches_every == 0:
            jax.clear_caches()
        recorder.log(row, step=episode * horizon)
    train_env.close()
    eval_env.close()
    return recorder.finish()


if __name__ == "__main__":
    main(tyro.cli(Args))
