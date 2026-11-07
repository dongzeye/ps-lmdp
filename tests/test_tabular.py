"""Regression checks for trajectory incorporation at latent learning checkpoints."""

import csv
from types import SimpleNamespace

import numpy as np
import pytest

import latent_psrl
import latent_ucrl


class RecordingAgent:
    """A full refit replaces counts; an online update adds one episode."""

    def __init__(self):
        self.episode = 0
        self.calls = []
        self.incorporated = []
        self.plans = []

    def fit(self, counts):
        episodes = counts[:, 0, 0, 0].astype(int).tolist()
        self.calls.append((self.episode, "fit", tuple(episodes)))
        self.incorporated = episodes.copy()
        probability = np.array(episodes) / (np.array(episodes) + 1)
        return np.column_stack((probability, 1 - probability))

    def update(self, counts):
        episode = int(counts[0, 0, 0])
        self.calls.append((self.episode, "update", (episode,)))
        self.incorporated.append(episode)
        return np.array([0.5, 0.5])

    def plan(self):
        self.plans.append((self.episode, tuple(self.incorporated)))
        return object()


@pytest.fixture(params=[latent_psrl, latent_ucrl], ids=["L-PSRL", "L-UCRL"])
def recorded_run(request, monkeypatch, tmp_path):
    module = request.param
    agent = RecordingAgent()
    env = SimpleNamespace(spec=SimpleNamespace(max_episode_steps=1), close=lambda: None)
    model = {
        "transitions": np.ones((2, 1, 1, 1)),
        "rewards": np.zeros((1, 1, 1)),
    }

    def collect(env, policy, horizon, seed):
        agent.episode += 1
        # Each transition-count tensor carries a unique episode identifier.
        return np.full((1, 1, 1), agent.episode), 0.0

    monkeypatch.setattr(module.gym, "make", lambda env_id: env)
    monkeypatch.setattr(module, "true_model", lambda env: model)
    monkeypatch.setattr(module, "run_episode", collect)
    monkeypatch.setattr(
        module, "evaluate", lambda *args, **kwargs: {"eval/return_mean": 0.0}
    )
    agent_name = "LatentPSRL" if module is latent_psrl else "LatentUCRL"
    monkeypatch.setattr(module, agent_name, lambda *args: agent)

    def run(*, cutoff=20, warmup=10, interval=10, total=40):
        rows = module.main(
            module.Args(
                total_episodes=total,
                warmup_episodes=warmup,
                update_every=interval,
                full_em_until=cutoff,
                eval_episodes=2,
                output=tmp_path / "recorded.csv",
            )
        )
        return agent, rows

    return run


def expected_calls(*, full_checkpoints, online_checkpoints, last_full=0):
    calls = [
        (checkpoint, "fit", tuple(range(1, checkpoint + 1)))
        for checkpoint in full_checkpoints
    ]
    first_pending = last_full + 1
    for checkpoint in online_checkpoints:
        calls.extend(
            (checkpoint, "update", (episode,))
            for episode in range(first_pending, checkpoint + 1)
        )
        first_pending = checkpoint + 1
    return calls


def assert_complete_checkpoints(agent, rows, checkpoints):
    assert agent.plans == [
        (checkpoint, tuple(range(1, checkpoint + 1))) for checkpoint in checkpoints
    ]
    assert [row["eval/first_episode"] for row in rows] == [
        1,
        *(checkpoint + 1 for checkpoint in checkpoints),
    ]


def test_aligned_cutoff_incorporates_every_episode_once(recorded_run):
    agent, rows = recorded_run(cutoff=20)
    assert agent.calls == expected_calls(
        full_checkpoints=[10, 20], online_checkpoints=[30, 40], last_full=20
    )
    assert_complete_checkpoints(agent, rows, [10, 20, 30, 40])


def test_unaligned_cutoff_carries_all_pending_episodes(recorded_run):
    agent, rows = recorded_run(cutoff=25)
    assert agent.calls == expected_calls(
        full_checkpoints=[10, 20], online_checkpoints=[30, 40], last_full=20
    )
    assert_complete_checkpoints(agent, rows, [10, 20, 30, 40])


@pytest.mark.parametrize("cutoff", [0, 5])
def test_early_cutoff_incorporates_all_warmup_episodes(recorded_run, cutoff):
    agent, rows = recorded_run(cutoff=cutoff)
    assert agent.calls == expected_calls(
        full_checkpoints=[], online_checkpoints=[10, 20, 30, 40]
    )
    assert_complete_checkpoints(agent, rows, [10, 20, 30, 40])


def test_main_settings_preserve_fit_and_update_sequence(recorded_run):
    agent, rows = recorded_run(cutoff=1000, warmup=100, interval=10, total=4000)
    full_checkpoints = list(range(100, 1001, 10))
    online_checkpoints = list(range(1010, 4001, 10))
    assert agent.calls == expected_calls(
        full_checkpoints=full_checkpoints,
        online_checkpoints=online_checkpoints,
        last_full=1000,
    )
    assert_complete_checkpoints(agent, rows, full_checkpoints + online_checkpoints)


def test_full_refit_logs_beliefs_for_the_pending_batch(recorded_run, tmp_path):
    agent, rows = recorded_run(cutoff=15, warmup=10, interval=3, total=15)
    assert_complete_checkpoints(agent, rows, [10, 13, 15])
    with (tmp_path / "recorded.csv").open() as file:
        belief_max = {
            int(row["train/step"]): float(row["value"])
            for row in csv.DictReader(file)
            if row["metric"] == "belief/max"
        }
    for checkpoint, pending in [
        (10, range(1, 11)),
        (13, range(11, 14)),
        (15, range(14, 16)),
    ]:
        episodes = np.array(pending)
        assert belief_max[checkpoint] == pytest.approx(
            np.mean(episodes / (episodes + 1))
        )
