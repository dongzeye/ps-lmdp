"""Numerical and CLI checks for the standalone algorithms."""

import csv
import hashlib
import importlib
import os
from pathlib import Path
import subprocess

import gymnasium as gym
import numpy as np
import pytest
import tyro

import common
import environments
import latent_psrl
import latent_ucrl
import ps4lmdp
import psrl
import ucbvi

ROOT = Path(__file__).resolve().parents[1]


def test_benchmark_hashes_and_seeded_streams():
    for name, digest in environments.SHA256.items():
        assert (
            hashlib.sha256((environments.DATA / f"{name}.npz").read_bytes()).hexdigest()
            == digest
        )
    left = gym.make("LMDP/FrozenLake-4x4-Z2-v0")
    right = gym.make("LMDP/FrozenLake-4x4-Z2-v0")
    for episode in range(3):
        seed = 17 if episode == 0 else None
        assert left.reset(seed=seed) == right.reset(seed=seed)
        for step in range(100):
            a = step % 4
            left_step, right_step = left.step(a), right.step(a)
            assert left_step == right_step
            assert left_step[2] is False and left_step[3] == (step == 99)
    left.close()
    right.close()


def test_qmdp_matches_dynamic_programming_and_filters_causally():
    rng = np.random.default_rng(3)
    transitions = rng.dirichlet(np.ones(3), size=(2, 3, 2))
    rewards = rng.random((2, 3, 2))
    q = common.solve_qmdp(transitions, rewards, 4)
    values = np.zeros((2, 3))
    for step in reversed(range(4)):
        expected = rewards + np.einsum("zsap,zp->zsa", transitions, values)
        np.testing.assert_array_equal(q[step], expected)
        values = expected.max(axis=-1)
    policy = common.QMDPPolicy(q, transitions, np.array([0.3, 0.7]))
    policy.start(np.array([0]))
    policy.observe(np.array([0]), np.array([1]), np.array([2]))
    expected = np.array([0.3, 0.7]) * transitions[:, 0, 1, 2]
    np.testing.assert_allclose(policy.belief()[0], expected / expected.sum())


@pytest.mark.parametrize("module", [latent_psrl, latent_ucrl])
def test_latent_fit_online_schedule_and_csv(module, tmp_path):
    output = tmp_path / f"{module.__name__}.csv"
    args = module.Args(
        total_episodes=12,
        warmup_episodes=4,
        full_em_until=8,
        update_every=2,
        em_restarts=2,
        em_max_iters=8,
        eval_episodes=3,
        output=output,
    )
    rows = module.main(args)
    assert [row["eval/first_episode"] for row in rows] == [1, 5, 7, 9, 11, 13]
    with output.open() as file:
        metrics = list(csv.DictReader(file))
    evaluated = [row for row in metrics if row["metric"] == "eval/return_mean"]
    np.testing.assert_array_equal(
        [float(row["value"]) for row in evaluated],
        [row["eval/return_mean"] for row in rows],
    )
    assert "seed/environment = " in output.with_suffix(".txt").read_text()
    with pytest.raises(FileExistsError):
        module.main(args)


@pytest.mark.parametrize("module", [psrl, ucbvi])
def test_pooled_evaluation_precedes_each_episode(module, tmp_path):
    rows = module.main(
        module.Args(
            total_episodes=3,
            eval_episodes=3,
            output=tmp_path / f"{module.__name__}.csv",
        )
    )
    assert [row["eval/first_episode"] for row in rows] == [1, 2, 3]


def test_invalid_schedule_creates_no_run_output(tmp_path):
    path = tmp_path / "invalid.csv"
    with pytest.raises(ValueError):
        latent_psrl.main(
            latent_psrl.Args(
                total_episodes=3, warmup_episodes=2, full_em_until=-1, output=path
            )
        )
    assert not path.exists()


def test_nuts_executes_real_posterior_and_warm_start():
    pytest.importorskip("numpyro")
    initial = np.array([0.5, 0, 0, 0.5, 0, 0])
    known = ps4lmdp.KnownModel(
        6, 3, 2, initial, np.zeros((6, 2, 6)), np.array([0.5, 0.5])
    )
    posterior = ps4lmdp.NUTSPosterior(
        known,
        np.random.default_rng(8),
        alphas=1.0,
        num_warmup=4,
        num_samples=4,
        num_chains=1,
        warm_start=True,
        mixture_prior_alpha=1.0,
    )
    posterior.add(ps4lmdp.Episode(np.array([0, 1, 2, 1]), np.array([0, 1, 0])))
    first = posterior.draw()
    posterior.add(ps4lmdp.Episode(np.array([0, 2, 1, 0]), np.array([1, 0, 1])))
    second = posterior.draw()
    for sample in (first, second):
        assert np.isfinite(sample["transition_kernel"]).all()
        np.testing.assert_allclose(sample["transition_kernel"].sum(-1), 1)
        np.testing.assert_allclose(sample["rho"].sum(), 1)
    assert not np.array_equal(first["transition_kernel"], second["transition_kernel"])


@pytest.mark.parametrize(
    "suite,count",
    [("main", 8), ("ps4lmdp", 2)],
)
def test_bash_suites_resolve_cli_flags_and_run_indices(suite, count, tmp_path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    capture = tmp_path / "arguments"
    fake_uv = fake_bin / "uv"
    fake_uv.write_text(
        '#!/usr/bin/env bash\nprintf "%s\\0" "$@" >> "$CAPTURE"\nprintf "\\0" >> "$CAPTURE"\n'
    )
    fake_uv.chmod(0o755)
    env = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "CAPTURE": str(capture),
        "RUNS": "2",
        "OUTPUT_DIR": str(tmp_path / "outputs with spaces"),
    }
    args = ["bash", str(ROOT / "scripts" / f"run_{suite}.sh")]
    if suite != "prior_ablation":
        args.append("DirichletRandoms-Z2")
    subprocess.run(args, env=env, check=True)
    calls = capture.read_bytes().split(b"\0\0")[:-1]
    assert len(calls) == count
    for call in calls:
        tokens = [item.decode() for item in call.split(b"\0")]
        index = tokens.index("python")
        module = importlib.import_module(Path(tokens[index + 1]).stem)
        resolved = tyro.cli(module.Args, args=tokens[index + 2 :])
        assert resolved.seed in (1, 2) and resolved.total_episodes == 4000
        assert resolved.env_id == "LMDP/DirichletRandoms-Z2-v0"
        assert resolved.eval_episodes == 100
        if module in (latent_psrl, latent_ucrl):
            assert resolved.em_max_iters == 100 and resolved.full_em_until == 1000
        if suite == "main" and module == latent_psrl:
            assert resolved.prior_alpha == 1.0


def test_source_archive_runs_without_git(monkeypatch):
    monkeypatch.setattr(common.shutil, "which", lambda executable: None)
    assert common.source_revision() == {
        "source_commit": "unknown",
        "source_dirty": "unknown",
    }
