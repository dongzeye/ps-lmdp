"""Check that default launch commands implement the manuscript's protocol."""

import importlib
import os
from pathlib import Path
import subprocess

import tyro

ROOT = Path(__file__).resolve().parents[1]


def capture_commands(tmp_path, script):
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
        "RUNS": "1",
        "OUTPUT_DIR": str(tmp_path / "outputs"),
    }
    subprocess.run(["bash", str(ROOT / "scripts" / script)], env=env, check=True)
    resolved = []
    for call in capture.read_bytes().split(b"\0\0")[:-1]:
        tokens = [item.decode() for item in call.split(b"\0")]
        assert tokens[:2] == ["run", "--no-sync"]
        index = tokens.index("python")
        module = importlib.import_module(Path(tokens[index + 1]).stem)
        args = tyro.cli(module.Args, args=tokens[index + 2 :])
        resolved.append((module.__name__, args))
    return resolved


def test_main_comparison_matches_paper(tmp_path):
    calls = capture_commands(tmp_path, "run_main.sh")
    assert len(calls) == 32
    assert len({args.env_id for _, args in calls}) == 8
    for name, args in calls:
        large = "FrozenLake-8x8" in args.env_id
        assert args.total_episodes == (16000 if large else 4000)
        assert args.seed == 1 and args.eval_episodes == 100
        if name in ("latent_psrl", "latent_ucrl"):
            assert args.full_em_until == (4000 if large else 1000)
            assert args.warmup_episodes == 100 and args.update_every == 10
            assert args.em_restarts == 20 and args.em_max_iters == 100
            assert args.em_tol == 1e-5
        if name in ("latent_psrl", "psrl"):
            assert args.prior_alpha == 1.0


def test_ps4_comparison_uses_completed_paper_cells(tmp_path):
    calls = capture_commands(tmp_path, "run_ps4lmdp.sh")
    assert {args.env_id for _, args in calls} == {
        "LMDP/DirichletRandoms-Z2-v0",
        "LMDP/DirichletRandoms-Z4-v0",
        "LMDP/FrozenLake-4x4-Z2-v0",
        "LMDP/FrozenLake-4x4-Z4-v0",
    }
    assert len(calls) == 4
    for name, args in calls:
        assert name == "ps4lmdp"
        assert args.mcmc_num_warmup == 200 and args.mcmc_num_samples == 200
        assert args.mcmc_num_chains == 1 and args.warm_start
        assert args.prior_alpha == 1.0 and args.mixture_prior_alpha == 1.0
        assert args.total_episodes == 4000 and args.eval_episodes == 100


def test_schedule_ablation_keeps_budget_and_prior_fixed(tmp_path):
    calls = capture_commands(tmp_path, "run_schedule_ablation.sh")
    assert len(calls) == 3
    assert {args.full_em_until for _, args in calls} == {0, 1000, 4000}
    assert len({args.output for _, args in calls}) == 3
    for name, args in calls:
        assert name == "latent_psrl"
        assert args.env_id == "LMDP/DirichletRandoms-Z2-v0"
        assert args.total_episodes == 4000 and args.warmup_episodes == 100
        assert args.prior_alpha == 1.0 and args.update_every == 10
