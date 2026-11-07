# Posterior sampling for latent MDPs

Code for **Efficient Online Learning for Latent MDPs via Posterior Sampling**,
by Dongze Ye, Ping Li, Haipeng Luo, Rahul Jain, and Ashutosh Nayyar.

The repository contains the five tabular learners, eight fixed benchmark
instances, paper settings, and focused tests. It uses NumPy and SciPy;
PS4LMDP additionally uses CPU JAX and NumPyro. No GPU is required.

## Setup

Install [uv](https://docs.astral.sh/uv/) and use Python 3.12 or 3.13.
From this directory:

```bash
uv sync --frozen
uv run --no-sync python latent_psrl.py --help
uv run --no-sync python latent_psrl.py --env-id LMDP/DirichletRandoms-Z2-v0 --seed 1
```

For PS4LMDP, include its optional dependencies once:

```bash
uv sync --frozen --extra ps4
uv run --no-sync python ps4lmdp.py --env-id LMDP/DirichletRandoms-Z2-v0 --seed 1
```

Dependencies are pinned in `uv.lock`. Use `uv sync` when changing the selected
extras; subsequent commands use `--no-sync`.

## Implementation

| File | Purpose |
| --- | --- |
| `latent_psrl.py` | L-PSRL: EM refitting, sequential belief-weighted counts, and Dirichlet model sampling. |
| `latent_ucrl.py` | L-UCRL: smoothed EM estimates and optimistic QMDP planning. |
| `ps4lmdp.py` | PS4LMDP: a static latent context, NUTS posterior sampling, and QMDP planning. |
| `psrl.py` | Pooled PSRL: posterior sampling of one MDP. |
| `ucbvi.py` | Pooled UCBVI-CH: optimistic finite-horizon planning. |
| `common.py` | Shared likelihood/count operations, planning, evaluation, seeds, and CSV output. |
| `environments.py` | The fixed Randoms and FrozenLake benchmarks. |

Each learner's training loop is in its runnable file. The two EM learners
share the same count and likelihood utilities; there is no trainer framework.
The implementation learns mixture weights and uses the known shared initial
law. The theoretical results impose different prior and information conditions,
as stated in the paper.

The active context is fixed within each episode and hidden from the learner.
True labels are checked only for protocol validation and are never policy or
inference inputs. The true model is used separately for evaluation.

## Paper experiments

The scripts use explicit arguments, run sequentially, and stop on failure.
They preserve existing outputs rather than overwriting them.

```bash
# Four lightweight methods on all eight instances.
bash scripts/run_main.sh

# PS4LMDP on its four completed paper settings.
# Run `uv sync --frozen --extra ps4` first.
bash scripts/run_ps4lmdp.sh

# Three L-PSRL update schedules on Randoms Z2.
bash scripts/run_schedule_ablation.sh
```

`RUNS` selects the number of independent runs, with seeds 1 through `RUNS`.
`OUTPUT_DIR` changes the output directory. The main and PS4LMDP scripts accept
environment names as arguments; for example:

```bash
RUNS=2 OUTPUT_DIR=outputs/randoms-z2 bash scripts/run_main.sh DirichletRandoms-Z2
```

The default scripts match these paper settings:

| Setting | Training episodes | EM refitting cutoff |
| --- | --- | --- |
| Randoms Z2/4/8/16; FrozenLake 4x4 Z2/4 | 4,000 | 1,000 |
| FrozenLake 8x8 Z2/4 | 16,000 | 4,000 |

All use horizon 100, ten independent runs, and 100 evaluation episodes per
context. Latent methods include 100 random-action warmup episodes and update
every ten episodes; pooled methods update before every episode. Posterior
sampling uses Dirichlet(1) transition priors. EM uses 20 restarts, at most 100
iterations, and tolerance 1e-5. L-UCRL uses bonus multiplier 0.001 and smoothing
0.001. PS4LMDP uses 200 NUTS warmup steps and 200 retained samples, one chain,
and warm starts. Its default script covers Randoms Z2/4 and FrozenLake 4x4 Z2/4.

The schedule ablation compares EM refitting throughout, EM through episode
1,000 followed by sequential updates, and sequential updates throughout.
All three retain the same 4,000-episode budget and warmup.

## Data and outputs

`data/benchmarks/` contains the eight fixed NPZ instances. Their SHA-256
digests are listed in `environments.py`; the tests check every digest.
Separate training runs measure learning randomness on these fixed instances.

Each run writes a long-form CSV and a neighboring text file containing the
effective arguments, run index, named seeds, benchmark digest, and Git revision.
The scripts also capture stdout and stderr. Latent evaluation rows identify
the first episode in which the evaluated policy acts; pooled policies are
evaluated before each training episode. Evaluation does not update training.

The paper's empirical curves sum estimated value gaps to the true-model QMDP
policy. This reference is not the optimal hidden-context policy. Evaluation
episodes are not independent training runs; comparisons across methods are
unpaired. Recorded paper outputs, the 100,000-episode QMDP reference estimates,
and the analysis/figure-generation pipeline are not included in this release.
The source and settings allow new runs; they do not promise bitwise recovery
of historical results or hardware timings.

## Tests and attribution

```bash
uv sync --frozen --extra dev --extra ps4
uv run --no-sync pytest -q
uv run --no-sync ruff check .
```

Tests cover benchmark hashes, seed streams, causal filtering, planning,
count incorporation, update/evaluation schedules, paper-script arguments,
and small real NUTS draws with warm starts. Tests are interface and regression
checks, not a reproduction of the paper's experiments.

The code and benchmark assets use the MIT license in `LICENSE`.
`THIRD_PARTY_NOTICES.md` preserves upstream attribution. `SOURCE.json` records
the source revision and the changes made for this standalone release.
