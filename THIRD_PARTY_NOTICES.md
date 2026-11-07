# Sources and attribution

L-PSRL, L-UCRL, pooled PSRL, UCBVI-CH, the latent-MDP environment and tabular
assets are extracted from the maintained research implementation at commit
`3e5a62b`. Numerical update rules, named seed derivation, tie breaking and
benchmark arrays are retained. The public release uses the paper's Dirichlet(1) transition priors.
Historical ablations and optional service integrations are omitted; training
loops are explicit in each runnable learner. Numerical updates, benchmark
arrays, random draws and update order are preserved. Release changes and
the exact export revision are recorded in `SOURCE.json`.

L-UCRL follows Kwon et al. (2021), *RL for Latent MDPs: Regret Guarantees and
a Lower Bound*. UCBVI-CH follows Azar, Osband and Munos (2017), *Minimax
Regret Bounds for Reinforcement Learning*. QMDP follows Littman, Cassandra
and Kaelbling (1995), *Learning Policies for Partially Observable Environments:
Scaling Up*.

PS4LMDP adapts the maintained PS4POMDP baseline's static-latent likelihood,
NUTS sampling, dense finite-horizon QMDP and causal belief filter. The baseline
was ported from `latent-mdp-rlxf` commit
`b009f7f9ecac057421547ee9097870d1eb9c47d0`; its upstream provenance identifies
`ps4pomdp-dev` commit `b31bb53c28457280dabe177a09606a4270238d40`. That upstream
revision had no root license file. This submission includes the maintained
static-latent adaptation and does not vendor the upstream generic inference
or planning files, or the SARSOP solver. This provenance is retained rather
than assigning a license to those omitted upstream sources.

Gymnasium, NumPy, SciPy, tyro, JAX, NumPyro and optional W&B are installed as
dependencies, not vendored. Their own licenses apply. The MIT license in
`LICENSE` applies to this project's code and benchmark assets.
