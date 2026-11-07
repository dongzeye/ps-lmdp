#!/usr/bin/env bash
set -euo pipefail
set -C
cd "$(dirname "$0")/.."
runs=${RUNS:-10}
[[ "$runs" =~ ^[1-9][0-9]*$ ]] || { echo 'RUNS must be a positive integer' >&2; exit 2; }
out=${OUTPUT_DIR:-outputs/main}
mkdir -p "$out"
envs=(
    DirichletRandoms-Z2 DirichletRandoms-Z4
    DirichletRandoms-Z8 DirichletRandoms-Z16
    FrozenLake-4x4-Z2 FrozenLake-4x4-Z4
    FrozenLake-8x8-Z2 FrozenLake-8x8-Z4
)
if (( $# )); then envs=("$@"); fi

lpsrl_args=(
    --warmup-episodes 100
    --update-every 10
    --em-restarts 20
    --em-max-iters 100
    --em-tol 1e-5
    --eval-episodes 100
    --prior-alpha 1.0
)
lucrl_args=(
    --warmup-episodes 100
    --update-every 10
    --em-restarts 20
    --em-max-iters 100
    --em-tol 1e-5
    --eval-episodes 100
    --bonus-scale 0.001
    --confidence-eta 0.05
    --belief-alpha 0.001
)
psrl_args=(--prior-alpha 1.0 --eval-episodes 100)
ucbvi_args=(--delta 0.05 --bonus-multiplier 1.0 --eval-episodes 100)

for env in "${envs[@]}"; do
    total=4000
    cutoff=1000
    if [[ "$env" == FrozenLake-8x8-* ]]; then total=16000; cutoff=4000; fi
    for ((i=1; i<=runs; i++)); do
        uv run --no-sync python latent_psrl.py "${lpsrl_args[@]}" --total-episodes "$total" --full-em-until "$cutoff" \
            --env-id "LMDP/$env-v0" --seed "$i" \
            --run-id "main-$env-latent_psrl-s$i" \
            --output "$out/$env-latent_psrl-s$i.csv" \
            > "$out/$env-latent_psrl-s$i.log" 2>&1
        uv run --no-sync python latent_ucrl.py "${lucrl_args[@]}" --total-episodes "$total" --full-em-until "$cutoff" \
            --env-id "LMDP/$env-v0" --seed "$i" \
            --run-id "main-$env-latent_ucrl-s$i" \
            --output "$out/$env-latent_ucrl-s$i.csv" \
            > "$out/$env-latent_ucrl-s$i.log" 2>&1
        uv run --no-sync python psrl.py "${psrl_args[@]}" --total-episodes "$total" \
            --env-id "LMDP/$env-v0" --seed "$i" \
            --run-id "main-$env-psrl-s$i" \
            --output "$out/$env-psrl-s$i.csv" \
            > "$out/$env-psrl-s$i.log" 2>&1
        uv run --no-sync python ucbvi.py "${ucbvi_args[@]}" --total-episodes "$total" \
            --env-id "LMDP/$env-v0" --seed "$i" \
            --run-id "main-$env-ucbvi-s$i" \
            --output "$out/$env-ucbvi-s$i.csv" \
            > "$out/$env-ucbvi-s$i.log" 2>&1
    done
done
