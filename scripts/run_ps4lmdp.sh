#!/usr/bin/env bash
set -euo pipefail
set -C
cd "$(dirname "$0")/.."
runs=${RUNS:-10}
[[ "$runs" =~ ^[1-9][0-9]*$ ]] || { echo 'RUNS must be a positive integer' >&2; exit 2; }
out=${OUTPUT_DIR:-outputs/ps4lmdp}
mkdir -p "$out"
envs=(
    DirichletRandoms-Z2 DirichletRandoms-Z4
    FrozenLake-4x4-Z2 FrozenLake-4x4-Z4
)
if (( $# )); then envs=("$@"); fi

args=(
    --total-episodes 4000
    --warmup-episodes 100
    --update-every 10
    --eval-episodes 100
    --mixture-prior-alpha 1.0
    --mcmc-num-warmup 200
    --mcmc-num-samples 200
    --mcmc-num-chains 1
    --warm-start
    --jax-clear-caches-every 200
    --prior-alpha 1.0
)
for env in "${envs[@]}"; do
    for ((i=1; i<=runs; i++)); do
        uv run --no-sync --extra ps4 python ps4lmdp.py "${args[@]}" \
            --env-id "LMDP/$env-v0" --seed "$i" \
            --run-id "main-$env-ps4lmdp-s$i" \
            --output "$out/$env-ps4lmdp-s$i.csv" > "$out/$env-ps4lmdp-s$i.log" 2>&1
    done
done
