#!/usr/bin/env bash
set -euo pipefail
set -C
cd "$(dirname "$0")/.."
runs=${RUNS:-10}
[[ "$runs" =~ ^[1-9][0-9]*$ ]] || { echo 'RUNS must be a positive integer' >&2; exit 2; }
out=${OUTPUT_DIR:-outputs/schedule-ablation}
mkdir -p "$out"
for cutoff in 4000 1000 0; do
    for ((i=1; i<=runs; i++)); do
        uv run --no-sync python latent_psrl.py \
            --env-id LMDP/DirichletRandoms-Z2-v0 --seed "$i" \
            --total-episodes 4000 --warmup-episodes 100 --update-every 10 \
            --full-em-until "$cutoff" --prior-alpha 1.0 \
            --em-restarts 20 --em-max-iters 100 --em-tol 1e-5 --eval-episodes 100 \
            --run-id "schedule-cutoff$cutoff-s$i" \
            --output "$out/cutoff$cutoff-s$i.csv" > "$out/cutoff$cutoff-s$i.log" 2>&1
    done
done
