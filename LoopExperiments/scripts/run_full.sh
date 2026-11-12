#!/usr/bin/env bash
# Full phase-1 collection. One process per GPU; process i handles shard i of every dataset.
# Resumable: completed batches are skipped on restart.
#   bash scripts/run_full.sh runs/v1 "1 5 7" [batch_size]
set -euo pipefail
OUT=${1:?run dir}
GPUS=(${2:?space-separated GPU ids})
BS=${3:-8}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
PY=$ROOT/.venv/bin/python
N=${#GPUS[@]}
mkdir -p "$OUT/logs"
for i in "${!GPUS[@]}"; do
  g=${GPUS[$i]}
  (
    for ds in math500 mbppplus gsm8k; do
      CUDA_VISIBLE_DEVICES=$g "$PY" "$ROOT/scripts/collect.py" --dataset $ds --out "$OUT" \
        --device cuda:0 --batch-size "$BS" --shard "$i" --num-shards "$N"
    done
  ) > "$OUT/logs/gpu$g-shard$i.log" 2>&1 &
  echo "shard $i -> GPU $g (pid $!)"
done
wait
echo "all shards finished"
