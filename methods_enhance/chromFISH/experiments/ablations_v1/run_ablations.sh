#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
CHROMFISH_ROOT="$(cd "$HERE/../.." && pwd)"
CONFIG_DIR="$HERE/configs"

READS="${1:-300}"
DEVICE="${2:-auto}"
SEEDS_CSV="${3:-100}"
IFS=',' read -r -a SEEDS <<< "$SEEDS_CSV"

VARIANTS=(
  baseline
  no_shifted_windows
  no_relative_position_bias
  no_rstb_residual
  no_long_skip
  no_reconstruction_head
)

for SEED in "${SEEDS[@]}"; do
  for VARIANT in "${VARIANTS[@]}"; do
    LABEL="${VARIANT//_/-}"
    EXPERIMENT_ID="ablation_v1__${READS}__${LABEL}__seed-${SEED}"
    OUT_DIR="$CHROMFISH_ROOT/runs/$EXPERIMENT_ID"

    echo "=== $EXPERIMENT_ID ==="
    python "$CHROMFISH_ROOT/train_chromFISH.py" \
      --config "$CONFIG_DIR/${VARIANT}.yaml" \
      --reads "$READS" \
      --seed "$SEED" \
      --device "$DEVICE" \
      --experiment-id "$EXPERIMENT_ID" \
      --out-dir "$OUT_DIR"
  done
done
