#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
CHROMFISH_ROOT="$(cd "$HERE/../.." && pwd)"
METHODS_ROOT="$(cd "$CHROMFISH_ROOT/.." && pwd)"
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
    RUN_DIR="$CHROMFISH_ROOT/runs/$EXPERIMENT_ID"
    EVAL_DIR="$CHROMFISH_ROOT/eval/$EXPERIMENT_ID"

    echo "=== $EXPERIMENT_ID ==="
    python "$CHROMFISH_ROOT/model/scripts/infer.py" \
      --config "$CONFIG_DIR/${VARIANT}.yaml" \
      --checkpoint "$RUN_DIR/best.pt" \
      --stats "$RUN_DIR/normalization.json" \
      --input "$METHODS_ROOT/data/mask_2018_science/test/dawn_sampled_${READS}.npy" \
      --target "$METHODS_ROOT/data/mask_2018_science/test/ground_distance_normalized.npy" \
      --output "$EVAL_DIR/test_predictions.npy" \
      --device "$DEVICE"
  done
done
