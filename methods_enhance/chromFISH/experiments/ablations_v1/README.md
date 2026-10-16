# chromFISH ablations v1

This experiment set measures five existing architectural switches with one-factor-at-a-time ablations. The baseline and every variant use complete standalone YAML files because the current configuration loader does not support inheritance.

## Variants

| Config | Changed setting |
| --- | --- |
| `baseline.yaml` | None |
| `no_shifted_windows.yaml` | `use_shifted_windows: false` |
| `no_relative_position_bias.yaml` | `use_relative_position_bias: false` |
| `no_rstb_residual.yaml` | `use_rstb_residual: false` |
| `no_long_skip.yaml` | `use_long_skip: false` |
| `no_reconstruction_head.yaml` | `use_reconstruction_head: false` |

All other model, loss, optimizer, data, and training settings must remain identical to the baseline.

## External GPU execution

Run these commands only in an environment with the project dependencies, data, and suitable GPU resources. Do not run them on the local code-editing machine.

```bash
bash experiments/ablations_v1/run_ablations.sh 300 cuda 100
```

For repeated seeds:

```bash
bash experiments/ablations_v1/run_ablations.sh 300 cuda 100,101,102
```

Evaluate with the same read depth, device, and seed list:

```bash
bash experiments/ablations_v1/evaluate_ablations.sh 300 cuda 100,101,102
```

Each run is written directly under `runs/` using an ID such as `ablation_v1__300__no-long-skip__seed-100`. Its resolved configuration, normalization statistics, checkpoints, log, and report stay together. Evaluation outputs use the same ID under `eval/`.

## Reporting

Compare PCC, Spearman correlation, RMSE, MAE, SSIM, and PSNR against the matching baseline. Formal conclusions should report the mean and variance across at least three seeds. A checkpoint must always be evaluated with the YAML and normalization statistics saved for the same run.
