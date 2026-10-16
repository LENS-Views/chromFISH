#!/bin/bash
# 用 unified/ 重新训出的 checkpoint 在测试集上评估五个方法, 并汇总成一张表。
#
# 用法: bash eval_all_test.sh        (读深 300)
#       bash eval_all_test.sh 50
#
# 四个 baseline 走 eval_baselines.py(它直接吃 [0,1] 原始数据);
# 我们的模型走 chromFISH/model/scripts/infer.py —— 它的输入要归一化到 [-1,1]、输出要反归一化,
# eval_baselines 那套不适用。两者的指标模块是同一个, 所以结果可比。
set -u

HERE="$(cd "$(dirname "$0")" && pwd)"
READS=${1:-300}
ROOT="$HERE/.."

# 方法名 -> 目录名 (与 common.py 的 METHOD_DIRS 一致)
declare -A DIR=([HiCNN]=HiCNN_software_package [HiCARN]=HiCARN \
               [ScUnicorn]=Unicorn [ScHiCEDRN]=ScHiCEDRN)


echo "########## 四个 baseline (测试集) ##########"
i=0
for spec in \
    "HiCNN:hicnn_${READS}/final.pt" \
    "HiCARN:hicarn_${READS}/best_ssim.pt" \
    "ScUnicorn:scunicorn_${READS}/final.pt" \
    "ScHiCEDRN:schicedrn_${READS}/best_ssim.pt"; do
  M="${spec%%:*}"; CK="${spec#*:}"
  i=$((i + 1))
  echo ""
  echo "--- [$i/5] $M"
  python "$HERE/eval_baselines.py" --method "$M" --reads "$READS" \
    --checkpoint "$ROOT/${DIR[$M]}/runs/$CK" || echo "!!!! $M 失败 !!!!"
done

echo ""
echo "########## [5/5] 我们的模型 ##########"
python "$ROOT/chromFISH/model/scripts/infer.py" \
  --config "$ROOT/chromFISH/model/configs/default.yaml" \
  --checkpoint "$ROOT/chromFISH/runs/chromFISH_${READS}/best.pt" \
  --stats "$ROOT/chromFISH/model/outputs/run/normalization.json" \
  --input "$ROOT/data/mask_2018_science/test/dawn_sampled_${READS}.npy" \
  --target "$ROOT/data/mask_2018_science/test/ground_distance_normalized.npy" \
  --output "$ROOT/chromFISH/eval/chromFISH_${READS}/test_predictions.npy" \
  || echo "!!!! Ours 失败 !!!!"

echo ""
python - "$HERE" "$READS" <<'PY'
import json, sys
from pathlib import Path
here, reads = Path(sys.argv[1]), sys.argv[2]
root = here.parent

# 各方法的评估结果就落在各自目录下
DIRS = {"HiCNN": "HiCNN_software_package", "HiCARN": "HiCARN",
        "ScUnicorn": "Unicorn", "ScHiCEDRN": "ScHiCEDRN", "chromFISH": "chromFISH"}

rows = {}
for name, sub in DIRS.items():
    out_dir = root / sub / "eval" / f"{name}_{reads}"
    # baseline 走 eval_baselines.py 写 metrics.json; chromFISH 走 infer.py 写 *.metrics.json
    for cand in (out_dir / "metrics.json", out_dir / "test_predictions.metrics.json"):
        if cand.exists():
            payload = json.loads(cand.read_text(encoding="utf-8"))
            rows[name] = payload.get("summary", payload)
            break

if not rows:
    print("没找到任何结果"); raise SystemExit

order = ["chromFISH", "ScUnicorn", "ScHiCEDRN", "HiCARN", "HiCNN"]
rows = dict(sorted(rows.items(), key=lambda kv: order.index(kv[0]) if kv[0] in order else 99))
label = {"chromFISH": "我们的模型 (chromFISH)"}

print(f"=== 测试集 1170 样本 (reads={reads}) ===")
print(f"{'方法':<11}{'pcc':>8}{'spearman':>10}{'rmse':>8}{'mae':>8}{'ssim':>8}{'psnr':>8}")
print("-" * 61)
for name, s in rows.items():
    print(f"{label.get(name, name):<11}{s['pcc']:>8.4f}{s['spearman']:>10.4f}"
          f"{s['rmse']:>8.4f}{s['mae']:>8.4f}{s['ssim']:>8.4f}{s['psnr']:>8.2f}")

out = here / f"final_{reads}_test.json"
out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
print("\n已写入 " + str(out))
PY
