#!/bin/bash
# 五个方法统一配置训练: batch 32 / 200 epoch / 每 epoch 127 步 = 25,400 步
# 用法: bash run_all.sh          (读深 300)
#       bash run_all.sh 50       (读深 50)
set -u
cd "$(dirname "$0")"

READS=${1:-300}
FAILED=""

declare -A DIR=([chromFISH]=chromFISH [hicnn]=HiCNN_software_package [hicarn]=HiCARN \
               [scunicorn]=Unicorn [schicedrn]=ScHiCEDRN)

for M in chromFISH hicnn hicarn scunicorn schicedrn; do
  echo ""
  echo "################ $M  (reads=$READS) ################"
  if ! python "../${DIR[$M]}/train_${M}.py" --reads "$READS" \
    --out-dir "../${DIR[$M]}/runs/${M}_${READS}"; then
    echo "!!!! $M 失败 !!!!"
    FAILED="$FAILED $M"
  fi
done

echo ""
if [ -n "$FAILED" ]; then
  echo "失败的:$FAILED"
else
  echo "五个方法全部完成"
fi
python summarize.py
