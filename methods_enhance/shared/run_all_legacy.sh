#!/bin/bash
# 4 方法 x 3 读深 = 12 次训练, 串行执行(避免抢显存)。
# 用法: bash run_all.sh            跑全部
#       bash run_all.sh 50         只跑读深 50
set -u
cd "$(dirname "$0")"   # 在 legacy/ 里调同级脚本

READS_LIST=${1:-"50 100 300"}
FAILED=""

for R in $READS_LIST; do
  declare -A DIR=([HiCNN]=HiCNN_software_package [ScHiCEDRN]=ScHiCEDRN [HiCARN]=HiCARN [ScUnion]=Unicorn)

for S in HiCNN ScHiCEDRN HiCARN ScUnion; do
    echo "############ $S  reads=$R  ############"
    if ! bash "../${DIR[$S]}/legacy/${S}_train.sh" "$R"; then
      echo "!!!! $S reads=$R 失败 !!!!"
      FAILED="$FAILED ${S}_${R}"
    fi
  done
done

echo
if [ -n "$FAILED" ]; then
  echo "失败的:"; for f in $FAILED; do echo "  $f"; done
else
  echo "12 次全部完成"
fi
