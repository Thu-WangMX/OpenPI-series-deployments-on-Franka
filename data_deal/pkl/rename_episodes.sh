#!/usr/bin/env bash
set -e

BASE_DIR="/work/wmx/openpi/data_from_bos"

# 按你希望的顺序列出三个数据集目录名
DATASET_DIRS=(
  "dataset_Put_the_red_chili_peppers_into_the_basket._2025-12-09_10-17-30"
  "dataset_Put_the_red_chili_peppers_into_the_basket._2025-12-09_11-29-45"
  "dataset_Put_the_red_chili_peppers_into_the_basket._2025-12-09_13-40-31"
)

offset=0  # 已经用了多少集，第一次是 0，第二次 +60，第三次 +120

for d in "${DATASET_DIRS[@]}"; do
  dir="${BASE_DIR}/${d}"
  echo "Processing $dir with offset $offset"

  # 倒序重命名，避免覆盖（虽然这里不会冲突，但习惯好一点）
  for i in $(seq 60 -1 1); do
    old="${dir}/episode${i}"
    if [[ ! -d "$old" && ! -f "$old" ]]; then
      echo "  Skip missing $old"
      continue
    fi

    new_idx=$((i + offset))
    new="${dir}/episode${new_idx}"

    echo "  mv '$old' '$new'"
    mv "$old" "$new"
  done

  offset=$((offset + 60))
done

echo "Done."
