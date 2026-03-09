#!/usr/bin/env bash
set -euo pipefail

LOG_DIR="/home/dct/unified_model/ULGP/ouput"
LOG_FILE="${LOG_DIR}/anomalyclip_mvtec_all_mid_d0.log"
CMD="/home/dct/miniconda3/envs/GAN/bin/python -u /home/dct/unified_model/AnomalyCLIP-main/experiments/uar_sra\\ copy.py --dataset mvtec --class_name ALL --device 0 --n_iter 40 --lr 1e-4 --lambda_delta 0.10 --delta_budget 0.03 --lambda_rank 0.3 --lambda_region 0.01 --lambda_cal 0.05"

mkdir -p "${LOG_DIR}"
{
  echo "CMD: ${CMD}"
  eval "${CMD}"
} > "${LOG_FILE}" 2>&1
