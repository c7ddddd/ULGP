#!/usr/bin/env bash
set -euo pipefail

LOG_DIR="/home/dct/unified_model/ULGP/ouput"
LOG_FILE="${LOG_DIR}/anomalyclip_visa_all.log"
CMD="/home/dct/miniconda3/envs/GAN/bin/python -u /home/dct/unified_model/AnomalyCLIP-main/experiments/uar_sra\\ copy.py --dataset visa --class_name ALL --device 0 --n_iter 25 --lr 5e-5 --lambda_delta 0.06 --delta_budget 0.02 --lambda_rank 0.3 --lambda_region 0.01 --lambda_cal 0.05"

mkdir -p "${LOG_DIR}"
{
  echo "CMD: ${CMD}"
  eval "${CMD}"
} > "${LOG_FILE}" 2>&1
