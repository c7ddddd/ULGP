#!/usr/bin/env bash
set -euo pipefail

LOG_DIR="/home/dct/unified_model/ULGP/ouput"
LOG_FILE="${LOG_DIR}/anomalyclip_ulgp_visa_all.log"
CMD="/home/dct/miniconda3/envs/GAN/bin/python -u /home/dct/unified_model/AnomalyCLIP-main/experiments/uar_sra_ulgp.py --dataset visa --class_name ALL --device 0 --refiner ulgp --ulgp_mode feature-lite --ulgp_candidate_ratio 0.05 --ulgp_dilation_radius 1 --ulgp_k 8 --ulgp_steps 3 --ulgp_alpha 0.5 --ulgp_beta 0.1 --ulgp_tau_u 0.5 --ulgp_clamp_delta 0.05 --ulgp_fusion_lambda 0.7 --sigma 4.0"

mkdir -p "${LOG_DIR}"
{
  echo "CMD: ${CMD}"
  eval "${CMD}"
} > "${LOG_FILE}" 2>&1
