#!/usr/bin/env bash
set -euo pipefail

LOG_DIR="/home/dct/unified_model/ULGP/ouput"
LOG_FILE="${LOG_DIR}/anomalyclip_ulgp_mvtec_all.log"
CMD="/home/dct/miniconda3/envs/GAN/bin/python -u /home/dct/unified_model/AnomalyCLIP-main/experiments/uar_sra_ulgp.py --dataset mvtec --class_name ALL --device 0 --refiner ulgp --ulgp_mode feature-lite --ulgp_candidate_ratio 0.06 --ulgp_dilation_radius 1 --ulgp_k 10 --ulgp_steps 4 --ulgp_alpha 0.6 --ulgp_beta 0.08 --ulgp_tau_u 0.45 --ulgp_clamp_delta 0.08 --ulgp_fusion_lambda 0.75 --sigma 4.0"

mkdir -p "${LOG_DIR}"
{
  echo "CMD: ${CMD}"
  eval "${CMD}"
} > "${LOG_FILE}" 2>&1
