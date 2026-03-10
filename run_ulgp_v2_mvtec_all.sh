#!/usr/bin/env bash
set -euo pipefail
export CRYPTOGRAPHY_OPENSSL_NO_LEGACY=1
source /home/dct/miniconda3/etc/profile.d/conda.sh
conda activate GAN
mkdir -p /home/dct/unified_model/ULGP/ouput
cd /home/dct/unified_model/AnomalyCLIP-main
LOG=/home/dct/unified_model/ULGP/ouput/anomalyclip_ulgp_v2_mvtec_all.log
CMD='python -u experiments/uar_sra_ulgp_v2.py --dataset mvtec --class_name ALL --device 0 --batch_size 4 --refiner ulgp --ulgp_mode feature-lite --ulgp_candidate_ratio 0.10 --ulgp_dilation_radius 1 --ulgp_k 10 --ulgp_steps 4 --ulgp_alpha 0.60 --ulgp_beta 0.12 --ulgp_tau_u 0.55 --ulgp_clamp_delta 0.08 --ulgp_fusion_lambda 0.75 --sigma 4'
echo "$CMD" > "$LOG"
nohup bash -lc "$CMD" >> "$LOG" 2>&1 &
echo $! > /home/dct/unified_model/ULGP/ouput/anomalyclip_ulgp_v2_mvtec_all.pid
