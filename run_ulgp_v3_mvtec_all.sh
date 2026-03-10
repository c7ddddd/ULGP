#!/usr/bin/env bash
set -euo pipefail
export CRYPTOGRAPHY_OPENSSL_NO_LEGACY=1
source /home/dct/miniconda3/etc/profile.d/conda.sh
conda activate GAN
mkdir -p /home/dct/unified_model/ULGP/ouput
cd /home/dct/unified_model/AnomalyCLIP-main
LOG=/home/dct/unified_model/ULGP/ouput/anomalyclip_ulgp_v3_mvtec_all.log
CMD='python -u experiments/uar_sra_ulgp_v3.py --dataset mvtec --class_name ALL --device 0 --batch_size 4 --refiner ulgp --ulgp_mode feature-lite --ulgp_candidate_ratio 0.06 --ulgp_dilation_radius 1 --ulgp_k 8 --ulgp_steps 3 --ulgp_alpha 0.45 --ulgp_beta 0.10 --ulgp_tau_u 0.60 --ulgp_clamp_delta 0.05 --ulgp_fusion_lambda 0.65 --sigma 4 --topk_ratio 0.01'
echo "$CMD" > "$LOG"
nohup bash -lc "$CMD" >> "$LOG" 2>&1 &
echo $! > /home/dct/unified_model/ULGP/ouput/anomalyclip_ulgp_v3_mvtec_all.pid
