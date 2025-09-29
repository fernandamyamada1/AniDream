################################################## Configuration ######################################################
# Configurations
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
text="${1}"
enable_expr_control=true
timestamp=$(date "+%Y%m%d_%H%M%S")

# Auto Setups
exp_root="$(echo "$text" | tr '[:upper:]' '[:lower:]' | sed 's/ /_/g')"
if ${enable_expr_control}; then
    predefined_body_parts=hands,face
    random_pose_sampler=random-body,hand,expr
else
    predefined_body_parts=hands
    random_pose_sampler=random-body,hand
fi

#!/bin/bash

# Define variables
REMOTE_USER="fernanda"
REMOTE_HOST="192.168.1.108"
REMOTE_DIR="/home/fernanda/NeoDream/AniDream/"



############################################## Stage I - NeRF Training ################################################
# 1.1 Canonical NeRF Training - Progressive Low Resolution: 64x64 -> 128x128 -> 256x256
last_ckpt="external/human_templates/instant-ngp/adult_neutral/"
exp_name="${exp_root}/nerf,64>256,10k"
CUDA_VISIBLE_DEVICES=0,1 python3 main.py \
    --guide.text "${text}" \
    --log.exp_name "${exp_name}" \
    --optim.ckpt "${last_ckpt}" \
    --predefined_body_parts ${predefined_body_parts} \
    --stage nerf \
    --nerf.bg_mode gray \
    --optim.fp16 True \
    --optim.iters 15000 \
    --prompt.scene canonical \
    --data.train_w "64,128,256" \
    --data.train_h "64,128,256" \
    --data.progressive_grid True \
    --use_sigma_guidance True




# Transfer the file/directory using scp
timestamped_name="anidream_${timestamp}_${exp_root}"
echo "Transferring ${timestamped_name} to remote host..."
scp -r "${exp_root}" "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/${timestamped_name}"

# Check if transfer was successful
if [ $? -eq 0 ]; then
    echo "Transfer completed successfully"
    
    # Delete local copy
    echo "Deleting local copy..."
    rm -rf "${exp_root}"
    echo "Local copy deleted"
else
    echo "Error: Transfer failed"
    exit 1
fi
