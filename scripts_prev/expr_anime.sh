################################################## Configuration ######################################################
# Configurations
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
text="${1}"
enable_expr_control=true
timestamp=$(date "+%Y%m%d_%H%M%S")

animation = True

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
REMOTE_DIR="/home/fernanda/ResultsDW/"



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

 # 1.2 Canonical NeRF Training - High Resolution: 512x512 (Could be Skipped if GPU Memory is Limited)
last_ckpt="${exp_name}/checkpoints/"
exp_name="${exp_name}-nerf,512,5k"
CUDA_VISIBLE_DEVICES=0,1 python3 main.py \
    --guide.text "${text}" \
    --log.exp_name "${exp_name}" \
    --optim.ckpt "${last_ckpt}" \
    --predefined_body_parts ${predefined_body_parts} \
    --stage nerf \
    --nerf.bg_mode gray \
    --optim.fp16 True \
    --optim.iters 5000 \
    --prompt.scene canonical \
    --data.train_w 512 \
    --data.train_h 512 \
    --use_sigma_guidance True

if ${animation}; then
    ############################################## Stage II - 3DGS Training ###############################################
    # 2.1 Animatable 3DGS Training - Canonical Pose
    last_ckpt="${exp_name}/checkpoints/"
    exp_name="${exp_name}-3dgs,cnl,5k"
    CUDA_VISIBLE_DEVICES=0,1 python3 main.py \
        --guide.text "${text}" \
        --log.exp_name "${exp_name}" \
        --render.from_nerf "${last_ckpt}" \
        --predefined_body_parts ${predefined_body_parts} \
        --stage gs \
        --optim.iters 5000 \
        --prompt.scene canonical \
        --render.learn_hand_betas True \
        --render.lbs_weight_smooth True \
        --render.bg_color [0.5,0.5,0.5]

    # 2.2 Animatable 3DGS Training - Random Canonical Pose
    last_ckpt="${exp_name}/checkpoints/"
    exp_name="${exp_name}-3dgs,rcnl,5k"
    CUDA_VISIBLE_DEVICES=0,1 python3 main.py \
        --guide.text "${text}" \
        --log.exp_name "${exp_name}" \
        --optim.ckpt "${last_ckpt}" \
        --predefined_body_parts ${predefined_body_parts} \
        --stage gs \
        --optim.iters 5000 \
        --prompt.scene canonical-R \
        --render.bg_color [0.5,0.5,0.5]

    # 2.3 Animatable 3DGS Training - Random Pose
    last_ckpt="${exp_name}/checkpoints/"
    exp_name="${exp_name}-3dgs,rand,5k"
    CUDA_VISIBLE_DEVICES=0,1 python3 main.py \
        --guide.text "${text}" \
        --log.exp_name "${exp_name}" \
        --optim.ckpt "${last_ckpt}" \
        --predefined_body_parts ${predefined_body_parts} \
        --stage gs \
        --optim.iters 5000 \
        --prompt.scene ${random_pose_sampler} \
        --render.bg_color [0.5,0.5,0.5]


    ################################################### Animation Test ####################################################
    # 3. Animation Test - TalkShow
    CUDA_VISIBLE_DEVICES=0,1 python3 main.py \
        --log.exp_name "${exp_name}" \
        --predefined_body_parts ${predefined_body_parts} \
        --stage gs \
        --log.eval_only True \
        --prompt.scene demo,talkshow \
        --data.eval_elevation 90 \
        --data.eval_camera_track fixed
fi

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
