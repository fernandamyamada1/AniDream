################################################## Configuration ######################################################
# Configurations
text="${1}"
enable_expr_control=false

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
#source activate dreamwaltz
#echo "Using Python at: $(which python)"

############################################## Stage I - NeRF Training ################################################
# 1.1 Canonical NeRF Training - Progressive Low Resolution: 64x64 -> 128x128 -> 256x256
last_ckpt="external/human_templates/instant-ngp/adult_neutral/"
exp_name="${exp_root}/nerf,64>256,10k"
/home3/staff/ya004545/.conda/envs/dw/bin/python3.11  main.py \
    --guide.text "${text}" \
    --log.exp_name "${exp_name}" \
    --optim.ckpt "${last_ckpt}" \
    --predefined_body_parts ${predefined_body_parts} \
    --stage nerf \
    --nerf.bg_mode gray \
    --optim.fp16 True \
    --optim.iters 10000 \
    --prompt.scene canonical \
    --data.train_w "64,128,256" \
    --data.train_h "64,128,256" \
    --data.progressive_grid True \
    --use_sigma_guidance True

# 1.2 Canonical NeRF Training - High Resolution: 512x512 (Could be Skipped if GPU Memory is Limited)
last_ckpt="${exp_name}/checkpoints/"
exp_name="${exp_name}-nerf,512,5k"
/home3/staff/ya004545/.conda/envs/dw/bin/python3.11 main.py \
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


