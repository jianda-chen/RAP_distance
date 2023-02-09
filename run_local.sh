#!/bin/bash

# DOMAIN=finger
# TASK=spin
# ACTION_REPEAT=2
# DOMAIN=cartpole
# TASK=swingup
# ACTION_REPEAT=8
# DOMAIN=walker
# TASK=walk
# ACTION_REPEAT=2
DOMAIN=cheetah
TASK=run
ACTION_REPEAT=4
# DOMAIN=reacher
# TASK=easy
# DOMAIN=ball_in_cup
# TASK=catch
# DOMAIN=humanoid
# TASK=walk
# ACTION_REPEAT=2

# SAVEDIR=./save
SAVEDIR=../log

NOW=$(date +"%Y-%m-%d-%H-%M-%S")
echo ${SAVEDIR}/${DOMAIN}_${TASK}_${NOW}

python train.py \
    --domain_name ${DOMAIN} \
    --task_name ${TASK} \
    --agent 'RAP' \
    --init_steps 1000 \
    --replay_buffer_capacity 1000000 \
    --num_train_steps 1000000 \
    --encoder_type pixel \
    --decoder_type identity \
    --img_source video \
    --resource_files '../kinetics-downloader/dataset/train/driving_car/*.mp4' \
    --eval_resource_files '../kinetics-downloader/dataset/train/driving_car/*.mp4' \
    --transition_model_type 'probabilistic' \
    --action_repeat ${ACTION_REPEAT} \
    --critic_tau 0.01 \
    --encoder_tau 0.05 \
    --hidden_dim 1024 \
    --encoder_feature_dim 100 \
    --total_frames 1000 \
    --num_layers 4 \
    --num_filters 32 \
    --batch_size 128 \
    --encoder_lr 5e-4 \
    --decoder_lr 5e-4 \
    --actor_lr 5e-4 \
    --critic_lr 5e-4 \
    --alpha_lr 1e-4 \
    --alpha_beta 0.5 \
    --init_temperature 0.1 \
    --num_eval_episodes 5 \
    --work_dir ${SAVEDIR}/${DOMAIN}_${TASK}_${NOW} \
    --rap_structural_distance 'mico_angular' \
    --rap_reward_dist \
    --rap_square_target \
    --save_tb \
    --save_model \
    --seed 100 "$@"
