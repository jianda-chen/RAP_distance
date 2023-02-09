#!/bin/bash

DOMAIN=carla
TASK=highway

SAVEDIR=../save
NOW=$(date +"%Y-%m-%d-%H-%M-%S")
echo ${SAVEDIR}/${DOMAIN}_${TASK}_${NOW}

mkdir -p ${SAVEDIR}

CUDA_VISIBLE_DEVICES=0 python train.py \
    --domain_name ${DOMAIN} \
    --task_name ${TASK} \
    --agent 'RAP' \
    --init_steps 200 \
    --num_train_steps 1000000 \
    --encoder_type pixelCarla098 \
    --decoder_type pixel \
    --transition_model_type 'probabilistic' \
    --action_repeat 8 \
    --critic_tau 0.01 \
    --encoder_tau 0.05 \
    --decoder_weight_lambda 0.0000001 \
    --hidden_dim 1024 \
    --encoder_feature_dim 100 \
    --total_frames 10000 \
    --num_filters 32 \
    --batch_size 128 \
    --init_temperature 0.1 \
    --encoder_lr 1e-3 \
    --decoder_lr 1e-3 \
    --actor_lr 1e-3 \
    --critic_lr 1e-3 \
    --alpha_lr 1e-4 \
    --alpha_beta 0.5 \
    --work_dir ${SAVEDIR}/${DOMAIN}_${TASK}_${NOW} \
    --seed 1 $@ \
    --rap_structural_distance 'mico_angular' \
    --rap_reward_dist \
    --rap_square_target \
    --frame_stack 3 \
    --image_size 84 \
    --eval_freq 20 \
    --num_eval_episodes 0 \
    "$@" 
