#!/bin/bash

DOMAIN=eleurent
TASK=highway
ACTION_REPEAT=1


# SAVEDIR=./save
SAVEDIR=../log

NOW=$(date +"%Y-%m-%d-%H-%M-%S")
echo ${SAVEDIR}/${DOMAIN}_${TASK}_${NOW}

python train.py \
    --domain_name ${DOMAIN} \
    --task_name ${TASK} \
    --agent 'RAP' \
    --eleurent \
    --init_steps 1000 \
    --num_train_steps 1000000 \
    --encoder_type pixel \
    --decoder_type identity \
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
    --encoder_lr 1e-4 \
    --decoder_lr 1e-4 \
    --actor_lr 1e-4 \
    --critic_lr 1e-4 \
    --alpha_lr 1e-4 \
    --alpha_beta 0.5 \
    --init_temperature 0.1 \
    --num_eval_episodes 1 \
    --work_dir ${SAVEDIR}/${DOMAIN}_${TASK}_${NOW} \
    --rap_structural_distance 'mico_angular' \
    --rap_reward_dist \
    --rap_square_target \
    --save_tb \
    --save_model \
    --seed 100 "$@"

