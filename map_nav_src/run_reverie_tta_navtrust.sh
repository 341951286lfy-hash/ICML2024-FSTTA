#!/usr/bin/env bash

set -e

DATA_ROOT=../datasets
train_alg=dagger
features=vitbase
ft_dim=768
obj_features=vitbase
obj_ft_dim=768
ngpus=1
seed=0

CORRUPTION=${1:-motion_blur}
SEVERITY=${2:-0.6}
CKPT=${3:-"${DATA_ROOT}/REVERIE/exprs_map/finetune/dagger-vitbase-seed.0/ckpts/best_val_unseen"}

sev_tag=$(echo "${SEVERITY}" | tr '.' 'p')
RGB_FEAT_FILE="${DATA_ROOT}/R2R/features/pth_vit_base_patch16_224_imagenet_navtrust_${CORRUPTION}_s${sev_tag}.hdf5"

if [ ! -f "${RGB_FEAT_FILE}" ]; then
  echo "[ERROR] corrupted feature file not found:"
  echo "        ${RGB_FEAT_FILE}"
  exit 1
fi

name=${train_alg}-${features}-navtrust-${CORRUPTION}-s${sev_tag}
name=${name}-seed.${seed}
outdir=${DATA_ROOT}/REVERIE/exprs_map/finetune/${name}

flag="--root_dir ${DATA_ROOT} \
 --dataset reverie \
 --output_dir ${outdir} \
 --world_size ${ngpus} \
 --seed ${seed} \
 --tokenizer bert \
 --enc_full_graph \
 --graph_sprels \
 --fusion dynamic \
 --multi_endpoints \
 --dagger_sample sample \
 --train_alg ${train_alg} \
 --num_l_layers 9 \
 --num_x_layers 4 \
 --num_pano_layers 2 \
 --max_action_len 15 \
 --max_instr_len 200 \
 --max_objects 20 \
 --batch_size 1 \
 --lr 1e-5 \
 --iters 200000 \
 --log_every 100 \
 --optim adamW \
 --features ${features} \
 --obj_features ${obj_features} \
 --image_feat_size ${ft_dim} \
 --angle_feat_size 4 \
 --obj_feat_size ${obj_ft_dim} \
 --ml_weight 0.2 \
 --feat_dropout 0.4 \
 --dropout 0.5 \
 --gamma 0. \
 --rgb_feat_file ${RGB_FEAT_FILE} \
 --rgb_corrupt_type ${CORRUPTION} \
 --rgb_corrupt_severity ${SEVERITY}"

CUDA_VISIBLE_DEVICES='0' python3 main_nav_obj.py $flag \
  --resume_file "${CKPT}" \
  --test --submit
