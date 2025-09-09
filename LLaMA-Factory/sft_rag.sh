#!/bin/bash

# 选择物理 GPU
export CUDA_VISIBLE_DEVICES=0,1
export GLOO_SOCKET_IFNAME=lo
export NCCL_SOCKET_IFNAME=lo

# 启动分布式训练
nohup torchrun --nproc_per_node=2 src/train.py \
    --stage sft \
    --finetuning_type full \
    --do_train \
    --model_name_or_path ../Base_models/Qwen3-1.7B \
    --dataset RAG_sft_v1 \
    --template qwen3 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 3 \
    --learning_rate 5.0e-6 \
    --num_train_epochs 3 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.1 \
    --cutoff_len 16184 \
    --output_dir /media/a822/82403B14403B0E83/Gwb/RAG/llm/RAG_1.7B_sft_v2 \
    --deepspeed deepspeed/ds_z3_config.json \
    --bf16 True \
    --logging_steps 10 \
    --save_steps 2000 \
    --plot_loss \
    --overwrite_cache \
    --preprocessing_num_workers 16 \
    --dataloader_num_workers 4 \
    --trust_remote_code \
    --freeze_vision_tower \
    --freeze_multi_modal_projector \
    --ddp_timeout 180000000 \
    > ../logs/model_train.log 2>&1
