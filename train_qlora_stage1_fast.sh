#!/bin/bash

# QLoRA Stage 1: Full Norwegian - OPTIMIZED FOR SPEED
# Dataset: heiertech/vibevoice-mcv-scripted-no-v24
# 3 epochs

source .venv/bin/activate
export PYTHONPATH=/home/me/ht/area51/VibeVoice-finetuning/src:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=1
export HF_TOKEN="${HF_TOKEN:-$AI_HF_TOKEN}"
export HF_DATASETS_CACHE=/tmp/hf_datasets_cache
export HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=true
export WANDB_DISABLED=true

echo "Stage 1: QLoRA on full Norwegian (FAST)"
echo "Base model: vibevoice/VibeVoice-7B"

python -m src.finetune_vibevoice_lora \
    --do_train True \
    --do_eval True \
    --model_name_or_path vibevoice/VibeVoice-7B \
    --processor_name_or_path src/vibevoice/processor \
    --dataset_name heiertech/vibevoice-mcv-scripted-no-v24 \
    --train_split_name train \
    --eval_split_name validation \
    --text_column_name text \
    --audio_column_name audio \
    --voice_prompts_column_name voice_prompts \
    --output_dir vibevoice-7b-nob-qlora-stage1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 2 \
    --dataloader_num_workers 4 \
    --dataloader_pin_memory True \
    --use_qlora False \
    --lora_r 32 \
    --lora_alpha 128 \
    --lora_dropout 0.05 \
    --optim adamw_bnb_8bit \
    --learning_rate 2.5e-4 \
    --num_train_epochs 3 \
    --max_steps -1 \
    --eval_strategy steps \
    --eval_steps 200 \
    --eval_on_start False \
    --logging_steps 10 \
    --save_steps 300 \
    --save_total_limit 2 \
    --report_to tensorboard \
    --remove_unused_columns False \
    --bf16 True \
    --gradient_clipping True \
    --gradient_checkpointing True \
    --ddpm_batch_mul 2 \
    --diffusion_loss_weight 1.4 \
    --train_diffusion_head True \
    --ce_loss_weight 0.04 \
    --voice_prompt_drop_rate 0.2 \
    --lora_target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.03 \
    --max_grad_norm 0.8 \
    --overwrite_output_dir True \
    --push_to_hub True \
    --hub_model_id heiertech/vibevoice-7b-nob-qlora-stage1 \
    --hub_strategy checkpoint \
    --hub_private_repo True

echo "Stage 1 complete!"
