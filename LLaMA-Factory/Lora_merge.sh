CUDA_VISIBLE_DEVICES=0 llamafactory-cli export \
    --model_name_or_path /media/a822/82403B14403B0E83/Gwb/WechatRobot/Base_models/Qwen3-4B \
    --adapter_name_or_path /media/a822/82403B14403B0E83/Gwb/WechatRobot/Saved_models/rlhf/4B_lora_PPO_V3/final_adapter \
    --template qwen3 \
    --finetuning_type lora \
    --export_dir /media/a822/82403B14403B0E83/Gwb/WechatRobot/Saved_models/rlhf/4B_lora_PPO_V3/merged \
    --export_size 2 \
    --export_legacy_format False