export GLOO_SOCKET_IFNAME=lo
export NCCL_SOCKET_IFNAME=lo

CUDA_VISIBLE_DEVICES=0 \
vllm serve /media/a822/82403B14403B0E83/Gwb/WechatRobot/Saved_models/rlhf/4B_lora_PPO_V3/merged \
    --tensor-parallel-size 1 \
    --port 8888 \
    --disable-log-requests \
    --enable-auto-tool-choice \
    --reasoning-parser deepseek_r1 \
    --tool-call-parser hermes \
    --max-model-len 3000 \
    > ./logs/model_output.log 2>&1 &