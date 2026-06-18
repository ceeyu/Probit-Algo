#!/bin/bash
cd /home/ec2-user/experiment

# 啟動批次實驗 1: batch_hw_noise_comparison (down_counter 排程)
nohup python3 batch_hw_noise_comparison.py \
    --graph_dir ./graph \
    --start_graph 1 \
    --end_graph 81 \
    > batch_noise_run.log 2>&1 &
echo "batch_hw_noise_comparison PID: $!"

# 啟動批次實驗 2: batch_hardware_comparison (linear 排程)
nohup python3 batch_hardware_comparison.py \
    --graph_dir ./graph \
    --start_graph 1 \
    --end_graph 81 \
    > batch_hw_run.log 2>&1 &
echo "batch_hardware_comparison PID: $!"

echo "兩個實驗已在背景啟動"
echo "查看進度: tail -f batch_noise_run.log"
echo "查看進度: tail -f batch_hw_run.log"
