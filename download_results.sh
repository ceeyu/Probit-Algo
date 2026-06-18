#!/bin/bash
# === 從 EC2 下載實驗結果 ===
EC2_IP="43.207.86.117"
KEY="credit-burn-key4.pem"
SSH_OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=NUL -i $KEY"
REMOTE_DIR="/home/ec2-user/experiment"
LOCAL_DIR="./ec2_results"

mkdir -p "$LOCAL_DIR"

echo "=== 下載結果檔案 ==="

# 下載 trial1000_steps1000 結果
echo "--- trial1000_steps1000 ---"
scp -r $SSH_OPTS ec2-user@$EC2_IP:$REMOTE_DIR/noise_hardware_comparison_down_counter/trial1000_steps1000 "$LOCAL_DIR/"

# 下載 trial100_steps10000 結果
echo "--- trial100_steps10000 ---"
scp -r $SSH_OPTS ec2-user@$EC2_IP:$REMOTE_DIR/noise_hardware_comparison_down_counter/trial100_steps10000 "$LOCAL_DIR/"

# 下載 log
scp $SSH_OPTS ec2-user@$EC2_IP:$REMOTE_DIR/batch_noise_run.log "$LOCAL_DIR/"

echo "=== 下載完成 ==="
echo "結果在: $LOCAL_DIR/"
ls -la "$LOCAL_DIR/"
