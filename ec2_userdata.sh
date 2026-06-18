#!/bin/bash
# === 自動關機腳本 ===
# 在 2026-02-28 23:50 UTC 自動 shutdown（點數到期前 10 分鐘）
# instance-initiated-shutdown-behavior=terminate 會讓 shutdown 直接 terminate

# 安裝 Python 和必要套件
yum update -y
yum install -y python3.11 python3.11-pip git
pip3.11 install numpy pandas matplotlib openpyxl

# 設定自動關機（計算距離目標時間的秒數）
TARGET_EPOCH=$(date -d "2026-02-28 23:50:00 UTC" +%s)
NOW_EPOCH=$(date +%s)
REMAINING=$((TARGET_EPOCH - NOW_EPOCH))

if [ $REMAINING -gt 0 ]; then
    echo "設定 $REMAINING 秒後自動關機 ($(date -d @$TARGET_EPOCH))"
    shutdown -h +$((REMAINING / 60))
else
    echo "目標時間已過，立即關機"
    shutdown -h now
fi

# 建立工作目錄
mkdir -p /home/ec2-user/experiment
chown ec2-user:ec2-user /home/ec2-user/experiment

echo "=== 初始化完成 ===" >> /var/log/user-data.log
echo "自動關機時間: $(date -d @$TARGET_EPOCH)" >> /var/log/user-data.log
