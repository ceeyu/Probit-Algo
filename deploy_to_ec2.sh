#!/bin/bash
# === 部署實驗程式碼到 EC2 ===
# EC2 IP: 13.113.237.37
# Instance: i-01d2d8037633a775c (c5.4xlarge)

EC2_IP="13.113.237.37"
KEY="credit-burn-key.pem"
REMOTE_DIR="/home/ec2-user/experiment"
SSH_OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -i $KEY"

echo "=== 步驟 1: 等待 EC2 SSH 就緒 ==="
for i in $(seq 1 30); do
    ssh $SSH_OPTS ec2-user@$EC2_IP "echo 'SSH ready'" 2>/dev/null && break
    echo "等待 SSH... ($i/30)"
    sleep 10
done

echo ""
echo "=== 步驟 2: 建立遠端目錄 ==="
ssh $SSH_OPTS ec2-user@$EC2_IP "mkdir -p $REMOTE_DIR/graph"

echo ""
echo "=== 步驟 3: 上傳 Python 程式碼 ==="
# 核心程式
scp $SSH_OPTS hw_noise_probit_limit.py ec2-user@$EC2_IP:$REMOTE_DIR/
scp $SSH_OPTS hardware_multiple_spin_probit_annealing.py ec2-user@$EC2_IP:$REMOTE_DIR/
scp $SSH_OPTS batch_hw_noise_comparison.py ec2-user@$EC2_IP:$REMOTE_DIR/
scp $SSH_OPTS batch_hardware_comparison.py ec2-user@$EC2_IP:$REMOTE_DIR/
scp $SSH_OPTS requirements.txt ec2-user@$EC2_IP:$REMOTE_DIR/

echo ""
echo "=== 步驟 4: 上傳 Graph 檔案 ==="
scp $SSH_OPTS graph/*.txt ec2-user@$EC2_IP:$REMOTE_DIR/graph/

echo ""
echo "=== 步驟 5: 安裝 Python 套件 ==="
ssh $SSH_OPTS ec2-user@$EC2_IP "
    # 檢查 python3 是否可用
    if command -v python3.11 &>/dev/null; then
        PYTHON=python3.11
        PIP=pip3.11
    elif command -v python3 &>/dev/null; then
        PYTHON=python3
        PIP=pip3
    else
        echo 'ERROR: Python not found, installing...'
        sudo yum install -y python3 python3-pip
        PYTHON=python3
        PIP=pip3
    fi
    
    echo \"Using: \$PYTHON\"
    \$PYTHON --version
    
    # 安裝套件
    \$PIP install --user numpy pandas matplotlib openpyxl
    
    echo ''
    echo '=== 環境就緒 ==='
    echo \"Python: \$(\$PYTHON --version)\"
    echo \"NumPy: \$(\$PYTHON -c 'import numpy; print(numpy.__version__)')\"
    echo \"Pandas: \$(\$PYTHON -c 'import pandas; print(pandas.__version__)')\"
"

echo ""
echo "=== 步驟 6: 確認檔案已上傳 ==="
ssh $SSH_OPTS ec2-user@$EC2_IP "ls -la $REMOTE_DIR/ && echo '' && ls -la $REMOTE_DIR/graph/"

echo ""
echo "=== 部署完成 ==="
echo "EC2 IP: $EC2_IP"
echo "工作目錄: $REMOTE_DIR"
echo ""
echo "=== 執行實驗的指令 ==="
echo "ssh $SSH_OPTS ec2-user@$EC2_IP"
echo ""
echo "# 在 EC2 上執行 (使用 nohup 防止斷線中斷):"
echo "cd $REMOTE_DIR"
echo "nohup python3 batch_hw_noise_comparison.py --graph_dir ./graph --start_graph 1 --end_graph 81 > batch_run.log 2>&1 &"
echo "# 查看進度: tail -f batch_run.log"
