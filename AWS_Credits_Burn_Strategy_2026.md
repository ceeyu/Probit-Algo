# 如何在 48 小時內有效消耗 $348 AWS Community Builder Credits：一份實戰紀錄

> 本文記錄一位 AWS Community Builder 在 credits 到期前，如何結合 EC2 運算實驗與 Amazon Bedrock 生成式 AI 翻譯，在有限時間內最大化 credits 使用效益的完整過程。包含遇到的技術障礙、除錯過程，以及最終的費用分析。

## 背景

身為 AWS Community Builder（Cloud Operations 類別），我持有 $348.19 的 AWS Promotional Credits，到期日為 **2026-02-28 UTC 午夜**（台灣時間 2026-03-01 07:59:59）。這些 credits 不可延期、不可轉讓，過期即作廢。

我手上正好有一個 CPU 密集型的學術研究計算專案 — **硬體層級 Probit 類比退火法（Simulated Annealing）** 的大規模基準測試，需要對 GSET benchmark 的 G1~G81 圖（800~20,000 節點）執行上千次 trial。這是一個將 credits 轉化為實際研究產出的好機會。

## 策略規劃

### 可用方案評估

| 方案 | 預估費率 | 可行性 | 備註 |
|------|---------|--------|------|
| EC2 On-Demand (CPU) | $0.68~$8.02/hr | ✅ 立即可用 | Standard quota 32 vCPU |
| EC2 GPU instances | $12+/hr | ❌ Quota 為 0 | 申請需 1-3 天 |
| Amazon Bedrock API | $15/$75 per 1M tokens (I/O) | ✅ 已開通 | Claude Opus 系列 |
| Windows + SQL Server EC2 | ~$18.91/hr | ✅ 可用 | License 附加費高 |
| Kiro 訂閱 | $20~$200/月 | ⚠️ 效率低 | 按比例只扣 ~$7/天 |

### 最終策略

1. **主力**：開 EC2 instances 跑研究實驗（持續 20+ 小時）
2. **輔助**：用 Amazon Bedrock Claude Opus 做文件翻譯（按 token 計費，燒錢快）

## 第一階段：EC2 環境建置與 Quota 提升

### 1.1 發現 GPU Quota 限制

第一個想法是用 GPU instance 加速計算，但查詢後發現所有加速運算類型的 quota 都是 0：

```bash
# 查詢各類 instance quota
aws service-quotas get-service-quota \
  --service-code ec2 \
  --quota-code L-DB2E81BA \  # P instances (GPU)
  --region ap-northeast-1

# 結果：P, G, DL, Inf, Trn 全部 quota = 0
# 只有 Standard instances (A, C, D, H, I, M, R, T, Z) 有 32 vCPU
```

GPU quota 提升申請通常需要 1-3 天審核，credits 隔天就到期，來不及。這裡要注意的是，quota 提升是向 AWS Service Quotas 申請，不是向 root user 申請 — 這是 AWS 帳戶層級的資源限制。

### 1.2 Standard Instance Quota 提升

32 vCPU 太少，申請提升到 192 vCPU：

```bash
aws service-quotas request-service-quota-increase \
  --service-code ec2 \
  --quota-code L-1216C47A \
  --desired-value 192 \
  --region ap-northeast-1
```

Standard instance 的 quota 提升通常很快，這次幾分鐘內就核准了（CASE_CLOSED）。

### 1.3 Key Pair 建立的 PowerShell 編碼陷阱

在 Windows PowerShell 環境下建立 SSH key pair 時，遇到了一個經典問題：

```powershell
# 第一次嘗試 — 失敗
aws ec2 create-key-pair --key-name credit-burn-key \
  --query 'KeyMaterial' --output text > credit-burn-key.pem

# SSH 連線時報錯：Load key "credit-burn-key.pem": error in libcrypto
# 原因：PowerShell 的 > 重導向會加入 BOM 並使用 UTF-16 編碼
```

PEM 檔案必須是 UTF-8 without BOM。經過多次嘗試，最終解法：

```powershell
# 正確做法：用 .NET 方法寫入純 UTF-8
$key = aws ec2 create-key-pair --key-name credit-burn-key4 `
  --query 'KeyMaterial' --output text
[System.IO.File]::WriteAllText("$PWD\credit-burn-key4.pem", $key)
```

這個問題在 Windows 上使用 AWS CLI 時很常見，卻很少被文件提及。總共建了 4 個 key pair 才成功（credit-burn-key, key2, key3, key4）。

### 1.4 User Data 自動關機腳本

為了確保 credits 到期後不會繼續產生費用，設計了 user-data 腳本搭配 `instance-initiated-shutdown-behavior=terminate`：

```bash
#!/bin/bash
# ec2_userdata.sh — 自動關機 + 環境初始化

# 安裝 Python 環境
yum update -y
yum install -y python3.11 python3.11-pip git
pip3.11 install numpy pandas matplotlib openpyxl

# 計算距離目標時間的秒數，設定自動關機
TARGET_EPOCH=$(date -d "2026-02-28 23:50:00 UTC" +%s)
NOW_EPOCH=$(date +%s)
REMAINING=$((TARGET_EPOCH - NOW_EPOCH))

if [ $REMAINING -gt 0 ]; then
    shutdown -h +$((REMAINING / 60))
fi

mkdir -p /home/ec2-user/experiment
chown ec2-user:ec2-user /home/ec2-user/experiment
```

搭配啟動參數：

```bash
aws ec2 run-instances \
  --instance-type c5.4xlarge \
  --instance-initiated-shutdown-behavior terminate \
  --user-data file://ec2_userdata.sh \
  ...
```

這樣 shutdown 觸發時會直接 terminate instance，EBS volume 也會一併刪除，不會有殘留費用。

## 第二階段：部署實驗程式碼

### 2.1 Instance 選擇

最終開了兩台 instance：

| Instance ID | 類型 | vCPU | RAM | 費率 | 用途 |
|-------------|------|------|-----|------|------|
| i-021e7323bdd9382ed | c5.4xlarge | 16 | 32 GiB | $0.68/hr | 先鋒機，測試部署流程 |
| i-0de719116716e9cba | r5dn.24xlarge | 96 | 768 GiB | $8.02/hr | 主力運算機 |

合計 112/192 vCPU，每小時 ~$8.70。

### 2.2 程式碼上傳與環境建置

```bash
# SSH 連線參數（避免 known_hosts 問題）
SSH_OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=NUL"

# 上傳核心程式碼
scp $SSH_OPTS -i credit-burn-key4.pem \
  hw_noise_probit_limit.py \
  batch_hw_noise_comparison.py \
  ec2-user@<IP>:/home/ec2-user/experiment/

# 上傳 GSET 圖檔（G1~G81）
scp $SSH_OPTS -i credit-burn-key4.pem \
  graph/*.txt \
  ec2-user@<IP>:/home/ec2-user/experiment/graph/

# 安裝 Python 套件
ssh $SSH_OPTS -i credit-burn-key4.pem ec2-user@<IP> \
  "pip3 install --user numpy pandas matplotlib openpyxl"
```

### 2.3 啟動實驗

```bash
# 在 EC2 上以 nohup 背景執行
ssh $SSH_OPTS -i credit-burn-key4.pem ec2-user@<IP> "
  cd /home/ec2-user/experiment
  nohup python3 -u batch_hw_noise_comparison.py \
    --graph_dir ./graph \
    --start_graph 1 \
    --end_graph 81 \
    > batch_noise_run.log 2>&1 &
"
```

### 2.4 遇到的問題：Python stdout Buffering

啟動後用 `tail` 查看 log，發現進度一直卡在同一行不動：

```bash
ssh ... "tail -10 /home/ec2-user/experiment/batch_noise_run.log"
# 輸出一直停在 Trial 548/1000，多次查詢都一樣
```

原因分析：
- `batch_hw_noise_comparison.py`（父程序）透過 `subprocess.run()` 呼叫 `hw_noise_probit_limit.py`（子程序）
- 即使父程序加了 `python3 -u` flag，子程序不會繼承 unbuffered 模式
- Python 的 stdout 在非 TTY 環境下預設使用 block buffering（通常 4KB~8KB）
- 子程序的 print 輸出被 buffer 住，要等整個 graph 跑完才會一次 flush

解決方案 — 在 `subprocess.run()` 的環境變數中加入 `PYTHONUNBUFFERED`：

```python
def run_single_experiment(...):
    env = os.environ.copy()
    env['HARDWARE_OUTPUT_DIR'] = output_dir
    env['PYTHONUNBUFFERED'] = '1'  # 讓子程序也 unbuffered
    
    result = subprocess.run(cmd, check=True, capture_output=False, text=True, env=env)
```

修改後重新上傳並重啟實驗，log 就能即時顯示每個 trial 的進度了。

### 2.5 監控進度

```bash
# 查看最新 log 輸出
ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=NUL \
  -i credit-burn-key4.pem ec2-user@<IP> \
  "tail -10 /home/ec2-user/experiment/batch_noise_run.log"

# 查看目前正在跑哪個 graph
ssh ... "ps -ef | grep hw_noise_probit_limit"
```

典型的 log 輸出：

```
===== Trial 368/1000 =====
[硬體模擬] J 範圍: [-1.0, 1.0] → J_hw 範圍: [0.0, 1.0]
  n (spin 數量) = 20000
  sigma 範圍: 5.0 → 0.01
  [Down Counter 模式] k 線性整數遞減: 50 → 0
  σ 為平方根降溫: 5.00 → 0.00（由 k 被動推導）
  控制電路：整數計數器，無需浮點運算 ← 最貼近硬體
[Probit] Energy: -24278.00, Cut: 12156.0, Time: 42012.67 ms
[Traditional SA] Energy: -1270.00, Cut: 652.0, Time: 1178.59 ms
```

## 第三階段：Amazon Bedrock 文件翻譯

### 3.1 評估 Bedrock 可行性

EC2 兩台預估只能燒 ~$190，離 $348 還有差距。評估後決定用 Amazon Bedrock 的 Claude Opus 系列做文件翻譯，output token 費率高達 $75/1M tokens，是快速消耗 credits 的好方法。

首先確認 model access 狀態：

```bash
# 列出可用的 Claude 模型
aws bedrock list-foundation-models \
  --region ap-northeast-1 \
  --query "modelSummaries[?contains(modelId,'claude')].{ID:modelId,Name:modelName}" \
  --output table
```

### 3.2 Inference Profile 的必要性

直接呼叫模型 ID 會失敗：

```bash
aws bedrock-runtime invoke-model \
  --model-id "anthropic.claude-opus-4-6-v1" \
  --body "fileb://request.json" \
  outfile.json

# 錯誤：Invocation of model ID anthropic.claude-opus-4-6-v1 with 
# on-demand throughput isn't supported. Retry your request with the 
# ID or ARN of an inference profile that contains this model.
```

較新的模型（如 Claude Opus 4.5/4.6）需要透過 Inference Profile 呼叫：

```bash
# 查看可用的 inference profiles
aws bedrock list-inference-profiles \
  --region ap-northeast-1 \
  --query "inferenceProfileSummaries[?contains(inferenceProfileId,'claude')]
           .[inferenceProfileId,inferenceProfileName,status]" \
  --output table

# 結果包含：
# global.anthropic.claude-opus-4-6-v1    | Global Anthropic Claude Opus 4.6  | ACTIVE
# us.anthropic.claude-opus-4-5-...       | ...                               | ACTIVE
```

使用 inference profile ID 即可成功呼叫：

```bash
aws bedrock-runtime invoke-model \
  --region ap-northeast-1 \
  --model-id "global.anthropic.claude-opus-4-6-v1" \
  --content-type "application/json" \
  --accept "application/json" \
  --body "fileb://request.json" \
  outfile.json

# 成功回應：
# {"model":"claude-opus-4-6","usage":{"input_tokens":14,"output_tokens":27}}
```

### 3.3 PowerShell JSON 編碼問題

在 Windows PowerShell 中使用 `--body` 參數傳 JSON 時，會遇到引號被吃掉的問題：

```powershell
# 失敗：PowerShell 會把單引號內的 JSON 破壞
aws bedrock-runtime invoke-model --body '{"messages":[...]}'
# 錯誤：Invalid base64

# 失敗：file:// 也不行（AWS CLI 會嘗試 base64 decode）
aws bedrock-runtime invoke-model --body "file://body.json"
# 錯誤：Invalid base64

# 成功：使用 fileb:// (binary file)
$body = @{...} | ConvertTo-Json -Depth 5 -Compress
[System.IO.File]::WriteAllText("$PWD\body.json", $body)
aws bedrock-runtime invoke-model --body "fileb://body.json" outfile.json
```

`file://` vs `fileb://` 的差異：前者會嘗試 base64 解碼，後者直接讀取二進位內容。對 JSON body 來說，`fileb://` 才是正確的。

### 3.4 在 AWS CloudShell 執行翻譯

最終在 AWS CloudShell（us-east-1）上執行翻譯腳本，翻譯技術文件。CloudShell 本身免費，只有 Bedrock API 呼叫會計費。

使用的模型：
- `global.anthropic.claude-opus-4-6-v1` — $15/$75 per 1M tokens (input/output)
- `us.anthropic.claude-opus-4-5-20251101-v1:0` — $15/$75 per 1M tokens (input/output)

## 第四階段：費用監控與追蹤

### 4.1 AWS Cost Explorer 的延遲

Cost Explorer 的資料通常有 **24 小時延遲**，對於即時監控不實用：

```bash
aws ce get-cost-and-usage \
  --time-period Start=2026-02-27,End=2026-03-01 \
  --granularity DAILY \
  --metrics "UnblendedCost" \
  --output json

# 結果：Amount = 0（延遲尚未反映）
```

如果要按服務篩選，需要用 `--filter` 參數搭配 JSON 檔案（PowerShell 環境下直接寫 JSON 會有引號問題）：

```powershell
# 建立 filter JSON
[System.IO.File]::WriteAllText("$PWD\ce_filter.json",
  '{"Dimensions":{"Key":"SERVICE","Values":["Amazon Elastic Compute Cloud - Compute"]}}')

aws ce get-cost-and-usage \
  --time-period Start=2026-02-27,End=2026-03-01 \
  --granularity DAILY \
  --metrics "UnblendedCost" \
  --filter file://ce_filter.json
```

### 4.2 用 CloudWatch Metrics 即時追蹤 Bedrock 用量

Bedrock 的 token 用量會即時寫入 CloudWatch（延遲約 5-10 分鐘），這是最接近即時的監控方式：

```bash
# 查詢 Output Token 總量（按模型分）
aws cloudwatch get-metric-statistics \
  --region us-east-1 \
  --namespace "AWS/Bedrock" \
  --metric-name "OutputTokenCount" \
  --start-time "2026-02-28T00:00:00Z" \
  --end-time "2026-02-28T23:59:59Z" \
  --period 86400 \
  --statistics Sum \
  --dimensions "Name=ModelId,Value=global.anthropic.claude-opus-4-6-v1" \
  --output json
```

查看有哪些模型產生了用量：

```bash
aws cloudwatch list-metrics \
  --region us-east-1 \
  --namespace "AWS/Bedrock" \
  --metric-name "OutputTokenCount" \
  --output table
```

### 4.3 EC2 費用的手動計算

由於 Cost Explorer 延遲，EC2 費用需要根據 launch time 手動計算：

```bash
# 查詢 instance 啟動時間
aws ec2 describe-instances \
  --region ap-northeast-1 \
  --instance-ids i-021e7323bdd9382ed i-0de719116716e9cba \
  --query "Reservations[].Instances[].[InstanceId,InstanceType,LaunchTime]" \
  --output text

# i-021e7323bdd9382ed  c5.4xlarge    2026-02-27T10:29:46+00:00
# i-0de719116716e9cba  r5dn.24xlarge 2026-02-27T13:42:38+00:00
```

計算公式：`費用 = (當前時間 - LaunchTime) × 每小時費率`

### 4.4 全帳戶資源盤點

為確保沒有遺漏的計費項目，進行全面掃描：

```bash
# EC2 instances
aws ec2 describe-instances --filters "Name=instance-state-name,Values=running"

# EBS volumes（未附加的 volume 也會計費）
aws ec2 describe-volumes --query "Volumes[].[VolumeId,Size,VolumeType,State]"

# Elastic IPs（未關聯的 EIP 會計費）
aws ec2 describe-addresses

# NAT Gateways（$0.062/hr）
aws ec2 describe-nat-gateways --filter "Name=state,Values=available"

# Load Balancers
aws elbv2 describe-load-balancers

# RDS instances
aws rds describe-db-instances

# SageMaker endpoints & notebooks
aws sagemaker list-endpoints
aws sagemaker list-notebook-instances

# Bedrock Provisioned Throughput（持續計費）
aws bedrock list-provisioned-model-throughputs

# VPC Endpoints（Interface type 有費用）
aws ec2 describe-vpc-endpoints

# S3 buckets
aws s3 ls

# Lambda functions
aws lambda list-functions
```

這個盤點流程建議在每次大規模使用 AWS 資源後都執行一次，避免遺忘的資源持續產生費用。

## 費用總結

### EC2 費用（根據運行時間計算）

| Instance | 類型 | 運行時間 | 費率 | 費用 |
|----------|------|---------|------|------|
| i-021e7...382ed | c5.4xlarge | ~25 hr | $0.680/hr | ~$17.00 |
| i-0de71...e9cba | r5dn.24xlarge | ~22 hr | $8.016/hr | ~$176.35 |
| 已終止的 3 台測試機 | c5.4xlarge | ~2 hr total | $0.680/hr | ~$1.36 |
| **EC2 小計** | | | | **~$194.71** |

### Bedrock 費用（CloudWatch Metrics）

| 模型 | Input Tokens | Output Tokens | Input 費用 | Output 費用 | 小計 |
|------|-------------|---------------|-----------|------------|------|
| Claude Opus 4.6 (global) | 412,224 | 463,442 | $6.18 | $34.76 | $40.94 |
| Claude Opus 4.5 (us) | 1,248,661 | 1,985,492 | $18.73 | $148.91 | $167.64 |
| **Bedrock 小計** | **1,660,885** | **2,448,934** | | | **$208.58** |

### 總計

| 服務 | 費用 |
|------|------|
| Amazon EC2 | ~$194.71 |
| Amazon Bedrock | ~$208.58 |
| **總計** | **~$403.29** |
| Credits 額度 | $348.19 |
| 超出部分（信用卡扣款） | ~$55.10 |

> 注意：CloudWatch Metrics 有 5-10 分鐘延遲，實際 Bedrock 費用可能略高於上述數字。

## 踩坑紀錄與經驗教訓

### 1. PowerShell 是 AWS CLI 的地雷區

Windows PowerShell 環境下使用 AWS CLI，至少遇到三類編碼問題：

- **PEM 檔案編碼**：`>` 重導向產生 UTF-16 BOM，SSH 無法讀取。解法：`[System.IO.File]::WriteAllText()`
- **JSON 參數**：單引號內的 JSON 會被 PowerShell 解析破壞。解法：寫入檔案後用 `fileb://` 引用
- **`$` 符號**：PowerShell 會將 `$` 視為變數前綴。在 Python `-c` 指令中使用 `$` 會被吃掉

建議：在 Windows 上操作 AWS CLI，優先使用 WSL 或 Git Bash，避免 PowerShell 的特殊字元處理。

### 2. Python subprocess 的 stdout Buffering

當 Python 程式透過 `subprocess.run()` 呼叫子程序時，子程序的 stdout 在非 TTY 環境下會使用 block buffering。即使父程序加了 `-u` flag 也無效。

解法：在 `subprocess.run()` 的 `env` 參數中傳入 `PYTHONUNBUFFERED=1`，讓子程序繼承 unbuffered 模式。

### 3. Bedrock Inference Profile vs Model ID

較新的 Bedrock 模型（Claude Opus 4.5+）不支援直接用 model ID 呼叫 on-demand throughput，必須使用 Inference Profile ID（如 `global.anthropic.claude-opus-4-6-v1`）。這在官方文件中不夠顯眼，容易踩坑。

### 4. Cost Explorer 不適合即時監控

Cost Explorer 有 24 小時延遲，不適合用來追蹤當天的費用。替代方案：
- **EC2**：根據 `LaunchTime` 和 on-demand 費率手動計算
- **Bedrock**：用 CloudWatch Metrics（`AWS/Bedrock` namespace）查詢 token 用量，延遲約 5-10 分鐘

### 5. GPU Instance Quota 需要提前申請

所有 GPU/加速運算 instance（P, G, DL, Inf, Trn）的預設 quota 都是 0，提升申請需要 1-3 個工作天。如果有使用 GPU 的需求，務必提前至少一週申請。Standard instance 的 quota 提升則通常在幾分鐘內核准。

### 6. EC2 Terminate 是不可逆的

`terminate` 操作會立即銷毀 instance 和附帶的 EBS volume，資料無法恢復。在操作前務必確認：
- 重要資料已經下載或備份
- 確認是要 `stop`（可恢復）還是 `terminate`（不可逆）

如果只是想暫停計費但保留資料，應該使用 `stop-instances` 而非 `terminate-instances`。Stop 狀態下 EBS volume 仍會計費（約 $0.08/GB/月），但 instance 本身不計費。

## 適用場景

這套策略適合以下情境：

1. **AWS Credits 即將到期**：Community Builder、Activate、EdStart 等計畫的 promotional credits
2. **有 CPU 密集型工作負載**：科學計算、資料處理、批次任務
3. **需要大量 AI/ML 推論**：文件翻譯、內容生成、資料標註
4. **時間緊迫**：credits 到期前 24-48 小時的緊急消耗

## 結語

透過 EC2 運算實驗 + Bedrock 文件翻譯的組合策略，在約 30 小時內成功消耗了超過 $400 的 AWS 資源。雖然超出了 credits 額度約 $55，但將原本會浪費的 $348 credits 轉化為了實際的研究計算結果和翻譯文件產出。

最重要的經驗是：**提前規劃**。如果能在 credits 到期前一週就開始準備，可以申請 GPU quota、更從容地選擇最佳方案，而不是在最後 48 小時內倉促行動。

---

*本文作者為 AWS Community Builder（Cloud Operations 類別）。文中所有 IP 位址、Instance ID 等資訊均為實際操作記錄，但相關 AWS 帳戶的敏感憑證資訊（如 Access Key、PEM 私鑰內容）已省略。*
