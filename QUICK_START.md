# 快速入門指南

## 🚀 3 分鐘快速開始

### 步驟 1：確認環境

```bash
# 確認 Python 環境
python --version  # 需要 Python 3.7+

# 確認必要套件
pip install numpy pandas matplotlib openpyxl
```

### 步驟 2：確認圖檔

```bash
# 確認 graph 目錄存在且包含圖檔
ls graph/
# 應該看到 G1.txt, G2.txt, ... G81.txt
```

### 步驟 3：快速測試（約 5 分鐘）

```bash
python test_batch_hardware.py
```

這會測試 G1~G3，共 12 個實驗，確認系統運作正常。

### 步驟 4：完整執行（約 5-6 小時）

```bash
# 選項 1：直接執行
python batch_hardware_comparison.py

# 選項 2：在背景執行（推薦）
nohup python batch_hardware_comparison.py > batch_log.txt 2>&1 &

# 選項 3：部分執行（例如 G1~G20）
python batch_hardware_comparison.py --start_graph 1 --end_graph 20
```

## 📊 結果在哪裡？

執行完成後，結果會在：

```
hardware_comparison_results/
├── trial100_steps1000/      # 100 次試驗 × 1,000 步
├── trial100_steps10000/     # 100 次試驗 × 10,000 步
├── trial1000_steps100/      # 1,000 次試驗 × 100 步
└── trial1000_steps10000/    # 1,000 次試驗 × 10,000 步
```

每個資料夾包含所有圖的結果：
- `.csv` - 統計摘要
- `.png` - 視覺化圖表
- `.xlsx` - 詳細數據

## 🔍 檢查進度

```bash
# 查看執行 log
tail -f batch_log.txt

# 查看目前完成的實驗數
find hardware_comparison_results -name "*.csv" | wc -l
```

## 💡 常用命令

```bash
# 預覽執行計劃（不實際執行）
python preview_batch_hardware.py --start 1 --end 10

# 只測試小圖（G1~G5）
python batch_hardware_comparison.py --start_graph 1 --end_graph 5

# 使用線性退火
python batch_hardware_comparison.py --schedule linear

# 調整 RPA 更新比例
python batch_hardware_comparison.py --epsilon 0.2
```

## 📖 詳細文檔

- **完整說明**：`批量硬體比較使用說明.md`
- **系統總覽**：`BATCH_HARDWARE_README.md`
- **演算法說明**：見 `hardware_multiple_spin_probit_annealing.py` 頂部註解

## ❓ 遇到問題？

### 問題 1：找不到圖檔

```bash
# 確認圖檔位置
ls graph/G1.txt
# 如果在其他位置，使用 --graph_dir 指定
python batch_hardware_comparison.py --graph_dir /path/to/graphs
```

### 問題 2：執行失敗

```bash
# 先測試單一實驗
python hardware_multiple_spin_probit_annealing.py \
    --file_path graph/G1.txt \
    --trial 10 \
    --timesteps 100
```

### 問題 3：執行太慢

```bash
# 減少測試範圍
python batch_hardware_comparison.py --start_graph 1 --end_graph 10
```

## 📝 快速測試清單

- [ ] 環境確認：`python --version`
- [ ] 套件確認：`pip list | grep numpy`
- [ ] 圖檔確認：`ls graph/G1.txt`
- [ ] 單一測試：`python hardware_multiple_spin_probit_annealing.py --file_path graph/G1.txt --trial 10 --timesteps 100`
- [ ] 預覽計劃：`python preview_batch_hardware.py --start 1 --end 3`
- [ ] 快速測試：`python test_batch_hardware.py`
- [ ] 完整執行：`python batch_hardware_comparison.py`

## 🎯 預期結果

完整執行（G1~G81）會產生：
- 324 個實驗（81 個圖 × 4 種參數組合）
- 約 1,296 個檔案（每個實驗 4 個檔案）
- 約 1-2 GB 磁碟空間
- 執行時間：5-6 小時（取決於硬體）

## 🔧 進階設定

修改 `batch_hardware_comparison.py` 中的 `configs` 可以自定義測試參數：

```python
configs = [
    ('trial100_steps1000', 100, 1000),
    ('trial100_steps10000', 100, 10000),
    ('trial1000_steps100', 1000, 100),
    ('trial1000_steps10000', 1000, 10000),
    # 新增自定義參數
    # ('trial500_steps5000', 500, 5000),
]
```

## 📞 需要幫助？

1. 查看詳細文檔：`批量硬體比較使用說明.md`
2. 檢查演算法說明：`hardware_multiple_spin_probit_annealing.py`
3. 查看執行 log：`batch_log.txt`

---

**提示**：首次使用建議先執行 `test_batch_hardware.py` 確認系統正常！

