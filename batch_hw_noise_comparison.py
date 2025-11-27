import os
import sys
import subprocess
import datetime
import argparse
from pathlib import Path

"""
批次執行硬體 Probit 類比退火法比較實驗
=======================================

功能：
1. 對 G1~G81 的所有圖檔執行實驗
2. 測試 4 種參數組合：
   - trial=100, timestep=1000
   - trial=100, timestep=10000
   - trial=1000, timestep=100
   - trial=1000, timestep=10000
3. 結果分別存到 4 個資料夾

使用方式：
    python batch_hw_noise_comparison.py --graph_dir ./graph --start_graph 1 --end_graph 81
"""

def parse_arguments():
    parser = argparse.ArgumentParser(description="批次執行硬體 Probit 類比退火法比較實驗")
    parser.add_argument('--graph_dir', type=str, default='./graph', 
                       help="圖檔所在目錄 (預設: ./graph)")
    parser.add_argument('--start_graph', type=int, default=1, 
                       help="起始圖檔編號 (預設: 1)")
    parser.add_argument('--end_graph', type=int, default=81, 
                       help="結束圖檔編號 (預設: 81)")
    parser.add_argument('--sigma_start', type=float, default=5.0, 
                       help="起始 sigma (預設: 5.0)")
    parser.add_argument('--sigma_end', type=float, default=0.01, 
                       help="結束 sigma (預設: 0.01)")
    parser.add_argument('--T_start', type=float, default=5.0, 
                       help="起始溫度 (預設: 5.0)")
    parser.add_argument('--T_end', type=float, default=0.01, 
                       help="結束溫度 (預設: 0.01)")
    parser.add_argument('--schedule', type=str, default='linear', 
                       choices=['exponential', 'linear'],
                       help="退火排程 (預設: linear)")
    parser.add_argument('--probit_mode', type=str, default='synchronous', 
                       choices=['synchronous', 'asynchronous'],
                       help="Probit 更新模式 (預設: synchronous)")
    parser.add_argument('--epsilon', type=float, default=0.1,
                       help="RPA 更新比例 (預設: 0.1)")
    parser.add_argument('--dry_run', action='store_true',
                       help="僅顯示要執行的命令，不實際執行")
    return parser.parse_args()

def get_graph_files(graph_dir, start_idx, end_idx):
    """
    取得指定範圍的圖檔路徑
    
    參數:
        graph_dir: 圖檔目錄
        start_idx: 起始編號
        end_idx: 結束編號
    
    返回:
        list of (graph_number, file_path)
    """
    graph_files = []
    graph_path = Path(graph_dir)
    
    if not graph_path.exists():
        print(f"錯誤：圖檔目錄不存在 - {graph_dir}")
        sys.exit(1)
    
    for i in range(start_idx, end_idx + 1):
        # GSET 圖檔命名格式通常是 G1.txt, G2.txt, ..., G81.txt
        graph_file = graph_path / f"G{i}.txt"
        if graph_file.exists():
            graph_files.append((i, str(graph_file)))
        else:
            print(f"警告：找不到圖檔 G{i}.txt，跳過")
    
    return graph_files

def create_output_directories(base_dir='./noise_hardware_comparison_limit_results'):
    """
    創建輸出目錄結構
    
    結構：
        noise_hardware_comparison_results/
        ├── trial100_steps1000/
        ├── trial100_steps10000/
        ├── trial1000_steps100/
        └── trial1000_steps10000/
    """
    configs = [
       ('trial100_steps1000', 100, 1000),
       # ('trial100_steps10000', 100, 10000),
       # ('trial1000_steps100', 1000, 100),
       # ('trial1000_steps10000', 1000, 10000)
       # ('trial1000_steps1000', 1000, 1000) #先測試1000 1000，因為10000跑太久crash
    ]
    
    base_path = Path(base_dir)
    base_path.mkdir(exist_ok=True)
    
    output_dirs = {}
    for dir_name, trial, steps in configs:
        dir_path = base_path / dir_name
        dir_path.mkdir(exist_ok=True)
        output_dirs[(trial, steps)] = str(dir_path)
        print(f"創建輸出目錄: {dir_path}")
    
    return output_dirs, configs

def run_single_experiment(graph_file, graph_num, trial, timesteps, output_dir, args, dry_run=False):
    """
    執行單一實驗
    
    參數:
        graph_file: 圖檔路徑
        graph_num: 圖檔編號
        trial: 試驗次數
        timesteps: 退火步數
        output_dir: 輸出目錄
        args: 命令列參數
        dry_run: 是否僅顯示命令
    """
    cmd = [
        sys.executable,  # Python 解釋器
        'hw_noise_probit_limit.py',
        '--file_path', graph_file,
        '--trial', str(trial),
        '--timesteps', str(timesteps),
        '--sigma_start', str(args.sigma_start),
        '--sigma_end', str(args.sigma_end),
        '--T_start', str(args.T_start),
        '--T_end', str(args.T_end),
        '--schedule', args.schedule,
        '--probit_mode', args.probit_mode,
        '--epsilon', str(args.epsilon)
    ]
    
    cmd_str = ' '.join(cmd)
    
    if dry_run:
        print(f"[Dry Run] {cmd_str}")
        return True
    
    print(f"\n{'='*80}")
    print(f"執行: G{graph_num} | Trial={trial}, Timesteps={timesteps}")
    print(f"{'='*80}")
    print(f"命令: {cmd_str}\n")
    
    try:
        # 設置環境變數，讓子程序知道輸出目錄
        env = os.environ.copy()
        env['HARDWARE_OUTPUT_DIR'] = output_dir
        
        result = subprocess.run(cmd, check=True, capture_output=False, text=True, env=env)
        print(f"✓ 完成: G{graph_num} | Trial={trial}, Timesteps={timesteps}")
        print(f"  結果已儲存至: {output_dir}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ 錯誤: G{graph_num} | Trial={trial}, Timesteps={timesteps}")
        print(f"  錯誤訊息: {e}")
        return False
    except KeyboardInterrupt:
        print("\n\n中斷執行")
        sys.exit(1)

def main():
    args = parse_arguments()
    
    print("="*80)
    print("批次執行硬體 Probit 類比退火法比較實驗")
    print("="*80)
    print(f"圖檔目錄: {args.graph_dir}")
    print(f"圖檔範圍: G{args.start_graph} ~ G{args.end_graph}")
    print(f"Probit 模式: {args.probit_mode}")
    print(f"退火排程: {args.schedule}")
    print(f"參數: sigma=[{args.sigma_start}, {args.sigma_end}], T=[{args.T_start}, {args.T_end}]")
    if args.probit_mode == 'synchronous':
        print(f"RPA epsilon: {args.epsilon}")
    print("="*80 + "\n")
    
    # 取得圖檔列表
    graph_files = get_graph_files(args.graph_dir, args.start_graph, args.end_graph)
    print(f"找到 {len(graph_files)} 個圖檔\n")
    
    if len(graph_files) == 0:
        print("錯誤：沒有找到任何圖檔")
        sys.exit(1)
    
    # 創建輸出目錄
    output_dirs, configs = create_output_directories()
    print()
    
    # 統計資訊
    total_experiments = len(graph_files) * len(configs)
    current_experiment = 0
    failed_experiments = []
    
    start_time = datetime.datetime.now()
    print(f"開始時間: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"總實驗數: {total_experiments}\n")
    
    # 主循環：對每個參數組合
    for dir_name, trial, timesteps in configs:
        output_dir = output_dirs[(trial, timesteps)]
        
        print("\n" + "="*80)
        print(f"參數組合: Trial={trial}, Timesteps={timesteps}")
        print(f"輸出目錄: {output_dir}")
        print("="*80)
        
        # 對每個圖檔執行實驗
        for graph_num, graph_file in graph_files:
            current_experiment += 1
            
            print(f"\n[{current_experiment}/{total_experiments}] 處理: G{graph_num}")
            
            # 執行實驗
            success = run_single_experiment(
                graph_file, graph_num, trial, timesteps, 
                output_dir, args, dry_run=args.dry_run
            )
            
            if not success:
                failed_experiments.append((graph_num, trial, timesteps))
    
    # 總結
    end_time = datetime.datetime.now()
    duration = end_time - start_time
    
    print("\n" + "="*80)
    print("批次執行完成")
    print("="*80)
    print(f"開始時間: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"結束時間: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"總耗時: {duration}")
    print(f"總實驗數: {total_experiments}")
    print(f"成功: {total_experiments - len(failed_experiments)}")
    print(f"失敗: {len(failed_experiments)}")
    
    if failed_experiments:
        print("\n失敗的實驗:")
        for graph_num, trial, timesteps in failed_experiments:
            print(f"  - G{graph_num} | Trial={trial}, Timesteps={timesteps}")
    
    print("="*80)

if __name__ == "__main__":
    main()

