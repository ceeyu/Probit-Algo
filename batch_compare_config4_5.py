#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Config 4 vs Config 5 批量比較腳本
自動對所有圖檔執行兩種配置並生成比較結果
"""

import os
import glob
import subprocess
import time
import pandas as pd
import re
import datetime
from pathlib import Path
import shutil

class Config45Comparator:
    def __init__(self):
        self.graph_dir = "./graph"
        self.output_dir = "./0711compare4_5"
        self.results = []
        self.energy_plots_dir = "./0708_energy_plots"
        
        # 創建輸出目錄
        os.makedirs(self.output_dir, exist_ok=True)
        
    def get_graph_files(self):
        """獲取所有圖檔"""
        pattern = os.path.join(self.graph_dir, "G*.txt")
        graph_files = sorted(glob.glob(pattern))
        print(f"找到 {len(graph_files)} 個圖檔:")
        for i, file in enumerate(graph_files, 1):
            graph_name = os.path.basename(file)
            print(f"  {i:2d}. {graph_name}")
        return graph_files
    
    def run_single_config(self, graph_file, config):
        """執行單個配置"""
        graph_name = os.path.basename(graph_file).replace('.txt', '')
        print(f"\n{'='*60}")
        print(f"執行 {graph_name} - Config {config}")
        print(f"{'='*60}")
        
        if config == 4:
            cmd = [
                'python', 'gpu_MAXCUT_var0708.py',
                '--config', '4',
                '--file_path', graph_file,
                '--cycle', '1000',
                '--trial', '10', 
                '--tau', '8',
                '--res', '2',
                '--l_scale', '0.1',
                '--d_scale', '0.1',
                '--n_scale', '0.1'
            ]
        elif config == 5:
            cmd = [
                'python', 'gpu_MAXCUT_var0708.py',
                '--config', '5',
                '--file_path', graph_file,
                '--cycle', '1000',
                '--trial', '10',
                '--tau', '8', 
                '--res', '2',
                '--l_scale', '0.1',
                '--d_scale', '0.1',
                '--n_scale', '0.1',
                '--stall_threshold', '400',
                '--eta_alpha', '0.0',
                '--eta_beta', '0.0005',
                '--target_drop', '20',
                '--window_size', '200'
            ]
        
        start_time = time.time()
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            end_time = time.time()
            
            if result.returncode == 0:
                print(f"✅ Config {config} 執行成功 ({end_time - start_time:.1f}秒)")
                return self.parse_output(result.stdout, graph_name, config, end_time - start_time)
            else:
                print(f"❌ Config {config} 執行失敗")
                print("錯誤:", result.stderr)
                return None
                
        except subprocess.TimeoutExpired:
            print(f"⏱️ Config {config} 執行超時 (10分鐘)")
            return None
        except Exception as e:
            print(f"❌ Config {config} 執行錯誤: {e}")
            return None
    
    def parse_output(self, output, graph_name, config, execution_time):
        """解析命令行輸出提取統計數據"""
        result = {
            'Graph': graph_name,
            'Config': config,
            'ExecutionTime': execution_time
        }
        
        lines = output.split('\n')
        for line in lines:
            line = line.strip()
            if 'Average cut:' in line:
                result['AvgCut'] = float(re.search(r'Average cut:\s*([\d.]+)', line).group(1))
            elif 'Maximum cut:' in line:
                result['MaxCut'] = int(re.search(r'Maximum cut:\s*(\d+)', line).group(1))
            elif 'Minimum cut:' in line:
                result['MinCut'] = int(re.search(r'Minimum cut:\s*(\d+)', line).group(1))
            elif 'Average annealing time:' in line:
                result['AvgTime'] = float(re.search(r'Average annealing time:\s*([\d.]+)', line).group(1))
            elif 'Average energy:' in line:
                result['AvgEnergy'] = float(re.search(r'Average energy:\s*([-\d.]+)', line).group(1))
            elif 'TTS(0.99):' in line:
                tts_match = re.search(r'TTS\(0\.99\):\s*(\S+)', line)
                result['TTS'] = tts_match.group(1) if tts_match else 'None'
            elif 'Average reachability [%]:' in line:
                result['AvgReachability'] = float(re.search(r'Average reachability \[%\]:\s*([\d.]+)', line).group(1))
            elif 'Maximum reachability [%]:' in line:
                result['MaxReachability'] = float(re.search(r'Maximum reachability \[%\]:\s*([\d.]+)', line).group(1))
            elif 'Std of cut value:' in line:
                result['StdCut'] = float(re.search(r'Std of cut value:\s*([\d.]+)', line).group(1))
            elif 'Total time:' in line:
                result['TotalTime'] = float(re.search(r'Total time:\s*([\d.]+)', line).group(1))
        
        return result
    
    def find_latest_energy_excel(self, graph_name, config):
        """找到最新生成的energy excel檔案"""
        pattern = f"{self.energy_plots_dir}/energy_vs_cycles_{graph_name}_config{config}_*.xlsx"
        files = glob.glob(pattern)
        if files:
            # 按修改時間排序，返回最新的
            latest_file = max(files, key=os.path.getmtime)
            return latest_file
        return None
    
    def create_comparison_plot(self, graph_name, config4_excel, config5_excel):
        """為單個圖創建比較圖表"""
        if not config4_excel or not config5_excel:
            print(f"⚠️ {graph_name}: 缺少Excel檔案，跳過比較圖生成")
            return False
        
        print(f"📊 生成 {graph_name} 的比較圖表...")
        
        try:
            cmd = [
                'python', 'compare_energy_plots_cli.py',
                '--file1', config4_excel,
                '--file2', config5_excel,
                '--output_dir', self.output_dir,
                '--label1', f'{graph_name}_Config4_SpSA',
                '--label2', f'{graph_name}_Config5_ApSA'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            if result.returncode == 0:
                print(f"✅ {graph_name} 比較圖表生成成功")
                return True
            else:
                print(f"❌ {graph_name} 比較圖表生成失敗: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"❌ {graph_name} 比較圖表生成錯誤: {e}")
            return False
    
    def create_summary_excel(self):
        """創建總結Excel表"""
        if not self.results:
            print("❌ 沒有結果數據，無法創建總結表")
            return
        
        print("\n📊 生成總結Excel表...")
        
        # 轉換為DataFrame
        df = pd.DataFrame(self.results)
        
        # 重新排列列的順序
        column_order = [
            'Graph', 'Config', 'AvgCut', 'MaxCut', 'MinCut', 'StdCut',
            'AvgReachability', 'MaxReachability', 'AvgEnergy', 'AvgTime', 
            'TotalTime', 'ExecutionTime', 'TTS'
        ]
        
        # 確保所有列都存在
        for col in column_order:
            if col not in df.columns:
                df[col] = 'N/A'
        
        df = df[column_order]
        
        # 創建多sheet Excel
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        excel_path = os.path.join(self.output_dir, f"config4_vs_5_summary_{timestamp}.xlsx")
        
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            # 所有數據
            df.to_excel(writer, sheet_name='All_Results', index=False)
            
            # Config 4 數據
            config4_df = df[df['Config'] == 4].copy()
            if not config4_df.empty:
                config4_df.to_excel(writer, sheet_name='Config4_SpSA', index=False)
            
            # Config 5 數據  
            config5_df = df[df['Config'] == 5].copy()
            if not config5_df.empty:
                config5_df.to_excel(writer, sheet_name='Config5_ApSA', index=False)
            
            # 比較統計
            comparison_data = []
            graphs = df['Graph'].unique()
            
            for graph in graphs:
                graph_data = df[df['Graph'] == graph]
                if len(graph_data) == 2:  # 兩個config都有數據
                    config4_data = graph_data[graph_data['Config'] == 4].iloc[0]
                    config5_data = graph_data[graph_data['Config'] == 5].iloc[0]
                    
                    comparison = {
                        'Graph': graph,
                        'Config4_AvgCut': config4_data['AvgCut'],
                        'Config5_AvgCut': config5_data['AvgCut'],
                        'Cut_Improvement': config5_data['AvgCut'] - config4_data['AvgCut'],
                        'Cut_Improvement_Pct': ((config5_data['AvgCut'] - config4_data['AvgCut']) / config4_data['AvgCut'] * 100) if config4_data['AvgCut'] != 0 else 0,
                        'Config4_AvgTime': config4_data['AvgTime'],
                        'Config5_AvgTime': config5_data['AvgTime'],
                        'Time_Speedup': config4_data['AvgTime'] / config5_data['AvgTime'] if config5_data['AvgTime'] != 0 else 0,
                        'Config4_Reachability': config4_data['AvgReachability'],
                        'Config5_Reachability': config5_data['AvgReachability'],
                        'Reachability_Improvement': config5_data['AvgReachability'] - config4_data['AvgReachability']
                    }
                    comparison_data.append(comparison)
            
            if comparison_data:
                comparison_df = pd.DataFrame(comparison_data)
                comparison_df.to_excel(writer, sheet_name='Comparison', index=False)
        
        print(f"✅ 總結Excel表已儲存: {excel_path}")
        
        # 顯示快速統計
        print(f"\n📈 快速統計:")
        print(f"總共測試: {len(graphs)} 個圖檔")
        print(f"Config 4 結果: {len(config4_df)} 個")
        print(f"Config 5 結果: {len(config5_df)} 個")
        if comparison_data:
            avg_improvement = sum(c['Cut_Improvement'] for c in comparison_data) / len(comparison_data)
            avg_speedup = sum(c['Time_Speedup'] for c in comparison_data) / len(comparison_data)
            print(f"平均Cut改善: {avg_improvement:.2f}")
            print(f"平均時間比例: {avg_speedup:.2f}x")
    
    def run_comparison(self):
        """執行完整的比較流程"""
        print("🚀 開始Config 4 vs Config 5批量比較測試")
        print(f"輸出目錄: {self.output_dir}")
        
        graph_files = self.get_graph_files()
        if not graph_files:
            print("❌ 未找到圖檔")
            return
        
        total_graphs = len(graph_files)
        
        for i, graph_file in enumerate(graph_files, 1):
            graph_name = os.path.basename(graph_file).replace('.txt', '')
            print(f"\n🔄 處理圖檔 {i}/{total_graphs}: {graph_name}")
            
            # 執行Config 4
            config4_result = self.run_single_config(graph_file, 4)
            if config4_result:
                self.results.append(config4_result)
            
            # 等待一下確保檔案系統同步
            time.sleep(2)
            
            # 執行Config 5  
            config5_result = self.run_single_config(graph_file, 5)
            if config5_result:
                self.results.append(config5_result)
            
            # 等待一下確保檔案系統同步
            time.sleep(2)
            
            # 找到對應的Excel檔案並生成比較圖
            config4_excel = self.find_latest_energy_excel(graph_name, 4)
            config5_excel = self.find_latest_energy_excel(graph_name, 5)
            
            if config4_excel and config5_excel:
                self.create_comparison_plot(graph_name, config4_excel, config5_excel)
            else:
                print(f"⚠️ {graph_name}: 找不到Excel檔案")
                print(f"  Config 4: {config4_excel}")
                print(f"  Config 5: {config5_excel}")
        
        # 創建總結表
        self.create_summary_excel()
        
        print(f"\n🎉 批量比較完成!")
        print(f"📁 所有結果已儲存在: {self.output_dir}")
        print(f"📊 比較圖表和總結Excel請查看輸出目錄")

def main():
    """主函數"""
    print("="*80)
    print("    Config 4 (SpSA) vs Config 5 (ApSA) 批量比較工具")
    print("="*80)
    
    comparator = Config45Comparator()
    
    try:
        comparator.run_comparison()
        
    except KeyboardInterrupt:
        print("\n⚠️ 用戶中斷執行")
        
    except Exception as e:
        print(f"\n❌ 執行過程中發生錯誤: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 