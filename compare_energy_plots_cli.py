#!/usr/bin/env python3
"""
命令行版本的Energy-Cycle圖表比較工具
適用於WSL環境或無GUI的伺服器環境
"""

import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse
import datetime
import glob

def parse_arguments():
    parser = argparse.ArgumentParser(description="比較兩個Excel檔案的energy-cycle圖表 (命令行版本)")
    parser.add_argument('--file1', type=str, help="第一個Excel檔案路徑")
    parser.add_argument('--file2', type=str, help="第二個Excel檔案路徑")
    parser.add_argument('--output_dir', type=str, default='./comparison_plots', help="輸出資料夾 (預設: ./comparison_plots)")
    parser.add_argument('--label1', type=str, help="第一個數據集的標籤 (預設: 自動從檔名提取)")
    parser.add_argument('--label2', type=str, help="第二個數據集的標籤 (預設: 自動從檔名提取)")
    parser.add_argument('--interactive', '-i', action='store_true', help="互動式選擇檔案")
    parser.add_argument('--list-files', '-l', action='store_true', help="列出可用的Excel檔案")
    return parser.parse_args()

def list_excel_files():
    """列出可用的Excel檔案"""
    print("=== 搜尋Excel檔案 ===")
    
    # 搜尋多個可能的路徑
    search_paths = [
        "./origin_energy_plots/*.xlsx",
        "./testvar_result*/*.xlsx", 
        "./*.xlsx",
        "./*/energy_vs_cycles*.xlsx"
    ]
    
    all_files = []
    for pattern in search_paths:
        files = glob.glob(pattern)
        all_files.extend(files)
    
    # 去重並排序
    all_files = sorted(list(set(all_files)))
    
    if not all_files:
        print("未找到Excel檔案")
        return []
    
    print(f"找到 {len(all_files)} 個Excel檔案:")
    for i, file in enumerate(all_files, 1):
        print(f"{i:2d}. {file}")
    
    return all_files

def extract_file_info(file_path):
    """從檔案名稱提取資訊"""
    filename = os.path.basename(file_path)
    name_without_ext = os.path.splitext(filename)[0]
    
    parts = name_without_ext.split('_')
    info = {}
    
    for part in parts:
        if part.startswith('config'):
            info['config'] = part
        elif part.startswith('tau'):
            info['tau'] = part
        elif part.startswith('res'):
            info['res'] = part
        elif part.startswith('G'):
            info['graph'] = part
    
    return info

def create_label_from_info(info):
    """從資訊建立標籤"""
    return f"{info.get('graph', 'G?')} {info.get('config', 'config?')} {info.get('tau', 'tau?')} {info.get('res', 'res?')}"

def interactive_file_selection():
    """互動式檔案選擇"""
    files = list_excel_files()
    if not files:
        return None, None
    
    print("\n=== 選擇第一個Excel檔案 ===")
    while True:
        try:
            choice1 = input(f"請輸入檔案編號 (1-{len(files)}): ").strip()
            idx1 = int(choice1) - 1
            if 0 <= idx1 < len(files):
                file1 = files[idx1]
                break
            else:
                print(f"請輸入 1 到 {len(files)} 之間的數字")
        except ValueError:
            print("請輸入有效的數字")
    
    print(f"已選擇: {file1}")
    
    print("\n=== 選擇第二個Excel檔案 ===")
    while True:
        try:
            choice2 = input(f"請輸入檔案編號 (1-{len(files)}): ").strip()
            idx2 = int(choice2) - 1
            if 0 <= idx2 < len(files):
                file2 = files[idx2]
                break
            else:
                print(f"請輸入 1 到 {len(files)} 之間的數字")
        except ValueError:
            print("請輸入有效的數字")
    
    print(f"已選擇: {file2}")
    
    if file1 == file2:
        print("⚠️  警告: 您選擇了相同的檔案")
    
    return file1, file2

def read_excel_data(file_path):
    """讀取Excel檔案中的Energy數據"""
    try:
        print(f"正在讀取: {file_path}")
        df = pd.read_excel(file_path, sheet_name='Energy_Data')
        
        if 'Cycle' not in df.columns or 'Energy' not in df.columns:
            raise ValueError(f"Excel檔案必須包含 'Cycle' 和 'Energy' 欄位")
        
        print(f"✅ 成功讀取 {len(df)} 筆數據")
        return df['Cycle'].values, df['Energy'].values
    
    except Exception as e:
        raise Exception(f"讀取檔案 {file_path} 時發生錯誤: {str(e)}")

def create_comparison_plot(cycles1, energy1, cycles2, energy2, label1, label2, output_path):
    """建立比較圖表"""
    print("正在建立比較圖表...")
    
    # 設定matplotlib後端為不需要顯示的版本
    plt.switch_backend('Agg')
    
    plt.figure(figsize=(12, 8))
    
    plt.plot(cycles1, energy1, label=label1, linewidth=2, alpha=0.8, color='blue')
    plt.plot(cycles2, energy2, label=label2, linewidth=2, alpha=0.8, color='red')
    
    plt.xlabel('Cycles', fontsize=12)
    plt.ylabel('Energy', fontsize=12)
    plt.title('Energy vs Cycles Comparison', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 圖表已儲存: {output_path}")

def create_combined_excel(cycles1, energy1, cycles2, energy2, label1, label2, output_path):
    """建立合併的Excel檔案"""
    print("正在建立Excel檔案...")
    
    min_length = min(len(cycles1), len(cycles2))
    
    df = pd.DataFrame({
        'Cycle': cycles1[:min_length],
        f'Energy_{label1}': energy1[:min_length],
        f'Energy_{label2}': energy2[:min_length]
    })
    
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Comparison_Data', index=False)
        
        workbook = writer.book
        worksheet = writer.sheets['Comparison_Data']
        
        try:
            from openpyxl.chart import LineChart, Reference
            chart = LineChart()
            chart.title = "Energy vs Cycles Comparison"
            chart.style = 13
            chart.x_axis.title = 'Cycles'
            chart.y_axis.title = 'Energy'
            
            cats = Reference(worksheet, min_col=1, min_row=2, max_row=min_length+1)
            
            data1 = Reference(worksheet, min_col=2, min_row=1, max_row=min_length+1)
            chart.add_data(data1, titles_from_data=True)
            
            data2 = Reference(worksheet, min_col=3, min_row=1, max_row=min_length+1)
            chart.add_data(data2, titles_from_data=True)
            
            chart.set_categories(cats)
            worksheet.add_chart(chart, "E2")
            
            print("✅ Excel圖表已建立")
        except Exception as e:
            print(f"⚠️  Excel圖表建立失敗: {e}")
    
    print(f"✅ Excel檔案已儲存: {output_path}")

def main():
    args = parse_arguments()
    
    print("="*60)
    print("    Energy vs Cycles 比較工具 (命令行版本)")
    print("="*60)
    
    # 如果要求列出檔案
    if args.list_files:
        list_excel_files()
        return
    
    # 確定檔案路徑
    if args.interactive or (not args.file1 or not args.file2):
        print("\n進入互動模式...")
        file1, file2 = interactive_file_selection()
        if not file1 or not file2:
            print("❌ 未選擇檔案，程式結束")
            return
    else:
        file1 = args.file1
        file2 = args.file2
    
    # 檢查檔案是否存在
    for f in [file1, file2]:
        if not os.path.exists(f):
            print(f"❌ 檔案不存在: {f}")
            return
    
    try:
        # 讀取數據
        print("\n" + "="*40)
        cycles1, energy1 = read_excel_data(file1)
        cycles2, energy2 = read_excel_data(file2)
        
        # 建立標籤
        if args.label1:
            label1 = args.label1
        else:
            info1 = extract_file_info(file1)
            label1 = create_label_from_info(info1) if info1 else os.path.basename(file1)
        
        if args.label2:
            label2 = args.label2
        else:
            info2 = extract_file_info(file2)
            label2 = create_label_from_info(info2) if info2 else os.path.basename(file2)
        
        print(f"\n標籤1: {label1}")
        print(f"標籤2: {label2}")
        
        # 建立輸出目錄
        output_dir = args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 建立檔案名稱
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        png_filename = f"energy_comparison_{timestamp}.png"
        excel_filename = f"energy_comparison_{timestamp}.xlsx"
        
        png_path = os.path.join(output_dir, png_filename)
        excel_path = os.path.join(output_dir, excel_filename)
        
        print("\n" + "="*40)
        # 建立圖表和Excel
        create_comparison_plot(cycles1, energy1, cycles2, energy2, label1, label2, png_path)
        create_combined_excel(cycles1, energy1, cycles2, energy2, label1, label2, excel_path)
        
        print("\n" + "="*60)
        print("🎉 比較完成！")
        print(f"📁 輸出資料夾: {output_dir}")
        print(f"🖼️  PNG圖表: {png_filename}")
        print(f"📊 Excel檔案: {excel_filename}")
        print("="*60)
        
        # 顯示統計資訊
        print(f"\n📈 數據統計:")
        print(f"數據集1 ({label1}): {len(cycles1)} 個週期")
        print(f"數據集2 ({label2}): {len(cycles2)} 個週期")
        print(f"能量範圍1: {min(energy1):.2f} ~ {max(energy1):.2f}")
        print(f"能量範圍2: {min(energy2):.2f} ~ {max(energy2):.2f}")
        
    except Exception as e:
        print(f"❌ 錯誤: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 