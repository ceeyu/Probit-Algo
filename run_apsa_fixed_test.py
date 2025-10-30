import subprocess
import sys

def test_single_config5():
    """測試單個Config 5實驗來驗證修復"""
    cmd = [
        "python3", "gpu_MAXCUT_var0708.py",
        "--gpu", "0",
        "--file_path", "./graph/G1.txt",
        "--param", "2",
        "--cycle", "200",  # 較短的測試
        "--trial", "3",    # 至少3次測試
        "--tau", "8",
        "--config", "5",
        "--res", "2",
        "--l_scale", "0.1",
        "--d_scale", "0.1",
        "--n_scale", "0.1",
        "--stall_threshold", "50",
        "--eta_alpha", "0.001",
        "--eta_beta", "0.002",
        "--target_drop", "80",
        "--window_size", "70"
    ]
    
    print("測試Config 5 (ApSA)...")
    print("命令:", " ".join(cmd))
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        print("\n=== 輸出 ===")
        print(result.stdout)
        
        if result.stderr:
            print("\n=== 錯誤 ===")
            print(result.stderr)
            
        return result.returncode == 0
        
    except subprocess.TimeoutExpired:
        print("測試超時")
        return False
    except Exception as e:
        print(f"測試失敗: {e}")
        return False

def compare_configs():
    """比較Config 2, 4, 5的能量表現"""
    configs = [
        (2, "pSA"),
        (4, "SpSA"), 
        (5, "ApSA")
    ]
    
    for config_num, config_name in configs:
        print(f"\n{'='*50}")
        print(f"測試 Config {config_num} ({config_name})")
        print(f"{'='*50}")
        
        cmd = [
            "python3", "gpu_MAXCUT_var0708.py",
            "--gpu", "0",
            "--file_path", "./graph/G1.txt",
            "--param", "2",
                         "--cycle", "100",
             "--trial", "3",
            "--tau", "8",
            "--config", str(config_num),
            "--res", "2",
            "--l_scale", "0.1",
            "--d_scale", "0.1",
            "--n_scale", "0.1"
        ]
        
        # Config 5特定參數
        if config_num == 5:
            cmd.extend([
                "--stall_threshold", "50",
                "--eta_alpha", "0.001",
                "--eta_beta", "0.002",
                "--target_drop", "80",
                "--window_size", "70"
            ])
        # Config 4特定參數
        elif config_num == 4:
            cmd.extend(["--stall_prop", "0.5"])
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            
            if result.returncode == 0:
                output = result.stdout
                # 提取關鍵指標
                lines = output.split('\n')
                for line in lines:
                    if any(keyword in line for keyword in [
                        'Average cut:', 'Average energy:', 'Maximum cut:', 
                        'Average reachability', 'Ising Energy:'
                    ]):
                        print(line)
            else:
                print(f"Config {config_num} 執行失敗")
                print("錯誤:", result.stderr)
                
        except subprocess.TimeoutExpired:
            print(f"Config {config_num} 執行超時")
        except Exception as e:
            print(f"Config {config_num} 執行異常: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "single":
        # 單獨測試Config 5
        success = test_single_config5()
        print(f"\n測試結果: {'成功' if success else '失敗'}")
    else:
        # 比較測試
        compare_configs() 