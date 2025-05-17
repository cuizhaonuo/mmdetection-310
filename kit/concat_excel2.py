import os
import pandas as pd
from tqdm import tqdm  # 导入进度条库

# 基础目录
# base_dir = "/public/home/zncui/pycharm_project/mmdetection/zhaoxian2022/20230526/"
base_dir = "/public/home/zncui/pycharm_project/mmdetection/zhaoxian2021/20220527/"

# 要遍历的顶级子目录
# target_dirs = ["D7H400cm", "D6H400cm"]
target_dirs = ["706D6point4m", "706D7point4m", "706D8point4m", "706D9point4m", "D9s"]

# 先收集所有 results2 目录下的 Excel 文件路径
excel_files = []

for top_dir in target_dirs:
    top_path = os.path.join(base_dir, top_dir)
    if os.path.isdir(top_path):
        for sub_dir in os.listdir(top_path):
            sub_dir_path = os.path.join(top_path, sub_dir)
            if os.path.isdir(sub_dir_path):
                results2_path = os.path.join(sub_dir_path, "results2")
                if os.path.isdir(results2_path):
                    for file_name in os.listdir(results2_path):
                        if file_name.endswith(".xlsx") or file_name.endswith(".xls"):
                            file_path = os.path.join(results2_path, file_name)
                            excel_files.append(file_path)

# 用于存放所有读取到的 DataFrame
dfs = []

# 利用 tqdm 显示读取进度
for file_path in tqdm(excel_files, desc="Processing Excel files"):
    try:
        df = pd.read_excel(file_path)
        dfs.append(df)
    except Exception as e:
        print(f"读取 {file_path} 失败: {e}")

# 合并所有 DataFrame 并保存
if dfs:
    merged_df = pd.concat(dfs, ignore_index=True)
    output_file = os.path.join(base_dir, "zhaoxian2021_20220527_r1-3-23-10-10.xlsx")
    merged_df.to_excel(output_file, index=False)
    print(f"合并完成，结果保存在：{output_file}")
else:
    print("未找到任何 Excel 文件。")
