import pandas as pd
import os
from tqdm import tqdm  # 进度条库

# 配置路径参数
directory = '/public/home/zncui/pycharm_project/mmdetection/zhaoxian2022/20230526/'
output_filename = 'zhaoxian2022_20230526.xlsx'
output_path = os.path.join(directory, output_filename)
expected_columns = ['L1', 'L2', 'L3', 'L4', 'file_name', 'counts']

# 递归获取所有子目录中的xlsx文件
all_files = []
for root, dirs, files in os.walk(directory):
    for file in files:
        if file.endswith('.xlsx') and file != output_filename:
            all_files.append(os.path.join(root, file))

# 合并数据逻辑
if not all_files:
    print("未找到任何Excel文件")
else:
    dfs = []
    # 添加带进度条的遍历
    for file in tqdm(all_files, desc="合并进度", unit="file", colour='green'):
        try:
            df = pd.read_excel(file)
            if list(df.columns) == expected_columns:
                dfs.append(df)
            else:
                print(f"\n跳过 {os.path.relpath(file, directory)}: 列名不匹配")
        except Exception as e:
            print(f"\n读取 {file} 失败: {str(e)}")

    if dfs:
        # 合并时添加二级进度条（可选）
        with tqdm(total=1, desc="最终合并", leave=False) as pbar:
            combined_df = pd.concat(dfs, ignore_index=True)
            combined_df.to_excel(output_path, index=False)
            pbar.update(1)

        print(f"\n成功合并 {len(dfs)} 个文件到:\n{output_path}")
    else:
        print("\n没有符合要求的文件可合并")