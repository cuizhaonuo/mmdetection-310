import os
import pandas as pd


def merge_excel_files(file_paths, output_path):
    """合并多个Excel文件到一个文件"""
    # 读取所有数据
    dfs = []
    for f in file_paths:
        df = pd.read_excel(f)
        dfs.append(df)
        print(f"已读取: {os.path.basename(f)} ({len(df)} 条记录)")

    # 合并数据
    merged_df = pd.concat(dfs, ignore_index=True)

    # 保存结果
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    merged_df.to_excel(output_path, index=False, engine='openpyxl')

    print(f"\n合并完成！总记录数: {len(merged_df)}")
    print(f"文件已保存至: {output_path}")


if __name__ == "__main__":
    # 输入文件列表
    input_files = [
        "/public/home/zncui/pycharm_project/mmdetection/infer_dirs/DJI_202305260737_001_D6H400cm-2-27-16-37.xlsx",
        "/public/home/zncui/pycharm_project/mmdetection/infer_dirs/DJI_202305260737_002_D6H400cm-2-28-9-31.xlsx",
        "/public/home/zncui/pycharm_project/mmdetection/infer_dirs/DJI_202305260737_003_D7NH400cm-2-28-10-56.xlsx",
        "/public/home/zncui/pycharm_project/mmdetection/infer_dirs/DJI_202305260737_004_D7NH400cm-2-28-23-28.xlsx",
        "/public/home/zncui/pycharm_project/mmdetection/infer_dirs/DJI_202305260737_005_D8H400cm-2-29-11-12.xlsx",
        "/public/home/zncui/pycharm_project/mmdetection/infer_dirs/DJI_202305260737_006_D8H400cm-2-29-11-16.xlsx"
    ]

    # 输出文件路径
    output_file = "/public/home/zncui/pycharm_project/mmdetection/infer_dirs/wheat_counts-3-2-10-42.xlsx"

    # 执行合并
    merge_excel_files(input_files, output_file)