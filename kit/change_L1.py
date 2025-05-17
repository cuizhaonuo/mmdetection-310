import os
import pandas as pd
from tqdm import tqdm  # 需要安装tqdm库


def replace_excel_l1_content(root_dir):
    """
    遍历指定目录及其子目录下的所有xlsx文件，将L1列中的'zhaoxian2022'替换为'zhaoxian2021'

    参数:
        root_dir (str): 要遍历的根目录路径
    """
    # 预扫描获取所有xlsx文件路径
    file_list = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('.xlsx'):
                file_path = os.path.join(dirpath, filename)
                file_list.append(file_path)

    # 初始化进度条
    with tqdm(total=len(file_list), desc="处理进度", unit="file") as pbar:
        for file_path in file_list:
            try:
                # 读取Excel文件
                df = pd.read_excel(file_path)

                # 检查是否存在L1列
                if 'L1' in df.columns:
                    # 替换L1列中的内容（完全匹配）
                    df['L1'] = df['L1'].apply(
                        lambda x: 'zhaoxian2021' if x == 'zhaoxian2022' else x
                    )

                    # 保存修改后的文件，覆盖原文件
                    df.to_excel(file_path, index=False, engine='openpyxl')
                    pbar.set_postfix_str(f"完成：{os.path.basename(file_path)}", refresh=False)
                else:
                    pbar.set_postfix_str(f"跳过：{os.path.basename(file_path)}（无L1列）", refresh=False)

                pbar.update(1)
            except Exception as e:
                pbar.write(f"错误：{file_path} - {str(e)}")  # 使用pbar.write避免进度条错乱
                pbar.update(1)


if __name__ == "__main__":
    target_dir = "/public/home/zncui/pycharm_project/mmdetection/zhaoxian2021/20220527/"
    replace_excel_l1_content(target_dir)