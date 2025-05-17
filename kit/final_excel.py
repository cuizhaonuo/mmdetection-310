import pandas as pd

# 定义文件路径
original_file_path = '/public/home/zncui/pycharm_project/mmdetection/zhaoxian2021/20220527/zhaoxian2021_20220527_r1_area-3-23-15-18.xlsx'
# original_file_path = '/public/home/zncui/pycharm_project/mmdetection/zhaoxian2022/20230526/zhaoxian2022_20230526_r1_area-3-23-15-18.xlsx'
new_file_path = '/public/home/zncui/pycharm_project/mmdetection/zhaoxian2021/20220527/zhaoxian2021_20220527_r1_sub_area-3-23-15-18.xlsx'
# new_file_path = '/public/home/zncui/pycharm_project/mmdetection/zhaoxian2022/20230526/zhaoxian2022_20230526_r1_sub_area-3-23-15-18.xlsx'

# 读取原始Excel文件
df = pd.read_excel(original_file_path)

# 定义聚合函数
# L1, L2, L3, r, h, w, area 保持不变（取第一个值）
# counts 求和
# per_area 取平均值
agg_func = {
    'L1': 'first',
    'L2': 'first',
    'L3': 'first',
    'r': 'first',
    'h': 'first',
    'w': 'first',
    'area': 'first',
    'counts': 'mean',
    'per_area': 'mean'
}

# 按L4分组并聚合
result_df = df.groupby('L4', as_index=False).agg(agg_func)

# 保留3位小数
result_df['counts'] = result_df['counts'].round(3)
# 将per_area保留3位小数
result_df['per_area'] = result_df['per_area'].round(3)

# 保存结果到新Excel文件
result_df.to_excel(new_file_path, index=False)