import pandas as pd

# 定义常量
W_full = 8192  # 原图宽度（像素）
H_full = 5460  # 原图高度（像素）
A_full = 11.25  # 原图实际面积（平方米）

# 读取Excel文件
# file_path = '/public/home/zncui/pycharm_project/mmdetection/zhaoxian2021/20220527/zhaoxian2021_20220527_r1-3-23-10-10.xlsx'
file_path = '/public/home/zncui/pycharm_project/mmdetection/zhaoxian2022/20230526/zhaoxian2022_20230526_r1-3-22-16-03.xlsx'
df = pd.read_excel(file_path)

# 计算实际面积
df['area'] = (df['w'] / W_full) * (df['h'] / H_full) * A_full

# 计算单位面积目标数
df['per_area'] = df['counts'] / df['area']

# 结果保留3位小数
df['area'] = df['area'].round(3)
df['per_area'] = df['per_area'].round(3)

# 定义新文件路径
# new_file_path = '/public/home/zncui/pycharm_project/mmdetection/zhaoxian2021/20220527/zhaoxian2021_20220527_r1_area-3-23-15-18.xlsx'
new_file_path = '/public/home/zncui/pycharm_project/mmdetection/zhaoxian2022/20230526/zhaoxian2022_20230526_r1_area-3-23-15-18.xlsx'
# 保存到新Excel文件
df.to_excel(new_file_path, index=False)