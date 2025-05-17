import pandas as pd
import os

# 定义文件路径
base_dir = "/public/home/zncui/pycharm_project/mmdetection/zhaoxian2022/20230526/"
input_file = os.path.join(base_dir, "zhaoxian2022_20230526_r1-3-22-16-03.xlsx")
output_file = os.path.join(base_dir, "zhaoxian2022_20230526_xiaoqu-3-22-16-21.xlsx")

# 读取 Excel 文件
df = pd.read_excel(input_file)

# 根据 L4 列进行分组，合并时 file_name 列去除，counts 列求和，其他列取第一条记录
grouped_df = df.groupby("L4", as_index=False).agg({
    "L1": "first",
    "L2": "first",
    "L3": "first",
    "counts": "sum",
    "h": "first",
    "w": "first",
    "r": "first"
})

# 保存合并后的结果到新的 Excel 文件
grouped_df.to_excel(output_file, index=False)
print("合并完成，结果保存在：", output_file)
