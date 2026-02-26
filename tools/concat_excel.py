import pandas as pd

excel_1 = '/public/home/zncui/datasets/wheat-guo-dataset/wheat-guo-square-results-debug/normalized_result.xlsx'
excel_2 = '/public/home/zncui/datasets/wheat-guo-dataset/wheat-guo-crop-result/detection_counts.xlsx'
out = '/public/home/zncui/pycharm_project/mmdetection/tools/result.xlsx'

# 读取两个Excel文件
df1 = pd.read_excel(excel_1)
df2 = pd.read_excel(excel_2)

# 根据filename列合并两个数据框
merged_df = pd.merge(df1, df2, on='filename', how='inner')

# 将合并后的结果保存到新的Excel文件
merged_df.to_excel(out, index=False)

print(f'合并完成，结果已保存到: {out}')

