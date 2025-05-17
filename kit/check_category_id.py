import json

# 文件路径
file_path = '/public/home/zncui/datasets/wheats_rle_coco/annotations/instances_val2017.json'

try:
    # 读取并解析JSON文件
    with open(file_path, 'r') as f:
        data = json.load(f)
except FileNotFoundError:
    print(f"错误：文件 {file_path} 不存在。")
    exit(1)
except json.JSONDecodeError:
    print("错误：文件格式不正确，无法解析为JSON。")
    exit(1)
except Exception as e:
    print(f"加载文件时发生未知错误：{e}")
    exit(1)

# 获取annotations列表
annotations = data.get('annotations')
if annotations is None:
    print("错误：文件中缺少 'annotations' 字段。")
    exit(1)

# 检查所有category_id是否为1
invalid_entries = []
for ann in annotations:
    ann_id = ann.get('id', '未知ID')
    # 检查是否存在category_id字段
    if 'category_id' not in ann:
        invalid_entries.append((ann_id, "缺少'category_id'字段"))
        continue
    # 检查category_id类型是否为整数
    cat_id = ann['category_id']
    if not isinstance(cat_id, int):
        invalid_entries.append((ann_id, f"类型错误（当前类型：{type(cat_id)}）"))
        continue
    # 检查category_id值是否为1
    if cat_id != 1:
        invalid_entries.append((ann_id, f"值错误（当前值：{cat_id}）"))

# 输出结果
total_annotations = len(annotations)
invalid_count = len(invalid_entries)
if invalid_count == 0:
    print("所有annotation的category_id均为1。")
else:
    print(f"发现 {invalid_count}/{total_annotations} 个annotation存在问题：")
    # 打印前10个错误示例
    for ann_id, error_msg in invalid_entries[:10]:
        print(f"  - 标注ID {ann_id}: {error_msg}")
    if invalid_count > 10:
        print(f"  （仅显示前10个错误，共{invalid_count}个错误）")

# 退出码：存在错误时返回1
exit(1 if invalid_count > 0 else 0)