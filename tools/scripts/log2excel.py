import os
import pandas as pd


def parse_image_path(img_path):
    """解析图片路径提取层级信息"""
    path_parts = img_path.split('/')

    # 初始化默认值
    l1, l2 = 'N/A', 'N/A'

    try:
        # 查找zhaoxian2022的索引位置
        zhaoxian_idx = path_parts.index('zhaoxian2022')
        l1 = path_parts[zhaoxian_idx]
        l2 = path_parts[zhaoxian_idx + 1]  # 下一个元素是L2
    except (ValueError, IndexError):
        pass

    # 获取L3（文件所在目录名）
    l3 = os.path.basename(os.path.dirname(img_path))

    # 获取文件名
    file_name = os.path.basename(img_path)

    return l1, l2, l3, file_name


def process_log_to_excel(log_file_path, output_dir):
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 构造输出路径
    base_name = os.path.basename(log_file_path)
    excel_name = os.path.splitext(base_name)[0] + ".xlsx"
    excel_path = os.path.join(output_dir, excel_name)

    # 解析日志数据
    data = []
    with open(log_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line.startswith('进度: '):
                parts = line.split('|')
                if len(parts) >= 5:
                    # 提取图片路径和框数
                    img_path = parts[-1].strip()
                    box_str = parts[2].split(':')[-1].strip()
                    box_count = int(box_str)

                    # 解析路径信息
                    l1, l2, l3, file_name = parse_image_path(img_path)

                    data.append([l1, l2, l3, file_name, box_count])

    # 创建DataFrame并保存
    df = pd.DataFrame(
        data,
        columns=["L1", "L2", "L3", "file_name", "counts"]
    )

    # 保存Excel文件
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        df.to_excel(writer, index=False)

    print(f"文件已生成: {excel_path}")
    print(f"总记录数: {len(df)} 条")


# 使用示例
if __name__ == "__main__":
    # 输入路径
    log_path = "/public/home/zncui/pycharm_project/mmdetection/DJI_202305260737_006_D8H400cm-2-29-11-16.log"
    output_dir = "/public/home/zncui/pycharm_project/mmdetection/infer_dirs"

    # 执行处理
    process_log_to_excel(log_path, output_dir)