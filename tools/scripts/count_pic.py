import os
from pathlib import Path


def count_image_files(root_dir, extensions=None):
    """
    递归统计图片文件数量和类型分布

    参数:
        root_dir (str): 要扫描的根目录
        extensions (set): 要统计的图片扩展名集合 (默认常见图片格式)

    返回:
        tuple: (总数量, 扩展名字典统计, 所有文件路径列表)
    """
    # 设置默认支持的图片格式
    default_ext = {'.jpg', '.jpeg', '.png', '.gif',
                   '.bmp', '.tiff', '.webp', '.heic'}
    ext_filter = extensions or default_ext

    total = 0
    ext_counter = {ext: 0 for ext in ext_filter}
    file_paths = []

    # 递归遍历文件系统
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            file_ext = Path(filename).suffix.lower()
            if file_ext in ext_filter:
                total += 1
                ext_counter[file_ext] += 1
                file_paths.append(str(Path(dirpath) / filename))

    return total, ext_counter, file_paths


if __name__ == "__main__":
    # 配置参数
    TARGET_DIR = "/public/lab_share/nfs_field/level_1/DJI/P1/zhaoxian2022"
    CUSTOM_EXT = None  # 使用自定义格式时示例: {'.jpg', '.raw'}

    try:
        # 执行统计
        total_count, ext_stats, all_files = count_image_files(TARGET_DIR, CUSTOM_EXT)

        # 输出统计报告
        print(f"\n{' 扫描结果 ':=^50}")
        print(f"• 目标目录: {TARGET_DIR}")
        print(f"• 总图片数: \033[1;32m{total_count}\033[0m")
        print("\n文件类型分布:")
        for ext, count in ext_stats.items():
            print(f"  {ext.upper():<7}: {count}")

        # 显示示例文件
        print("\n示例文件路径:")
        for f in all_files[:3]:
            print(f"  {Path(f).parent}/.../{Path(f).name}")

    except FileNotFoundError:
        print(f"错误: 目录不存在 - {TARGET_DIR}")
    except PermissionError:
        print(f"错误: 无权限访问目录 - {TARGET_DIR}")
    except Exception as e:
        print(f"意外错误: {str(e)}")