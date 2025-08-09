import os
from os import makedirs
import json
import re
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from tools.analysis_tools.analyze_logs import load_json_logs, plot_curve



class Logger:
    def __init__(self, json_log, key, legend, out):
        self.backend = None
        self.eval_interval = '1'
        # '/public/home/zncui/pycharm_project/mmdetection/work_dirs/co_dino_5scale_swin_l_16xb1_16e_o365tococo/20250221_132517/vis_data/20250221_132517.json'
        self.json_logs = [json_log]
        self.keys = [key]
        self.legend = [legend]
        self.out = out
        self.start_epoch = '1'
        self.style = 'dark'
        self.task = 'plot_curve'
        self.title = None

class LogPlotter:
    def __init__(self, work_dirs):
        self.work_dirs = work_dirs
        self.keys = ['loss', 'loss_cls', 'loss_bbox', 'loss_iou']
        self.legend = ['loss', 'loss_cls', 'loss_bbox', 'loss_iou']
        # out directory for saving plots
        self.out_dir = os.path.join(self.work_dirs, 'plot_logs')
        makedirs(self.out_dir, exist_ok=True)
        self.init_json_list(work_dirs)

    def init_json_list(self, work_dir):
        self.json_logs = []
        self.json_scalars = []
        pattern = re.compile('\d+_\d+\.json$')
        for root, dirs, files in os.walk(work_dir):
            for file in files:
                if file.endswith('.json'):
                    json_path = os.path.join(root, file)
                    if pattern.search(json_path):
                        self.json_logs.append(json_path)
                    else:
                        self.json_scalars.append(json_path)
        self.json_list = self.json_logs + self.json_scalars
        print('json_logs:\n', self.json_logs)
        print('json_scalars:\n', self.json_scalars)



    def plot_logs(self):
        for key in self.keys:
            for json_path in self.json_list:
                log_dicts = load_json_logs([json_path])
                try:
                    out = None
                    if json_path in self.json_logs:
                        file_name = os.path.splitext(os.path.basename(json_path))[0]
                        out_logs = os.path.join(self.out_dir, 'logs')
                        makedirs(out_logs, exist_ok=True)
                        out = os.path.join(out_logs, f'{file_name}_{key}.png')
                    elif json_path in self.json_scalars:
                        pattern = re.compile('/(\d+_\d+)/')
                        m = pattern.search(json_path)
                        out_scalars = os.path.join(self.out_dir, 'scalars')
                        makedirs(out_scalars, exist_ok=True)
                        out = os.path.join(out_scalars, f'{m.group(1)}_{key}.png')
                    assert out is not None, 'json_log should be in json_logs or json_scalars'
                    logger = Logger(json_log=json_path, key=key, legend=key, out=out)
                    plot_curve(log_dicts, logger)
                except Exception as e:
                    print(f'Error plotting {key} for {json_path}: {e}')
                    continue


def mosaic_images(input_dir: str, out_dir: str, out_filename: str = 'mosaic.png'):
    """
    将 input_dir 下所有按 <timestamp>_<metric>.png 命名的图片，
    按 metric 列、timestamp 行拼成一个大图，并保存到 out_dir/out_filename.

    Args:
        input_dir (str): 存放各 timestamp 和 metric 图的目录。
        out_dir (str): 拼接图的输出目录。
        out_filename (str): 输出文件名，默认为 mosaic.png。
    """
    # 1) 列出所有 png 文件
    files = [f for f in os.listdir(input_dir) if f.endswith('.png')]
    # 2) 用正则提取 timestamp 和 metric
    pattern = re.compile(r'(?P<ts>\d+_\d+)_(?P<metric>.+)\.png$')
    timestamps = sorted({ pattern.match(f).group('ts')
                          for f in files if pattern.match(f) })
    metrics    = sorted({ pattern.match(f).group('metric')
                          for f in files if pattern.match(f) })

    if not timestamps or not metrics:
        raise ValueError(f'在 {input_dir} 下没有匹配 "<ts>_<metric>.png" 的文件。')

    # 3) 创建画布
    n_rows, n_cols = len(timestamps), len(metrics)
    fig, axes = plt.subplots(n_rows,
                             n_cols,
                             figsize=(n_cols * 4, n_rows * 3),
                             squeeze=False)

    # 4) 逐行逐列加载并绘制
    for i, ts in enumerate(timestamps):
        for j, metric in enumerate(metrics):
            fname = f'{ts}_{metric}.png'
            path = os.path.join(input_dir, fname)
            ax = axes[i][j]
            if os.path.exists(path):
                img = mpimg.imread(path)
                ax.imshow(img)
            # 不显示坐标轴
            ax.axis('off')

            # 第一行加上列标题
            if i == 0:
                ax.set_title(metric, fontsize=12)
            # 第一列加上行标签
            if j == 0:
                ax.set_ylabel(ts, fontsize=12)

    plt.tight_layout()

    # 5) 保存结果
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, out_filename)
    fig.savefig(out_path)
    plt.close(fig)
    print(f'拼图已保存到 {out_path}')

def plot_work_dirs_logs(model_output_dir):
    work_dirs = '/public/home/zncui/pycharm_project/mmdetection/work_dirs'
    target_dir = os.path.join(work_dirs, model_output_dir)

    logger = LogPlotter(target_dir)
    logger.plot_logs()
    for key in ['logs', 'scalars']:
        input_dir = os.path.join(target_dir, 'plot_logs', key)
        output_dir = os.path.join(target_dir, 'plot_logs')
        # 输出文件名可以自定义，比如 'combined_loss.png' 或 'all_metrics.png'
        mosaic_images(input_dir, output_dir, out_filename=f'all_metrics_grid_{key}.png')

# plot_work_dirs_logs(model_output_dir='co_dino_5scale_swin_l_16xb1_16e_o365tococo')
plot_work_dirs_logs(model_output_dir='co_dino_guo_yolo2coco_test2')

