#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import math
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
from PIL import Image

# ============== 三个 CSV 路径（可用命令行覆盖） ==============
MMDET_CSV_PATH = "/public/home/zncui/pycharm_project/mmdetection/tools/myscripts/wheat_work/mmdet_all_result.csv"
MMDET_MASK_CSV_PATH = "/public/home/zncui/pycharm_project/mmdetection/tools/myscripts/mask_result.csv"
MMROT_CSV_PATH = "/public/home/zncui/pycharm_project/mmrotate/tools/myscripts/wheat_work/result.csv"

# 输出目录
OUT_DIR = "/public/home/zncui/pycharm_project/mmdetection/tools/myscripts/wheat_work/vis"

# 两个数据来源前缀
SRC_A_PREFIX = "/public/home/zncui/datasets/zhaoxian/wheats_rle_coco/coco/test2017/"
SRC_B_PREFIX = "/public/home/zncui/datasets/wheat-guo-dataset/yolo_wheat_big_image/images/test/"

# 两个来源的短代号（用于文件名）
SRC_A_SLUG = "coco_test2017"
SRC_B_SLUG = "yolo_big_image_test"

# 三个 CSV 的短代号（列顺序：mmdet, mmdet_mask, mmrot）
CSV_SLUGS = ["mmdet", "mmdet_mask", "mmrot"]
CSV_PATHS = [MMDET_CSV_PATH, MMDET_MASK_CSV_PATH, MMROT_CSV_PATH]

# 三种图的后缀
BAR_SUFFIX = "diff_bar.png"
LINE_SUFFIX = "diff_line.png"
SCATTER_SUFFIX = "ins_vs_pred_scatter.png"


# ---------------- 参数 ----------------
def parse_args():
    ap = argparse.ArgumentParser(description="Per CSV & per source plotting, accuracy, and 2x3 stitching")
    ap.add_argument("--mmdet", default=MMDET_CSV_PATH, help="MMDetection CSV path")
    ap.add_argument("--mmdet-mask", default=MMDET_MASK_CSV_PATH, help="MMDetection (mask) CSV path")
    ap.add_argument("--mmrot", default=MMROT_CSV_PATH, help="MMRotate CSV path")
    ap.add_argument("--out-dir", default=OUT_DIR, help="Directory to save figures")
    ap.add_argument("--tick-step", type=float, default=5.0, help="主刻度步长(间隔)")
    ap.add_argument("--tick-fontsize", type=int, default=10, help="刻度文字大小")
    return ap.parse_args()


# ---------------- 工具函数 ----------------
def _tune_axis(ax, which: str, max_val: float, step: float, use_minor: bool = True):
    """设置轴刻度为固定步长并将上限对齐到步长倍数。which in {'x','y'}"""
    if step <= 0:
        step = 1.0
    if max_val <= 0:
        max_val = step
    top = step * math.ceil(max_val / step)
    if which == "y":
        ax.yaxis.set_major_locator(MultipleLocator(step))
        if use_minor:
            ax.yaxis.set_minor_locator(AutoMinorLocator(5))
        ax.set_ylim(0, top)
    else:
        ax.xaxis.set_major_locator(MultipleLocator(step))
        if use_minor:
            ax.xaxis.set_minor_locator(AutoMinorLocator(5))
        ax.set_xlim(0, top)


def plot_bar(df_plot, out_png, tick_step: float, tick_fontsize: int):
    n = len(df_plot)
    fig_w = max(6, 0.25 * n)
    fig_h = 6
    plt.figure(figsize=(fig_w, fig_h))
    plt.bar(df_plot["shortname"], df_plot["abs_diff"])
    plt.xticks(rotation=75, ha="right", fontsize=tick_fontsize)
    plt.yticks(fontsize=tick_fontsize)
    plt.ylabel("Absolute Difference |ins_counts - pred_counts|")
    plt.title("Per-file Absolute Difference (Bar)")
    ax = plt.gca()
    _tune_axis(ax, "y", float(df_plot["abs_diff"].max()), step=tick_step, use_minor=True)
    ax.grid(axis="y", which="major", linestyle="--", alpha=0.5)
    ax.grid(axis="y", which="minor", linestyle=":", alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def plot_line(df_plot, out_png, tick_step: float, tick_fontsize: int):
    n = len(df_plot)
    fig_w = max(6, 0.25 * n)
    fig_h = 5
    plt.figure(figsize=(fig_w, fig_h))
    plt.plot(range(n), df_plot["abs_diff"], marker="o", linewidth=1)
    step_idx = max(1, n // 20)  # 控制最多约 20 个刻度标签
    xticks = list(range(0, n, step_idx))
    plt.xticks(xticks, df_plot["shortname"].iloc[xticks], rotation=75, ha="right", fontsize=tick_fontsize)
    plt.yticks(fontsize=tick_fontsize)
    plt.ylabel("Absolute Difference |ins_counts - pred_counts|")
    plt.title("Per-file Absolute Difference (Line)")
    ax = plt.gca()
    _tune_axis(ax, "y", float(df_plot["abs_diff"].max()), step=tick_step, use_minor=True)
    ax.grid(axis="y", which="major", linestyle="--", alpha=0.5)
    ax.grid(axis="y", which="minor", linestyle=":", alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def plot_scatter_ins_vs_pred(df, out_png, tick_step: float, tick_fontsize: int):
    x = df["ins_counts"].values
    y = df["pred_counts"].values
    if len(x) == 0:
        print(f"[WARN] 空数据，跳过散点图: {out_png}")
        return
    max_val = max(float(x.max()), float(y.max()))
    plt.figure(figsize=(6.6, 6.2))
    plt.scatter(x, y, s=24, alpha=0.9)
    # 对角线 y=x
    top = tick_step * math.ceil((max_val if max_val > 0 else tick_step) / tick_step)
    plt.plot([0, top], [0, top], linestyle="--", linewidth=1.2)
    plt.xlabel("ins_counts")
    plt.ylabel("pred_counts")
    plt.title("Scatter: ins_counts vs pred_counts (with y=x)")
    ax = plt.gca()
    _tune_axis(ax, "x", max_val, step=tick_step, use_minor=True)
    _tune_axis(ax, "y", max_val, step=tick_step, use_minor=True)
    ax.tick_params(axis="both", which="both", labelsize=tick_fontsize)
    ax.grid(which="major", linestyle="--", alpha=0.5)
    ax.grid(which="minor", linestyle=":", alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    required = {"filename", "ins_counts", "pred_counts"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV 缺少列: {missing}")

    df = df.copy()
    df["abs_diff"] = (df["ins_counts"] - df["pred_counts"]).abs()
    df["shortname"] = df["filename"].apply(lambda p: os.path.basename(str(p)))
    return df


def compute_error_percent(df: pd.DataFrame) -> float:
    """整体聚合：sum(|ins - pred|) / sum(ins) * 100。忽略 ins_counts == 0 的样本。"""
    if len(df) == 0:
        return float("nan")
    df = df[df["ins_counts"] > 0]
    if len(df) == 0:
        return float("nan")
    num = (df["ins_counts"] - df["pred_counts"]).abs().sum()
    den = df["ins_counts"].sum()
    return float(num) / float(den) * 100.0


def process_one_csv(csv_path: str, slug: str, out_dir: str, tick_step: float, tick_fontsize: int):
    """读取一个 CSV，按来源各出三图，并返回该 CSV 对应的三图路径字典。"""
    paths = {
        "bar": {SRC_A_SLUG: None, SRC_B_SLUG: None},
        "line": {SRC_A_SLUG: None, SRC_B_SLUG: None},
        "scatter": {SRC_A_SLUG: None, SRC_B_SLUG: None},
    }

    if not os.path.exists(csv_path):
        print(f"[WARN] CSV 不存在，跳过: {csv_path}")
        return paths

    df = pd.read_csv(csv_path)
    df = prepare_df(df)
    df_a = df[df["filename"].astype(str).str.startswith(SRC_A_PREFIX)].copy()
    df_b = df[df["filename"].astype(str).str.startswith(SRC_B_PREFIX)].copy()

    # 排序仅用于 bar/line
    df_a_plot = df_a.sort_values("abs_diff", ascending=False).reset_index(drop=True)
    df_b_plot = df_b.sort_values("abs_diff", ascending=False).reset_index(drop=True)

    # 输出路径
    os.makedirs(out_dir, exist_ok=True)

    # 来源 A
    if len(df_a_plot) > 0:
        bar_a = os.path.join(out_dir, f"{slug}_{SRC_A_SLUG}_{BAR_SUFFIX}")
        line_a = os.path.join(out_dir, f"{slug}_{SRC_A_SLUG}_{LINE_SUFFIX}")
        scatter_a = os.path.join(out_dir, f"{slug}_{SRC_A_SLUG}_{SCATTER_SUFFIX}")
        plot_bar(df_a_plot, bar_a, tick_step, tick_fontsize)
        plot_line(df_a_plot, line_a, tick_step, tick_fontsize)
        plot_scatter_ins_vs_pred(df_a, scatter_a, tick_step, tick_fontsize)
        paths["bar"][SRC_A_SLUG] = bar_a
        paths["line"][SRC_A_SLUG] = line_a
        paths["scatter"][SRC_A_SLUG] = scatter_a
    else:
        print(f"[WARN] {slug}: 未匹配到来源A数据 ({SRC_A_PREFIX})")

    # 来源 B
    if len(df_b_plot) > 0:
        bar_b = os.path.join(out_dir, f"{slug}_{SRC_B_SLUG}_{BAR_SUFFIX}")
        line_b = os.path.join(out_dir, f"{slug}_{SRC_B_SLUG}_{LINE_SUFFIX}")
        scatter_b = os.path.join(out_dir, f"{slug}_{SRC_B_SLUG}_{SCATTER_SUFFIX}")
        plot_bar(df_b_plot, bar_b, tick_step, tick_fontsize)
        plot_line(df_b_plot, line_b, tick_step, tick_fontsize)
        plot_scatter_ins_vs_pred(df_b, scatter_b, tick_step, tick_fontsize)
        paths["bar"][SRC_B_SLUG] = bar_b
        paths["line"][SRC_B_SLUG] = line_b
        paths["scatter"][SRC_B_SLUG] = scatter_b
    else:
        print(f"[WARN] {slug}: 未匹配到来源B数据 ({SRC_B_PREFIX})")

    # 准确率：按来源聚合
    acc_a = compute_error_percent(df_a)
    acc_b = compute_error_percent(df_b)
    print(f"[ACC] {slug} - {SRC_A_SLUG}: {acc_a:.4f}%  (定义: sum|ins-pred|/sum ins * 100)")
    print(f"[ACC] {slug} - {SRC_B_SLUG}: {acc_b:.4f}%  (定义: sum|ins-pred|/sum ins * 100)")

    return paths


# ---------------- 拼接 2×3 ----------------
def _safe_open(path, fallback_size=(800, 500)):
    """打开图像；若不存在返回指定大小的白底图。"""
    if path and os.path.exists(path):
        return Image.open(path).convert("RGB")
    # 生成空白
    return Image.new("RGB", fallback_size, (255, 255, 255))


def stitch_2x3_grid(per_plot_paths, out_path):
    """
    per_plot_paths: list of list with shape [2 rows][3 cols] of image paths
                    rows: [SRC_A, SRC_B]; cols: [mmdet, mmdet_mask, mmrot]
    """
    # 先找出所有存在的图，确定 cell 尺寸（取最大 w,h）
    images = []
    for row in per_plot_paths:
        for p in row:
            if p and os.path.exists(p):
                try:
                    im = Image.open(p)
                    images.append(im)
                except Exception:
                    pass
    if images:
        cell_w = max(im.width for im in images)
        cell_h = max(im.height for im in images)
    else:
        cell_w, cell_h = 800, 500

    # 统一 resize 为 cell 大小
    grid = []
    for row in per_plot_paths:
        grid_row = []
        for p in row:
            im = _safe_open(p, (cell_w, cell_h))
            if im.size != (cell_w, cell_h):
                im = im.resize((cell_w, cell_h), Image.LANCZOS)
            grid_row.append(im)
        grid.append(grid_row)

    # 拼接
    cols = 3
    rows = 2
    canvas = Image.new("RGB", (cols * cell_w, rows * cell_h), (255, 255, 255))
    for r in range(rows):
        for c in range(cols):
            canvas.paste(grid[r][c], (c * cell_w, r * cell_h))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    canvas.save(out_path, "PNG")
    print(f"[OK] 拼接完成: {out_path}")


# ---------------- 主流程 ----------------
def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # 逐个 CSV 处理 & 记录各自图的路径
    csv_inputs = [
        (args.mmdet, "mmdet"),
        (args.mmdet_mask, "mmdet_mask"),
        (args.mmrot, "mmrot"),
    ]

    # 收集三类图在 2×3 网格中的路径
    # 行顺序：SRC_A_SLUG, SRC_B_SLUG；列顺序：mmdet, mmdet_mask, mmrot
    bar_grid = [[None, None, None], [None, None, None]]
    line_grid = [[None, None, None], [None, None, None]]
    scatter_grid = [[None, None, None], [None, None, None]]

    for col, (csv_path, slug) in enumerate(csv_inputs):
        paths_dict = process_one_csv(csv_path, slug, args.out_dir, args.tick_step, args.tick_fontsize)
        # 第一行：来源 A；第二行：来源 B
        bar_grid[0][col] = paths_dict["bar"][SRC_A_SLUG]
        bar_grid[1][col] = paths_dict["bar"][SRC_B_SLUG]
        line_grid[0][col] = paths_dict["line"][SRC_A_SLUG]
        line_grid[1][col] = paths_dict["line"][SRC_B_SLUG]
        scatter_grid[0][col] = paths_dict["scatter"][SRC_A_SLUG]
        scatter_grid[1][col] = paths_dict["scatter"][SRC_B_SLUG]

    # 拼接为 2×3 总图
    stitch_2x3_grid(bar_grid, os.path.join(args.out_dir, "stitched_bar_2x3.png"))
    stitch_2x3_grid(line_grid, os.path.join(args.out_dir, "stitched_line_2x3.png"))
    stitch_2x3_grid(scatter_grid, os.path.join(args.out_dir, "stitched_scatter_2x3.png"))

    print(f"[DONE] 所有图已保存在：{args.out_dir}")


if __name__ == "__main__":
    main()
