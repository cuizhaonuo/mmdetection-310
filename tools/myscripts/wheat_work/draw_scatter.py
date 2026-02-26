#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
读取 CSV（列: filename, ins_counts, pred_counts），
计算 sum(|ins - pred|) / sum(ins)，并绘制散点图：
- x 轴: ins_counts
- y 轴: pred_counts
- 加粗加深对角线 y = x
使用示例：
python plot_ins_pred_scatter.py \
  --input /public/home/zncui/pycharm_project/mmdetection/tools/myscripts/wheat_work/mmdet_all_result.csv \
  --output /public/home/zncui/pycharm_project/mmdetection/tools/myscripts/wheat_work/mmdet_all_result_scatter.png
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_csv(csv_path: str) -> pd.DataFrame:
    # 尝试常见编码
    for enc in ("utf-8", "utf-8-sig", "gbk"):
        try:
            df = pd.read_csv(csv_path, encoding=enc)
            break
        except Exception:
            df = None
    if df is None:
        raise RuntimeError(f"无法读取文件：{csv_path}")

    # 统一列名小写，去掉两侧空格
    df.columns = [str(c).strip().lower() for c in df.columns]

    required = {"filename", "ins_counts", "pred_counts"}
    if not required.issubset(set(df.columns)):
        raise ValueError(f"CSV 中必须包含列: {required}；当前列为: {list(df.columns)}")

    # 仅保留需要的列，且清理无效值
    df = df[["filename", "ins_counts", "pred_counts"]].copy()
    # 转数值（遇到非数值强制为 NaN）
    df["ins_counts"] = pd.to_numeric(df["ins_counts"], errors="coerce")
    df["pred_counts"] = pd.to_numeric(df["pred_counts"], errors="coerce")
    # 去掉 NaN
    df = df.dropna(subset=["ins_counts", "pred_counts"])

    # 只保留非负值，且 ins_counts 与 pred_counts 取整
    df = df[(df["ins_counts"] >= 0) & (df["pred_counts"] >= 0)].copy()
    df["ins_counts"] = df["ins_counts"].astype(int)
    df["pred_counts"] = df["pred_counts"].astype(int)

    return df


def compute_error_ratio(df: pd.DataFrame) -> tuple[float, int, int]:
    """
    返回: (错误率, sum_abs_err, sum_ins)
    错误率 = sum(|ins - pred|) / sum(ins)
    """
    abs_err = (df["ins_counts"] - df["pred_counts"]).abs()
    sum_abs_err = int(abs_err.sum())
    sum_ins = int(df["ins_counts"].sum())
    if sum_ins == 0:
        ratio = np.nan
    else:
        ratio = sum_abs_err / sum_ins
    return ratio, sum_abs_err, sum_ins


def plot_scatter(df: pd.DataFrame, out_path: str, title_suffix: str = ""):
    x = df["ins_counts"].values
    y = df["pred_counts"].values

    # 计算界限
    max_val = max(int(x.max()) if len(x) else 1, int(y.max()) if len(y) else 1)
    pad = max(1, int(0.05 * max_val))
    upper = max_val + pad

    plt.figure(figsize=(8, 8))
    plt.scatter(x, y, s=10, alpha=0.6)

    # 加粗加深对角线 y=x
    plt.plot([0, upper], [0, upper], linewidth=2.8, color="black", alpha=0.9)

    plt.xlim(0, upper)
    plt.ylim(0, upper)
    plt.gca().set_aspect("equal", adjustable="box")

    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    plt.xlabel("ins_counts")
    plt.ylabel("pred_counts")

    import matplotlib.ticker as ticker

    tick_interval = 50  # 自定义间隔
    plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(tick_interval))
    plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(tick_interval))

    # 标题包含统计信息
    ratio, sum_abs_err, sum_ins = compute_error_ratio(df)
    if np.isnan(ratio):
        ratio_str = "NaN"
        pct_str = "NaN"
    else:
        ratio_str = f"{ratio:.6f}"
        pct_str = f"{ratio * 100:.2f}%"

    title = f"Scatter: ins_counts vs pred_counts {title_suffix}\n" \
            f"sum(|ins - pred|)={sum_abs_err}, sum(ins)={sum_ins}, ratio={ratio_str} ({pct_str})"
    plt.title(title)

    # 保存
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        required=False,
        default="/public/home/zncui/pycharm_project/mmdetection/tools/myscripts/wheat_work/mmdet_all_result.csv",
        help="输入 CSV 文件路径（包含列 filename, ins_counts, pred_counts）",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=False,
        default="mmdet_all_result_scatter.png",
        help="输出散点图路径（.png/.jpg 等）",
    )
    args = parser.parse_args()

    csv_path = args.input
    out_path = args.output

    if not os.path.exists(csv_path):
        print(f"[错误] 找不到输入文件：{csv_path}", file=sys.stderr)
        sys.exit(1)

    df = load_csv(csv_path)
    ratio, sum_abs_err, sum_ins = compute_error_ratio(df)

    # 控制台打印指标
    if np.isnan(ratio):
        print("sum(ins_counts) = 0，错误率无法计算（NaN）。")
    else:
        print(f"图像数: {len(df)}")
        print(f"∑|ins - pred| = {sum_abs_err}")
        print(f"∑ins_counts = {sum_ins}")
        print(f"错误率 ratio = ∑|ins - pred| / ∑ins = {ratio:.6f} ({ratio * 100:.2f}%)")

    # 绘图
    title_suffix = f"(N={len(df)})"
    plot_scatter(df, out_path, title_suffix=title_suffix)
    print(f"散点图已保存到：{out_path}")


if __name__ == "__main__":
    main()
