#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统计 labelme 标注 box 范围内的预测数量。

- labelme json 位于 <results_run>/tmp/，每个文件只有一个 rectangle 框，
  表示该图限定的统计区域（像素坐标，[[x1,y1],[x2,y2]]）。
- 预测结果为 YOLO 归一化 txt，位于 <results_run>/txt/<nms_..._score_...>/。
- 判定规则：预测框的中心点落在 box 内则计数。

用法：
    python tools/count_in_region.py
可选参数见 main() 顶部常量。
"""

import os
import glob
import json

# ------- 路径配置 -------
RESULTS_RUN = "/public/home/zncui/datasets/wheat-guo-dataset/CZ师兄小区-results-run"
TMP_DIR = os.path.join(RESULTS_RUN, "tmp")
TXT_ROOT = os.path.join(RESULTS_RUN, "txt")
# txt 下的子目录（nms/score 参数）。为 None 时自动取第一个子目录。
TXT_SUBDIR = None


def find_txt_subdir() -> str:
    if TXT_SUBDIR is not None:
        return os.path.join(TXT_ROOT, TXT_SUBDIR)
    subs = [d for d in sorted(glob.glob(os.path.join(TXT_ROOT, "*"))) if os.path.isdir(d)]
    if not subs:
        raise FileNotFoundError(f"在 {TXT_ROOT} 下未找到 txt 子目录")
    return subs[0]


def load_region(json_path: str):
    """返回 (x1, y1, x2, y2, W, H)。"""
    d = json.load(open(json_path, "r", encoding="utf-8"))
    W = d.get("imageWidth")
    H = d.get("imageHeight")
    # 取第一个 rectangle 框
    shape = None
    for s in d["shapes"]:
        if s.get("shape_type") == "rectangle":
            shape = s
            break
    if shape is None:
        shape = d["shapes"][0]
    (px1, py1), (px2, py2) = shape["points"][0], shape["points"][1]
    x1, x2 = sorted([px1, px2])
    y1, y2 = sorted([py1, py2])
    return x1, y1, x2, y2, W, H


def load_pred_centers(txt_path: str, W: int, H: int):
    """读取 YOLO txt，返回像素坐标下的中心点列表 [(cx, cy), ...]。"""
    centers = []
    if not os.path.exists(txt_path):
        return centers
    with open(txt_path, "r") as f:
        for line in f:
            sp = line.strip().split()
            if len(sp) < 5:
                continue
            _, xc, yc, w, h = map(float, sp[:5])
            centers.append((xc * W, yc * H))
    return centers


def main():
    txt_dir = find_txt_subdir()
    print(f"标注目录: {TMP_DIR}")
    print(f"预测目录: {txt_dir}")
    print("-" * 72)
    print(f"{'图片':<12}{'区域计数':>10}{'全图计数':>10}{'区域占比':>10}   box(x1,y1,x2,y2)")
    print("-" * 72)

    json_files = sorted(glob.glob(os.path.join(TMP_DIR, "*.json")))
    if not json_files:
        print(f"未在 {TMP_DIR} 找到 json 标注文件")
        return

    total_region = 0
    total_all = 0
    for jp in json_files:
        name = os.path.splitext(os.path.basename(jp))[0]
        x1, y1, x2, y2, W, H = load_region(jp)
        txt_path = os.path.join(txt_dir, name + ".txt")
        centers = load_pred_centers(txt_path, W, H)

        in_region = sum(1 for (cx, cy) in centers if x1 <= cx <= x2 and y1 <= cy <= y2)
        all_cnt = len(centers)
        ratio = (in_region / all_cnt) if all_cnt else 0.0
        total_region += in_region
        total_all += all_cnt

        box_str = f"({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f})"
        print(f"{name:<12}{in_region:>10}{all_cnt:>10}{ratio:>9.1%}   {box_str}")

    print("-" * 72)
    print(f"{'合计':<12}{total_region:>10}{total_all:>10}")


if __name__ == "__main__":
    main()
