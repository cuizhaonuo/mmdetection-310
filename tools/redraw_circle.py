#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
用更大的红点重画 circle 可视化（覆盖原 circle 目录）。

数据来源：
  - ori/<name>.jpg          原图
  - txt/<subdir>/<name>.txt  YOLO 预测框（cls xc yc w h，归一化）
每个框取中心点画一个红色实心圆，半径比原来(2px)大很多。
"""

import os
import glob
import argparse

import cv2
import numpy as np

RESULT_DIR = "/public/home/zncui/datasets/wheat-guo-dataset/CZ师兄小区-area-results-run"
TXT_SUBDIR = "nms_0.35_score_0.3"   # 与推理 config 的 nms_iou/score_thr 对应
RADIUS = 8                           # 红点半径(px)，由 12 缩小三分之一


def parse_args():
    p = argparse.ArgumentParser(description="用更大的红点重画 circle 可视化")
    p.add_argument("--result-dir", default=RESULT_DIR, help="包含 ori/ 和 txt/ 的结果目录")
    p.add_argument("--txt-subdir", default=TXT_SUBDIR, help="txt/ 下的子目录名")
    p.add_argument("--radius", type=int, default=RADIUS, help="红点半径(px)")
    return p.parse_args()


def main():
    args = parse_args()
    ORI_DIR = os.path.join(args.result_dir, "ori")
    TXT_DIR = os.path.join(args.result_dir, "txt", args.txt_subdir)
    CIRCLE_DIR = os.path.join(args.result_dir, "circle")
    RADIUS = args.radius

    os.makedirs(CIRCLE_DIR, exist_ok=True)
    img_paths = sorted(glob.glob(os.path.join(ORI_DIR, "*.jpg")))
    print(f"共 {len(img_paths)} 张图，红点半径={RADIUS}，输出覆盖到: {CIRCLE_DIR}")

    for img_path in img_paths:
        name = os.path.splitext(os.path.basename(img_path))[0]
        txt_path = os.path.join(TXT_DIR, name + ".txt")
        img = cv2.imread(img_path)  # BGR
        if img is None:
            print(f"  跳过无法读取: {img_path}")
            continue
        H, W = img.shape[:2]

        n = 0
        if os.path.exists(txt_path):
            with open(txt_path) as f:
                for line in f:
                    sp = line.split()
                    if len(sp) < 5:
                        continue
                    xc, yc = float(sp[1]) * W, float(sp[2]) * H
                    cv2.circle(img, (int(xc), int(yc)), RADIUS, (0, 0, 255), -1,
                               lineType=cv2.LINE_AA)
                    n += 1
        else:
            print(f"  缺少 txt: {txt_path}")

        cv2.imwrite(os.path.join(CIRCLE_DIR, name + ".jpg"), img)
        print(f"  {name}: {n} 点")

    print("完成。")


if __name__ == "__main__":
    main()
