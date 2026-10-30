#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
按 labelme 标注的 area 区域裁剪原图。

输入目录下每个 jpg 都有一个同名 labelme json，json 中恰好一个 label=area 的区域：
  - rectangle（2 个对角点）：直接裁剪其包围盒。
  - polygon（4 个点等）：裁剪包围盒后，把多边形外、包围盒内的多余区域填黑。

输出到 <输入目录>-area/，文件名与源 jpg 同名。
"""

import os
import glob
import json
import logging

import cv2
import numpy as np

SRC_DIR = "/public/home/zncui/datasets/wheat-guo-dataset/CZ师兄小区"
DST_DIR = SRC_DIR.rstrip("/") + "-area"

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s', datefmt='%H:%M:%S'
)
logger = logging.getLogger('crop-area')


def load_region(json_path: str):
    """读 labelme json，返回 (poly[float32 Nx2], shape_type, W, H)。只取第一片区域。"""
    with open(json_path, 'r', encoding='utf-8') as f:
        d = json.load(f)
    W, H = d.get('imageWidth'), d.get('imageHeight')
    shape = d['shapes'][0]
    pts = np.array(shape['points'], dtype=np.float32)
    st = shape.get('shape_type')
    if st == 'rectangle':
        # labelme 矩形是两个对角点，转成轴对齐 4 角点
        x1, y1 = pts.min(axis=0)
        x2, y2 = pts.max(axis=0)
        poly = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
    else:
        poly = pts
    return poly, st, W, H


def main():
    os.makedirs(DST_DIR, exist_ok=True)

    img_paths = sorted(glob.glob(os.path.join(SRC_DIR, "*.jpg")))
    if not img_paths:
        logger.warning("未发现 jpg 图片。")
        return
    logger.info(f"共 {len(img_paths)} 张图，结果输出到: {DST_DIR}")

    done = 0
    for img_path in img_paths:
        name = os.path.splitext(os.path.basename(img_path))[0]
        json_path = os.path.join(SRC_DIR, name + ".json")
        if not os.path.exists(json_path):
            logger.warning(f"缺少 json，跳过: {json_path}")
            continue

        img = cv2.imread(img_path)  # BGR
        if img is None:
            logger.warning(f"无法读取图片，跳过: {img_path}")
            continue
        H, W = img.shape[:2]

        poly, st, jW, jH = load_region(json_path)
        if jW and jH and (jW != W or jH != H):
            logger.warning(f"{name}: json尺寸({jW}x{jH}) 与图片({W}x{H})不一致，按图片尺寸处理。")

        # 包围盒（clip 到图片范围内）
        x1 = int(np.floor(poly[:, 0].min())); x1 = max(0, min(x1, W))
        y1 = int(np.floor(poly[:, 1].min())); y1 = max(0, min(y1, H))
        x2 = int(np.ceil(poly[:, 0].max()));  x2 = max(0, min(x2, W))
        y2 = int(np.ceil(poly[:, 1].max()));  y2 = max(0, min(y2, H))
        if x2 <= x1 or y2 <= y1:
            logger.warning(f"{name}: 区域为空，跳过。")
            continue

        crop = img[y1:y2, x1:x2].copy()

        # 四边形/多边形：把多边形外区域填黑
        if st != 'rectangle':
            mask = np.zeros(crop.shape[:2], dtype=np.uint8)
            shifted = (poly - np.array([x1, y1], dtype=np.float32)).astype(np.int32)
            cv2.fillPoly(mask, [shifted.reshape(-1, 1, 2)], 255)
            crop = cv2.bitwise_and(crop, crop, mask=mask)

        out_path = os.path.join(DST_DIR, name + ".jpg")
        cv2.imwrite(out_path, crop)
        done += 1
        logger.info(f"{name}: {st} -> {crop.shape[1]}x{crop.shape[0]}  保存 {out_path}")

    logger.info(f"完成 {done}/{len(img_paths)} 张。输出目录: {DST_DIR}")


if __name__ == "__main__":
    main()
