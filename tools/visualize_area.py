#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
按 labelme 标注区域过滤已有检测结果，并重新可视化。

输入：
  - 图片 + 同名 labelme json（label=area，rectangle 2 点 或 polygon 4 点）
    位于 config.data 的 input_dir（CZ师兄小区）
  - 已有检测 YOLO txt：results_dir/txt/nms_{nms_iou}_score_{score_thr}/<name>.txt

逻辑：只保留“框中心点落在区域内”的检测，重新画一张可视化图。

输出到 area_dir（= output_dir 去掉结尾斜杠后加 "-area"）：
  - vis/<name>.jpg     ：区域轮廓 + 区域内框(绿) + 中心点(红) + 计数
  - txt/<subdir>/<name>.txt ：过滤后的 YOLO txt（坐标仍按整图归一化）
  - counts.csv         ：image,in_area_count,total_count（含合计行）
"""

import os
import csv
import json
import glob
import logging

import cv2
import numpy as np

from infer_wheat import load_config  # 复用配置加载，保证路径口径一致

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s', datefmt='%H:%M:%S'
)
logger = logging.getLogger('wheat-area')


def load_region_polygon(json_path: str):
    """读 labelme json，返回 (poly_pts[float32 Nx2], W, H)。只取第一片区域。"""
    with open(json_path, 'r', encoding='utf-8') as f:
        d = json.load(f)
    W, H = d.get('imageWidth'), d.get('imageHeight')
    shape = d['shapes'][0]
    pts = np.array(shape['points'], dtype=np.float32)
    st = shape.get('shape_type')
    if st == 'rectangle':
        # labelme 矩形是两个对角点，转成 4 角点（轴对齐）
        x1, y1 = pts.min(axis=0)
        x2, y2 = pts.max(axis=0)
        poly = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
    else:
        # polygon / 其它：直接用所有点
        poly = pts
    return poly, W, H


def load_yolo_boxes(txt_path: str, W: int, H: int):
    """读 YOLO txt，返回 list[(cls, xc_n, yc_n, w_n, h_n, cx_px, cy_px, x1,y1,x2,y2)]。"""
    out = []
    with open(txt_path, 'r') as f:
        for line in f:
            sp = line.strip().split()
            if len(sp) < 5:
                continue
            cls = int(float(sp[0]))
            xc, yc, w, h = map(float, sp[1:5])
            cx, cy = xc * W, yc * H
            x1 = (xc - w / 2) * W
            y1 = (yc - h / 2) * H
            x2 = (xc + w / 2) * W
            y2 = (yc + h / 2) * H
            out.append((cls, xc, yc, w, h, cx, cy, x1, y1, x2, y2))
    return out


def main():
    cfg_path = os.environ.get("WHEAT_CFG", "config.yaml")
    cfg = load_config(cfg_path)

    input_dir = cfg.data.input_dir
    results_dir = cfg.data.output_dir
    subdir = f"nms_{cfg.nms.nms_iou}_score_{cfg.nms.score_thr}"
    txt_dir = os.path.join(results_dir, "txt", subdir)

    area_dir = os.path.join(cfg.data.dataset_dir, cfg.data.output_rel.rstrip("/") + "-area")
    vis_dir = os.path.join(area_dir, "vis")
    out_txt_dir = os.path.join(area_dir, "txt", subdir)
    os.makedirs(vis_dir, exist_ok=True)
    os.makedirs(out_txt_dir, exist_ok=True)

    img_paths = []
    for ext in cfg.data.image_exts:
        img_paths.extend(glob.glob(os.path.join(input_dir, f"*{ext}")))
    img_paths.sort()
    if not img_paths:
        logger.warning("未发现图片。")
        return
    logger.info(f"共 {len(img_paths)} 张图，结果输出到: {area_dir}")

    rows = []
    sum_in = sum_total = 0

    for img_path in img_paths:
        name = os.path.splitext(os.path.basename(img_path))[0]
        json_path = os.path.join(input_dir, name + ".json")
        txt_path = os.path.join(txt_dir, name + ".txt")

        if not os.path.exists(json_path):
            logger.warning(f"缺少 json，跳过: {json_path}")
            continue
        if not os.path.exists(txt_path):
            logger.warning(f"缺少检测 txt，跳过: {txt_path}")
            continue

        poly, jW, jH = load_region_polygon(json_path)

        img = cv2.imread(img_path)  # BGR
        if img is None:
            logger.warning(f"无法读取图片，跳过: {img_path}")
            continue
        H, W = img.shape[:2]
        # 以图片实际尺寸为准（json 尺寸应一致）
        if jW and jH and (jW != W or jH != H):
            logger.warning(f"{name}: json尺寸({jW}x{jH}) 与图片({W}x{H})不一致，按图片尺寸处理。")

        boxes = load_yolo_boxes(txt_path, W, H)
        total = len(boxes)

        poly_int = poly.astype(np.int32).reshape(-1, 1, 2)
        kept = []
        for b in boxes:
            cx, cy = b[5], b[6]
            if cv2.pointPolygonTest(poly_int, (float(cx), float(cy)), False) >= 0:
                kept.append(b)
        in_cnt = len(kept)

        # ---- 重新可视化 ----
        cv2.polylines(img, [poly_int], isClosed=True, color=(0, 255, 255), thickness=4)
        for b in kept:
            x1, y1, x2, y2 = map(int, b[7:11])
            cx, cy = int(b[5]), int(b[6])
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(img, (cx, cy), 4, (0, 0, 255), -1, lineType=cv2.LINE_AA)
        cv2.putText(img, f"in-area: {in_cnt}/{total}", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 4, cv2.LINE_AA)
        cv2.imwrite(os.path.join(vis_dir, name + ".jpg"), img)

        # ---- 过滤后的 YOLO txt ----
        with open(os.path.join(out_txt_dir, name + ".txt"), "w") as f:
            for b in kept:
                cls, xc, yc, w, h = b[0], b[1], b[2], b[3], b[4]
                f.write(f"{cls} {xc} {yc} {w} {h}\n")

        rows.append((os.path.basename(img_path), in_cnt, total))
        sum_in += in_cnt
        sum_total += total
        logger.info(f"{name}: 区域内 {in_cnt} / 总 {total}")

    # ---- 汇总 ----
    with open(os.path.join(area_dir, "counts.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "in_area_count", "total_count"])
        w.writerows(rows)
        w.writerow(["TOTAL", sum_in, sum_total])

    logger.info(f"完成 {len(rows)} 张。区域内合计 {sum_in} / 总 {sum_total}。"
                f" 汇总: {os.path.join(area_dir, 'counts.csv')}")


if __name__ == "__main__":
    main()
