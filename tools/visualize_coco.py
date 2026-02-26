#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import random
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

# pycocotools 安装: pip install pycocotools
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils


def ann_to_mask(coco, ann, h, w):
    """
    将任意 COCO segmentation 转成 0/1 mask，兼容：
    - polygon: list[list[float]]、list[str]、str（用空格分隔坐标）
    - RLE: dict(size, counts为str或list) 或 直接 str(压缩RLE)
    """
    seg = ann.get("segmentation", None)
    if seg is None:
        return np.zeros((h, w), dtype=np.uint8)

    # ---------- 1) polygon ----------
    if isinstance(seg, list):
        # 可能是规范的 list[list[float]]，也可能是 list[str]
        if len(seg) > 0 and isinstance(seg[0], str):
            # 将每个字符串的坐标转成 float 列表
            poly_list = []
            for s in seg:
                nums = s.strip().split()
                # 转 float；有些数据可能是整型没小数，用 float 处理更稳妥
                coords = [float(x) for x in nums]
                poly_list.append(coords)
            rles = maskUtils.frPyObjects(poly_list, h, w)
        else:
            # 正常的 list[list[float]]
            rles = maskUtils.frPyObjects(seg, h, w)
        rle = maskUtils.merge(rles)
        m = maskUtils.decode(rle)
        return (m.astype(np.uint8))

    # ---------- 2) RLE dict ----------
    if isinstance(seg, dict):
        counts = seg.get("counts", None)
        # counts 可能是 str / bytes / list
        if isinstance(counts, str):
            seg = {"size": seg["size"], "counts": counts.encode("ascii")}
        elif isinstance(counts, bytes):
            seg = {"size": seg["size"], "counts": counts}
        elif isinstance(counts, (list, tuple)):
            seg = {"size": seg["size"], "counts": list(counts)}
        m = maskUtils.decode(seg)
        if m.ndim == 3:
            m = m[:, :, 0]
        return m.astype(np.uint8)

    # ---------- 3) 纯字符串 ----------
    if isinstance(seg, str):
        # 有两种可能：
        # A) 压缩 RLE 字符串（常见）
        # B) 一个长的多边形坐标字符串（不常见，但你的数据就是）
        # 先尝试当作 RLE decode，不行再按 polygon 解析
        try:
            rle = {"size": [h, w], "counts": seg.encode("ascii")}
            m = maskUtils.decode(rle)
            if m.ndim == 3:
                m = m[:, :, 0]
            return m.astype(np.uint8)
        except Exception:
            # 当作 polygon 字符串：空格分隔的 x y x y ...
            nums = seg.strip().split()
            coords = [float(x) for x in nums]
            rles = maskUtils.frPyObjects([coords], h, w)
            rle = maskUtils.merge(rles)
            m = maskUtils.decode(rle)
            return m.astype(np.uint8)

    # ---------- 4) 其他兜底 ----------
    return np.zeros((h, w), dtype=np.uint8)



def color_for_id(ann_id):
    """
    根据 annotation id 生成稳定/不同的颜色 (BGR)
    """
    rnd = random.Random(int(ann_id))
    return (rnd.randint(0, 255), rnd.randint(0, 255), rnd.randint(0, 255))


def overlay_mask(img, mask, color, alpha=0.5, draw_edge=True):
    """
    将二值 mask 半透明叠加到 img 上，颜色为 color(BGR)
    """
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    if mask.max() == 1:
        mask = (mask * 255).astype(np.uint8)

    colored = np.zeros_like(img, dtype=np.uint8)
    colored[:, :] = color
    # 只在 mask 区域混合
    m3 = cv2.merge([mask, mask, mask]) > 0
    out = img.copy()
    out[m3] = cv2.addWeighted(img, 1 - alpha, colored, alpha, 0)[m3]

    if draw_edge:
        contours, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, color, thickness=2)

    return out


def visualize_split(root, split, vis_root):
    """
    可视化一个 split: train2017 / val2017 / test2017 (如果有对应 instances_*.json)
    """
    ann_path = root / "annotations" / f"instances_{split}.json"
    img_dir = root / split
    if not ann_path.exists():
        print(f"[SKIP] 标注不存在: {ann_path}")
        return

    coco = COCO(str(ann_path))
    img_ids = coco.getImgIds()

    out_bbox_dir = vis_root / "bbox" / split
    out_seg_dir = vis_root / "segmentation" / split
    out_bbox_dir.mkdir(parents=True, exist_ok=True)
    out_seg_dir.mkdir(parents=True, exist_ok=True)

    for img_id in tqdm(img_ids, desc=f"Vis {split}"):
        img_info = coco.loadImgs([img_id])[0]
        file_name = img_info["file_name"]
        h, w = img_info["height"], img_info["width"]

        img_path = img_dir / file_name
        if not img_path.exists():
            # 有些数据集 file_name 里可能带子目录；尝试相对路径
            img_path = (root / file_name)
            if not img_path.exists():
                print(f"[WARN] 图像未找到: {file_name}")
                continue

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[WARN] 无法读取图像: {file_name}")
            continue

        # 复制两份
        img_bbox = img.copy()
        img_seg = img.copy()

        ann_ids = coco.getAnnIds(imgIds=[img_id], iscrowd=None)
        anns = coco.loadAnns(ann_ids)

        for ann in anns:
            # ------- 画 bbox（统一红色） -------
            if "bbox" in ann and ann["bbox"] is not None:
                x, y, w_box, h_box = ann["bbox"]
                x1, y1 = int(round(x)), int(round(y))
                x2, y2 = int(round(x + w_box)), int(round(y + h_box))
                cv2.rectangle(img_bbox, (x1, y1), (x2, y2), (0, 0, 255), thickness=2)

            # ------- 画 segmentation（不同颜色 & 半透明） -------
            try:
                m = ann_to_mask(coco, ann, h, w)
                if m is not None and m.any():
                    col = color_for_id(ann.get("id", 0))
                    img_seg = overlay_mask(img_seg, m, col, alpha=0.5, draw_edge=True)
            except Exception as e:
                print(f"[WARN] decode seg 失败: img={file_name}, ann_id={ann.get('id')}, err={e}")

        # 保存
        cv2.imwrite(str(out_bbox_dir / file_name), img_bbox)
        cv2.imwrite(str(out_seg_dir / file_name), img_seg)


def main():
    # 数据根目录
    root = Path(os.path.expanduser("~/datasets/wheats_rle_coco"))
    vis_root = root / "vis"

    splits = ["train2017", "val2017", "test2017"]  # 有哪个就画哪个
    for sp in splits:
        visualize_split(root, sp, vis_root)

    print("✔ 完成。结果已保存到:", vis_root)


if __name__ == "__main__":
    main()
