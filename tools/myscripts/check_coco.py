#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
COCO instance segmentation 可视化 + 一致性检测脚本

功能：
- 在 <root>/vis 下生成 segmentation/ 与 detection/ 两类可视化图（区分 train/val/test）
- 检测每个 annotation 的 bbox 与 segmentation 是否一致，输出 CSV 报告

用法（默认你的数据根目录）：
    python coco_vis_and_check.py \
        --root /public/home/zncui/datasets/wheats_rle_coco/coco \
        --draw  \
        --check

只可视化或只检测：分别保留 --draw 或 --check。
可选阈值见 argparse。
"""

import os
import json
import math
import csv
import argparse
from collections import defaultdict
from typing import Dict, List, Tuple, Any

import cv2
import numpy as np
from tqdm import tqdm

try:
    from pycocotools import mask as maskUtils
except Exception as e:
    raise RuntimeError("需要安装 pycocotools: pip install pycocotools") from e


# -------------------------- 配置 --------------------------
DEFAULT_ROOT = "/public/home/zncui/datasets/wheats_rle_coco/coco"
SUBSETS = ["train2017", "val2017", "test2017"]
ANN_FILES = {
    "train2017": "annotations/instances_train2017.json",
    "val2017": "annotations/instances_val2017.json",
    "test2017": "annotations/instances_test2017.json",  # 若不存在会自动跳过
}

# 颜色表（BGR）
PALETTE = [
    (46, 204, 113), (52, 152, 219), (155, 89, 182), (241, 196, 15),
    (230, 126, 34), (231, 76, 60), (26, 188, 156), (52, 73, 94),
    (149, 165, 166), (243, 156, 18), (41, 128, 185), (39, 174, 96),
]

# 一致性检测阈值（可在命令行覆盖）
IOU_MIN = 0.85         # bbox 与 mask 紧包围框的 IoU 下限（越高越严格）
INSIDE_MIN = 0.98      # mask 像素在 bbox 内的比例下限
REL_EPS = 0.05         # bbox 与 mask 紧包围框的相对误差容忍（宽/高）
ABS_EPS_PX = 2.0       # bbox 与 mask 紧包围框的绝对像素误差容忍
AREA_TOL = 0.10        # ann['area'] 与 mask 像素面积允许的相对误差


# ----------------------- 工具函数 -------------------------
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def color_for_cat(cat_id: int) -> Tuple[int, int, int]:
    # 稳定映射：不同类别有不同颜色；单类数据也OK
    return PALETTE[(cat_id - 1) % len(PALETTE)]


def load_coco(ann_path: str) -> Tuple[Dict[int, Any], Dict[int, Any], Dict[int, str]]:
    """读取 COCO 标注：返回 images_dict, anns_by_img, catid_to_name"""
    with open(ann_path, 'r', encoding='utf-8') as f:
        coco = json.load(f)

    images_dict = {im['id']: im for im in coco.get('images', [])}
    catid_to_name = {c['id']: c.get('name', str(c['id'])) for c in coco.get('categories', [])}

    anns_by_img: Dict[int, List[dict]] = defaultdict(list)
    for ann in coco.get('annotations', []):
        img_id = ann['image_id']
        anns_by_img[img_id].append(ann)

    return images_dict, anns_by_img, catid_to_name


def polygon_or_rle_to_mask(segm, h: int, w: int) -> np.ndarray:
    """
    将 segmentation（polygon 或 RLE）统一解码为 HxW 的 bool mask。
    兼容：
      - polygon: List[List[float]]
      - RLE 压缩：{"size":[h,w], "counts":"..."}
      - RLE 非压缩：{"size":[h,w], "counts":[...]}
    """
    if isinstance(segm, list):
        # polygon(s)
        rles = maskUtils.frPyObjects(segm, h, w)
        m = maskUtils.decode(rles)  # HxW or HxWxN
        if m.ndim == 3:
            m = np.any(m, axis=2)
        return m.astype(bool)

    if isinstance(segm, dict):
        rle = segm
        # 兼容 counts 为 str 的情况（部分 pycocotools 版本需要 bytes）
        counts = rle.get("counts")
        if isinstance(counts, str):
            rle = {"size": rle["size"], "counts": counts.encode('utf-8')}
        m = maskUtils.decode(rle)
        if m.ndim == 3:
            # 理论上 RLE 单个实例应该是 HxW；这里兜底
            m = np.any(m, axis=2)
        return m.astype(bool)

    raise ValueError(f"Unsupported segmentation format: {type(segm)}")


def draw_mask_on_image(img: np.ndarray, mask: np.ndarray, color: Tuple[int,int,int], alpha: float = 0.5):
    """半透明叠加 mask；并描边轮廓"""
    overlay = img.copy()
    overlay[mask] = (np.array(color, dtype=np.uint8) * 0.6 + overlay[mask] * 0.4).astype(np.uint8)
    img[:] = ((1 - alpha) * img + alpha * overlay).astype(np.uint8)

    # 轮廓（只画边）
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, cnts, -1, color, thickness=2)


def put_label(img: np.ndarray, text: str, org: Tuple[int, int], color=(255,255,255), bg=(0,0,0)):
    """带底色的文字，防止看不清"""
    x, y = org
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    thickness = 1
    (tw, th), bl = cv2.getTextSize(text, font, scale, thickness)
    cv2.rectangle(img, (x, y - th - 2), (x + tw + 2, y + 2), bg, -1)
    cv2.putText(img, text, (x + 1, y - 2), font, scale, color, thickness, cv2.LINE_AA)


def xywh_to_xyxy(b: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    x, y, w, h = b
    return x, y, x + w, y + h


def xyxy_to_xywh(b: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = b
    return x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)


def bbox_iou_xywh(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = xywh_to_xyxy(a)
    bx1, by1, bx2, by2 = xywh_to_xyxy(b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
    area_b = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
    union = area_a + area_b - inter + 1e-6
    return float(inter / union)


def mask_tight_bbox(mask: np.ndarray) -> Tuple[float, float, float, float]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return 0.0, 0.0, 0.0, 0.0
    x1, y1 = int(xs.min()), int(ys.min())
    x2, y2 = int(xs.max()) + 1, int(ys.max()) + 1
    return float(x1), float(y1), float(x2 - x1), float(y2 - y1)


def mask_area(mask: np.ndarray) -> int:
    return int(mask.sum())


def inside_ratio(mask: np.ndarray, bbox_xywh: Tuple[float,float,float,float]) -> float:
    x, y, w, h = bbox_xywh
    H, W = mask.shape
    x1 = max(0, int(math.floor(x)))
    y1 = max(0, int(math.floor(y)))
    x2 = min(W, int(math.ceil(x + w)))
    y2 = min(H, int(math.ceil(y + h)))
    if x1 >= x2 or y1 >= y2:
        return 0.0
    sub = mask[y1:y2, x1:x2]
    inside = int(sub.sum())
    total = int(mask.sum()) + 1e-6
    return float(inside / total)


def rel_abs_ok(a: float, b: float, rel_eps=REL_EPS, abs_eps=ABS_EPS_PX) -> bool:
    return (abs(a - b) <= abs_eps) or (abs(a - b) <= rel_eps * max(1.0, abs(b)))


# --------------------- 可视化与检测主流程 --------------------
def visualize_subset(root: str, subset: str, images_dict, anns_by_img, catid_to_name,
                     draw_seg_dir: str, draw_det_dir: str, draw: bool = True):
    """绘制 segmentation 与 detection，可选"""
    if not draw:
        return

    ensure_dir(draw_seg_dir)
    ensure_dir(draw_det_dir)

    # 若没有标注（例如 test2017），直接创建空目录并返回
    if not anns_by_img:
        return

    for img_id, iminfo in tqdm(images_dict.items(), desc=f"[VIS] {subset}", unit="img"):
        file_name = iminfo['file_name']
        H, W = int(iminfo['height']), int(iminfo['width'])
        img_path = os.path.join(root, subset, file_name)
        if not os.path.isfile(img_path):
            # 有些数据集 file_name 带子目录；尝试直接拼上
            alt_path = os.path.join(root, file_name)
            if os.path.isfile(alt_path):
                img_path = alt_path
            else:
                print(f"[WARN] Missing image: {img_path}")
                continue

        img = cv2.imread(img_path)
        if img is None:
            print(f"[WARN] Failed to read image: {img_path}")
            continue

        img_seg = img.copy()
        img_det = img.copy()

        anns = anns_by_img.get(img_id, [])
        for ann in anns:
            cat_id = int(ann['category_id'])
            color = color_for_cat(cat_id)
            # 1) segmentation 可视化
            try:
                mask = polygon_or_rle_to_mask(ann['segmentation'], H, W)
                if mask.any():
                    draw_mask_on_image(img_seg, mask, color=color, alpha=0.35)
                    # 在 mask 的质心处打 label
                    M = cv2.moments(mask.astype(np.uint8))
                    if M["m00"] > 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        put_label(img_seg, f"{catid_to_name.get(cat_id,'id:'+str(cat_id))}#{ann['id']}", (max(0, cx-20), max(20, cy)))
            except Exception as e:
                print(f"[WARN] seg decode failed: img={file_name}, ann_id={ann['id']}, err={e}")

            # 2) bbox 可视化
            if 'bbox' in ann:
                x, y, w, h = ann['bbox']
                p1, p2 = (int(x), int(y)), (int(x + w), int(y + h))
                cv2.rectangle(img_det, p1, p2, color, thickness=2)
                put_label(img_det, f"{catid_to_name.get(cat_id,'id:'+str(cat_id))}#{ann['id']}", (p1[0], max(15, p1[1])))

        out_seg = os.path.join(draw_seg_dir, file_name)
        out_det = os.path.join(draw_det_dir, file_name)
        ensure_dir(os.path.dirname(out_seg))
        ensure_dir(os.path.dirname(out_det))
        cv2.imwrite(out_seg, img_seg)
        cv2.imwrite(out_det, img_det)


def check_consistency_subset(root: str, subset: str, images_dict, anns_by_img,
                             csv_writer, iou_min=IOU_MIN, inside_min=INSIDE_MIN,
                             rel_eps=REL_EPS, abs_eps=ABS_EPS_PX, area_tol=AREA_TOL):
    """检测 bbox 与 segmentation 是否一致，写入 csv 报告"""
    # 若没有标注（例如 test2017），直接返回
    if not anns_by_img:
        return

    for img_id, iminfo in tqdm(images_dict.items(), desc=f"[CHECK] {subset}", unit="img"):
        file_name = iminfo['file_name']
        H, W = int(iminfo['height']), int(iminfo['width'])
        anns = anns_by_img.get(img_id, [])
        for ann in anns:
            reasons = []
            segm = ann.get('segmentation', None)
            bbox = ann.get('bbox', None)
            if segm is None or bbox is None:
                continue

            try:
                mask = polygon_or_rle_to_mask(segm, H, W)
            except Exception as e:
                reasons.append(f"seg_decode_fail:{e}")
                mask = None

            if mask is None or not mask.any():
                reasons.append("mask_empty")
                iou = 0.0
                inside = 0.0
                mask_bbox = (0.0, 0.0, 0.0, 0.0)
                mask_area = 0
            else:
                mask_bbox = mask_tight_bbox(mask)
                mask_area = mask_area_px = mask_area(mask)
                iou = bbox_iou_xywh(tuple(bbox), mask_bbox)
                inside = inside_ratio(mask, tuple(bbox))

                # 尺寸/位置误差
                x, y, w, h = bbox
                mx, my, mw, mh = mask_bbox
                if not (rel_abs_ok(x, mx, rel_eps, abs_eps) and
                        rel_abs_ok(y, my, rel_eps, abs_eps) and
                        rel_abs_ok(w, mw, rel_eps, abs_eps) and
                        rel_abs_ok(h, mh, rel_eps, abs_eps)):
                    reasons.append("bbox_vs_maskbbox_mismatch")

                if iou < iou_min:
                    reasons.append(f"low_iou:{iou:.3f}")

                if inside < inside_min:
                    reasons.append(f"low_inside:{inside:.3f}")

                # area 字段校验（非必须，但常见）
                ann_area = float(ann.get('area', -1))
                if ann_area > 0:
                    rel_diff = abs(ann_area - mask_area_px) / max(1.0, mask_area_px)
                    if rel_diff > area_tol:
                        reasons.append(f"area_mismatch:{rel_diff:.3f}")

            # 写报告（仅记录有问题的样本；如需全量，把判断去掉）
            if reasons:
                csv_writer.writerow({
                    "subset": subset,
                    "image_id": img_id,
                    "ann_id": ann.get('id', -1),
                    "category_id": ann.get('category_id', -1),
                    "file_name": file_name,
                    "reasons": "|".join(reasons),
                    "iou": f"{iou:.6f}",
                    "inside_ratio": f"{inside:.6f}",
                    "bbox_given": json.dumps([round(float(v), 3) for v in ann.get('bbox', [0,0,0,0])]),
                    "bbox_from_mask": json.dumps([round(float(v), 3) for v in mask_bbox]),
                    "area_given": ann.get('area', -1),
                    "area_mask": int(mask.sum()) if (mask is not None) else -1,
                })


def process(root: str, do_draw: bool, do_check: bool,
            iou_min: float, inside_min: float, rel_eps: float, abs_eps: float, area_tol: float):
    # 输出目录
    vis_root = os.path.join(root, "vis")
    seg_root = os.path.join(vis_root, "segmentation")
    det_root = os.path.join(vis_root, "detection")
    ensure_dir(seg_root); ensure_dir(det_root)

    # 结果 CSV
    csv_path = os.path.join(vis_root, "bbox_mask_inconsistencies.csv")
    csv_file = open(csv_path, "w", newline='', encoding="utf-8")
    csv_writer = csv.DictWriter(csv_file, fieldnames=[
        "subset", "image_id", "ann_id", "category_id", "file_name",
        "reasons", "iou", "inside_ratio", "bbox_given", "bbox_from_mask",
        "area_given", "area_mask"
    ])
    csv_writer.writeheader()

    # 逐子集处理
    for subset in SUBSETS:
        ann_file = os.path.join(root, ANN_FILES[subset])
        # test 可能没有 instances_*.json，跳过 check/vis
        if not os.path.isfile(ann_file):
            # 但仍创建空的输出目录结构
            ensure_dir(os.path.join(seg_root, subset))
            ensure_dir(os.path.join(det_root, subset))
            print(f"[INFO] {subset}: annotation file not found -> {ann_file} (skip draw/check)")
            continue

        images_dict, anns_by_img, catid_to_name = load_coco(ann_file)

        # 1) 可视化（保存到 vis/segmentation/<subset> 与 vis/detection/<subset>）
        visualize_subset(
            root=root,
            subset=subset,
            images_dict=images_dict,
            anns_by_img=anns_by_img,
            catid_to_name=catid_to_name,
            draw_seg_dir=os.path.join(seg_root, subset),
            draw_det_dir=os.path.join(det_root, subset),
            draw=do_draw
        )

        # 2) 一致性检测（CSV 只记录有问题的样本）
        if do_check:
            check_consistency_subset(
                root=root,
                subset=subset,
                images_dict=images_dict,
                anns_by_img=anns_by_img,
                csv_writer=csv_writer,
                iou_min=iou_min,
                inside_min=inside_min,
                rel_eps=rel_eps,
                abs_eps=abs_eps,
                area_tol=area_tol
            )

    csv_file.close()
    print(f"[DONE] 可视化输出: {vis_root}")
    if do_check:
        print(f"[DONE] 一致性报告: {csv_path}")


# ------------------------------ CLI ------------------------------
def parse_args():
    ap = argparse.ArgumentParser(description="COCO segmentation 可视化 + bbox/seg 一致性检测")
    ap.add_argument("--root", type=str, default=DEFAULT_ROOT, help="COCO 根目录（含 annotations/, train2017/, val2017/, test2017/）")
    ap.add_argument("--draw", action="store_true", help="执行可视化（保存 segmentation 与 detection 结果）")
    ap.add_argument("--check", action="store_true", help="执行一致性检测（输出 CSV）")

    ap.add_argument("--iou_min", type=float, default=IOU_MIN, help="bbox 与 mask 紧包围框的 IoU 最小阈值")
    ap.add_argument("--inside_min", type=float, default=INSIDE_MIN, help="mask 像素落在 bbox 内的比例最小阈值")
    ap.add_argument("--rel_eps", type=float, default=REL_EPS, help="bbox 与 maskbbox 允许的相对误差")
    ap.add_argument("--abs_eps", type=float, default=ABS_EPS_PX, help="bbox 与 maskbbox 允许的绝对像素误差")
    ap.add_argument("--area_tol", type=float, default=AREA_TOL, help="area 字段与 mask 像素面积允许的相对误差")

    args = ap.parse_args()
    if not args.draw and not args.check:
        # 默认两者都做
        args.draw = True
        args.check = True
    return args


if __name__ == "__main__":
    args = parse_args()
    process(
        root=args.root,
        do_draw=args.draw,
        do_check=args.check,
        iou_min=args.iou_min,
        inside_min=args.inside_min,
        rel_eps=args.rel_eps,
        abs_eps=args.abs_eps,
        area_tol=args.area_tol
    )
