#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
将 COCO 分割标注（polygon 或 RLE）转换为 DOTA OBB (oriented bounding box) 标注。
同时复制图片为 DOTA 目录结构，并在 <DOTA_ROOT>/vis/<split>/ 下保存 OBB 可视化结果。

用法（直接回车即可使用你的默认路径）：
    python coco2dota_obb.py \
        --coco-root /public/home/zncui/datasets/wheats_rle_coco/coco \
        --dota-root /public/home/zncui/datasets/wheats_rle_dota
"""

import os
import os.path as osp
import json
import argparse
import shutil
from collections import defaultdict
from typing import List, Tuple, Dict, Any, Union

import numpy as np
import cv2

try:
    from pycocotools import mask as coco_mask
    HAS_COCO = True
except Exception:
    HAS_COCO = False


# ------------------------------ 工具函数 ------------------------------

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def copy_if_needed(src: str, dst: str):
    ensure_dir(osp.dirname(dst))
    if not osp.exists(dst):
        shutil.copy2(src, dst)


def _order_points_clockwise(pts: np.ndarray) -> np.ndarray:
    assert pts.shape == (4, 2)
    c = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])  # [-pi, pi]
    order = np.argsort(-angles)  # 顺时针
    pts = pts[order]
    # 旋转起点到 (x+y) 最小
    start = np.argmin(pts.sum(axis=1))
    pts = np.roll(pts, -start, axis=0)
    return pts


def _clip_points(pts: np.ndarray, w: int, h: int) -> np.ndarray:
    pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
    return pts


def polygon_to_points(seg_polygons: List[List[float]]) -> np.ndarray:
    """COCO polygon: [x1,y1,x2,y2,...]（可能多段），拼成 Nx2 点集"""
    all_pts = []
    for poly in seg_polygons:
        arr = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
        if arr.shape[0] >= 3:
            all_pts.append(arr)
    if not all_pts:
        return np.zeros((0, 2), dtype=np.float32)
    return np.concatenate(all_pts, axis=0)


def mask_to_points(binary_mask: np.ndarray) -> np.ndarray:
    bin_uint8 = binary_mask.astype(np.uint8)
    if bin_uint8.max() == 0:
        return np.zeros((0, 2), dtype=np.float32)
    contours, _ = cv2.findContours(bin_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return np.zeros((0, 2), dtype=np.float32)
    pts = np.concatenate(contours, axis=0).reshape(-1, 2).astype(np.float32)
    return pts


def compute_obb_from_points(pts: np.ndarray, img_wh: Tuple[int, int]) -> np.ndarray:
    w, h = img_wh
    if pts.shape[0] < 3:
        return np.zeros((0, 2), dtype=np.float32)
    rect = cv2.minAreaRect(pts)   # (center(x,y), (w,h), angle)
    box = cv2.boxPoints(rect)     # 4x2
    box = _clip_points(box, w, h)
    box = _order_points_clockwise(box)
    return box


def hbb_to_obb_from_bbox(bbox_xywh: List[float]) -> np.ndarray:
    """兜底：当 segmentation 不可用时，从 HBB 生成“无旋转”的 OBB。bbox: [x, y, w, h]"""
    x, y, w, h = bbox_xywh
    box = np.array(
        [[x, y], [x + w, y], [x + w, y + h], [x, y + h]],
        dtype=np.float32
    )
    return _order_points_clockwise(box)


def draw_obb(img: np.ndarray, obb: np.ndarray, color: Tuple[int, int, int], text: str = None):
    pts = obb.astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(img, [pts], isClosed=True, color=color, thickness=2)
    if text:
        p = tuple(obb[0].astype(int).tolist())
        cv2.putText(img, text, (p[0] + 3, p[1] - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def name_from_id(cat_id: int, id2name: Dict[int, str]) -> str:
    return id2name.get(cat_id, f"cat_{cat_id}")


# ------------------------------ RLE 解码更鲁棒实现 ------------------------------

def _decode_single_rle(rle_obj: Dict[str, Any], img_h: int, img_w: int) -> np.ndarray:
    """
    解码单个 RLE 对象为 2D mask (H, W)；兼容 counts 为 str（压缩）或 list（非压缩）。
    """
    assert HAS_COCO, "需要 pycocotools 来解码 RLE segmentation，请先 pip install pycocotools"

    # 取优先使用 rle 自带的 size，若没有就用图像尺寸
    size = rle_obj.get("size", [img_h, img_w])
    if not isinstance(size, (list, tuple)) or len(size) != 2:
        size = [img_h, img_w]
    rh, rw = int(size[0]), int(size[1])

    counts = rle_obj.get("counts", None)
    if counts is None:
        return np.zeros((img_h, img_w), dtype=np.uint8)

    # 压缩 RLE：counts 是字符串（需要 bytes）
    if isinstance(counts, str):
        rle = {"size": [rh, rw], "counts": counts.encode("ascii", "ignore")}
        m = coco_mask.decode(rle)
    # 非压缩 RLE：counts 是整数列表
    elif isinstance(counts, (list, tuple)):
        # 用 frPyObjects 正规化
        rle = coco_mask.frPyObjects({"size": [rh, rw], "counts": list(counts)}, rh, rw)
        m = coco_mask.decode(rle)
    else:
        # 其他意外类型，尝试最后一次：直接交给 frPyObjects
        rle = coco_mask.frPyObjects(rle_obj, rh, rw)
        m = coco_mask.decode(rle)

    # m 可能是 (H, W) 或 (H, W, N)
    if m.ndim == 3:
        m = (m.sum(axis=2) > 0).astype(np.uint8)
    else:
        m = (m > 0).astype(np.uint8)

    # 如果 rle 的 size 与图像尺寸不一致，不强行拉伸，这里只是用于求 OBB，保持原 mask 尺寸
    return m


def rle_to_mask(seg: Union[Dict[str, Any], List[Dict[str, Any]]], img_h: int, img_w: int) -> np.ndarray:
    """
    seg 为 RLE dict 或 RLE dict 列表（多连通域），返回 2D mask。
    """
    if isinstance(seg, dict):
        return _decode_single_rle(seg, img_h, img_w)
    elif isinstance(seg, list) and len(seg) > 0 and isinstance(seg[0], dict):
        # 多个 RLE 的并集
        union = np.zeros((img_h, img_w), dtype=np.uint8)
        for r in seg:
            m = _decode_single_rle(r, img_h, img_w)
            # 尺寸不一致时，跳过该片段（避免错误）
            if m.shape[0] != img_h or m.shape[1] != img_w:
                continue
            union |= m
        return union
    else:
        return np.zeros((img_h, img_w), dtype=np.uint8)


def segmentation_to_obb(seg: Any, image_size: Tuple[int, int]) -> np.ndarray:
    """
    seg 可能是 polygon(list)、RLE(dict)、或 RLE(dict) 列表。返回 4x2 OBB 顶点。
    """
    img_h, img_w = image_size

    # polygon：list[ list[float] ]
    if isinstance(seg, list) and len(seg) > 0 and isinstance(seg[0], (list, tuple, np.ndarray)):
        pts = polygon_to_points(seg)
        if pts.shape[0] >= 3:
            return compute_obb_from_points(pts, (img_w, img_h))
        return np.zeros((0, 2), dtype=np.float32)

    # RLE: dict 或 list[dict]
    if isinstance(seg, (dict, list)):
        m = rle_to_mask(seg, img_h, img_w)
        pts = mask_to_points(m)
        if pts.shape[0] >= 3:
            return compute_obb_from_points(pts, (img_w, img_h))
        return np.zeros((0, 2), dtype=np.float32)

    return np.zeros((0, 2), dtype=np.float32)


# ------------------------------ 主流程 ------------------------------

def convert_split(coco_root: str, dota_root: str, split_coco: str, split_dota: str):
    img_dir_src = osp.join(coco_root, split_coco)
    ann_json = osp.join(coco_root, "annotations", f"instances_{split_coco}.json")
    has_ann = osp.exists(ann_json)

    # DOTA 目录
    img_dir_dst = osp.join(dota_root, split_dota, "images")
    label_dir_dst = osp.join(dota_root, split_dota, "labelTxt")
    vis_dir_dst = osp.join(dota_root, "vis", split_dota)
    ensure_dir(img_dir_dst)
    ensure_dir(label_dir_dst)
    ensure_dir(vis_dir_dst)

    imgid_to_meta = {}
    imgid_to_anns = defaultdict(list)
    id2name = {}

    if has_ann:
        with open(ann_json, "r", encoding="utf-8") as f:
            coco = json.load(f)

        for c in coco.get("categories", []):
            id2name[int(c["id"])] = c["name"]

        for img in coco.get("images", []):
            imgid_to_meta[int(img["id"])] = dict(
                file_name=img["file_name"],
                width=int(img["width"]),
                height=int(img["height"])
            )

        for ann in coco.get("annotations", []):
            img_id = int(ann["image_id"])
            imgid_to_anns[img_id].append(ann)

        img_ids = list(imgid_to_meta.keys())
    else:
        img_files = [f for f in os.listdir(img_dir_src) if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"))]
        img_ids = list(range(1, len(img_files) + 1))
        for i, fn in zip(img_ids, img_files):
            fp = osp.join(img_dir_src, fn)
            try:
                img = cv2.imread(fp, cv2.IMREAD_UNCHANGED)
                h, w = img.shape[:2]
            except Exception:
                h, w = 0, 0
            imgid_to_meta[i] = dict(file_name=fn, width=w, height=h)

    print(f"[{split_dota}] images: {len(img_ids)}, anns: {sum(len(v) for v in imgid_to_anns.values())}")

    for idx, img_id in enumerate(img_ids, 1):
        meta = imgid_to_meta[img_id]
        file_name = meta["file_name"]
        w, h = meta["width"], meta["height"]

        src_img_path = osp.join(img_dir_src, file_name)
        dst_img_path = osp.join(img_dir_dst, file_name)
        copy_if_needed(src_img_path, dst_img_path)

        base_noext = osp.splitext(osp.basename(file_name))[0]
        label_txt_path = osp.join(label_dir_dst, base_noext + ".txt")

        vis_img = None
        vis_path = osp.join(vis_dir_dst, file_name)

        objects_lines: List[str] = []
        anns = imgid_to_anns.get(img_id, [])
        if len(anns) == 0:
            with open(label_txt_path, "w", encoding="utf-8") as f:
                f.write("")
        else:
            vis_img = cv2.imread(src_img_path, cv2.IMREAD_COLOR)

            for ann in anns:
                cat_id = int(ann["category_id"])
                cat_name = name_from_id(cat_id, id2name)

                obb = np.zeros((0, 2), dtype=np.float32)
                seg = ann.get("segmentation", None)

                if seg not in (None, [], [{}]):
                    try:
                        obb = segmentation_to_obb(seg, (h, w))
                    except Exception as e:
                        # 更简洁的告警信息，方便定位
                        print(f"  [warn] seg->obb 失败 image_id={img_id}, ann_id={ann.get('id')}: {e}")

                if obb.size == 0 and "bbox" in ann and ann["bbox"]:
                    obb = hbb_to_obb_from_bbox(ann["bbox"])

                if obb.size == 0:
                    continue

                obb_round = np.round(obb, 2).reshape(-1).tolist()
                line = "{:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {} {}".format(
                    obb_round[0], obb_round[1], obb_round[2], obb_round[3],
                    obb_round[4], obb_round[5], obb_round[6], obb_round[7],
                    cat_name, 0
                )
                objects_lines.append(line)

                if vis_img is not None:
                    color_seed = (hash(cat_name) % 255, (hash(cat_name) // 255) % 255, (hash(cat_name) // 65025) % 255)
                    color = tuple(int(c) for c in color_seed)
                    draw_obb(vis_img, obb, color=color, text=cat_name)

            with open(label_txt_path, "w", encoding="utf-8") as f:
                f.write("\n".join(objects_lines))

            if vis_img is not None:
                ensure_dir(osp.dirname(vis_path))
                cv2.imwrite(vis_path, vis_img)

        if idx % 100 == 0 or idx == len(img_ids):
            print(f"  [{split_dota}] processed {idx}/{len(img_ids)}")


def main():
    parser = argparse.ArgumentParser(description="COCO segmentation -> DOTA OBB converter with visualization")
    parser.add_argument("--coco-root", type=str, default="/public/home/zncui/datasets/wheats_rle_coco/coco")
    parser.add_argument("--dota-root", type=str, default="/public/home/zncui/datasets/wheats_rle_dota")
    args = parser.parse_args()

    coco_root = args.coco_root
    dota_root = args.dota_root

    ensure_dir(dota_root)

    # train
    if osp.isdir(osp.join(coco_root, "train2017")):
        ann_train = osp.join(coco_root, "annotations", "instances_train2017.json")
        if osp.exists(ann_train) or osp.isdir(osp.join(coco_root, "train2017")):
            convert_split(coco_root, dota_root, "train2017", "train")

    # val
    if osp.isdir(osp.join(coco_root, "val2017")):
        ann_val = osp.join(coco_root, "annotations", "instances_val2017.json")
        if osp.exists(ann_val) or osp.isdir(osp.join(coco_root, "val2017")):
            convert_split(coco_root, dota_root, "val2017", "val")

    # test（无标注也会复制图并生成空 labelTxt）
    if osp.isdir(osp.join(coco_root, "test2017")):
        convert_split(coco_root, dota_root, "test2017", "test")

    print(f"完成！DOTA 目录在：{dota_root}")
    print(f"可视化样例在：{osp.join(dota_root, 'vis')}")


if __name__ == "__main__":
    main()
