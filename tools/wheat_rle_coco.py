#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
from typing import Dict, Any, List, Tuple, Union, Iterable

import numpy as np
import cv2
from pycocotools import mask as maskUtils


# ------------------------- 基础工具：start-length 解析/解码 -------------------------

def parse_startlen_list(seg: Any) -> List[Tuple[int, int]]:
    """
    将你的 segmentation 转成 (start, length) 对的列表。
    允许的输入：
      - ["s l s l ..."] （字符串列表，常见）
      - ["s l ...", "s l ..."] （多段）
      - [[s, l, s, l, ...], ...]
    不是此三类则抛错；若本就是 COCO RLE dict（counts/size），本函数应不被调用。
    """
    pairs: List[Tuple[int, int]] = []

    if isinstance(seg, dict) and "counts" in seg and "size" in seg:
        return pairs  # 已是 COCO RLE；调用方会跳过

    if not isinstance(seg, (list, tuple)):
        raise TypeError(f"unexpected segmentation type: {type(seg)}")

    def _nums_from_str(s: str) -> List[int]:
        return [int(x) for x in re.findall(r"\d+", s)]

    for item in seg:
        if isinstance(item, str):
            nums = _nums_from_str(item)
        elif isinstance(item, (list, tuple, np.ndarray)):
            nums = [int(x) for x in item]
        else:
            raise TypeError(f"unexpected inner seg type: {type(item)}")

        if len(nums) % 2 != 0:
            raise ValueError(f"start-length nums must be even, got {len(nums)}")

        # pairs.extend(list(zip(nums[::2], nums[1::2])))
        # 修复原始 start-length 从 1 开始的问题，改为从 0 开始
        pairs.extend([(start - 1, length) for start, length in zip(nums[::2], nums[1::2])])
    return pairs


def decode_startlen_to_mask(
    pairs: List[Tuple[int, int]], H: int, W: int
) -> np.ndarray:
    """
    (start, length)（行优先） → (H,W) 0/1 掩码
    """
    total = H * W
    flat = np.zeros(total, dtype=np.uint8)
    for start, length in pairs:
        if length <= 0:
            continue
        if start >= total:
            continue
        end = min(start + length, total)
        flat[start:end] = 1
    mask = flat.reshape(H, W)  # 行优先
    return mask


def mask_to_coco_rle(mask_hw: np.ndarray) -> Dict[str, Any]:
    """
    (H,W) 0/1 掩码 → COCO RLE（列优先，counts 为 str）
    """
    rle = maskUtils.encode(np.asfortranarray(mask_hw.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("ascii")
    return rle


# ------------------------- 转换：annotations -> new_annotations -------------------------

def convert_one_annotations_file(in_json_path: str, out_json_path: str) -> Dict[str, Any]:
    """
    读取 instances_xxx.json，转换 start-length segmentation 为 COCO RLE，
    重算 area 与 bbox，写出到 out_json_path。
    返回转换后的完整 JSON（dict），供后续可视化使用。
    """
    with open(in_json_path, "r") as f:
        data = json.load(f)

    # 确保 COCO 格式必需的字段存在，避免使用 pycocotools 时出错
    if "info" not in data:
        data["info"] = {}

    if "licenses" not in data:
        data["licenses"] = []

    # image_id -> (H,W)
    id2size: Dict[int, Tuple[int, int]] = {}
    for im in data.get("images", []):
        h, w = int(im["height"]), int(im["width"])
        id2size[int(im["id"])] = (h, w)

    new_annotations = []
    changed, skipped, already_coco, kept_poly = 0, 0, 0, 0

    for ann in data.get("annotations", []):
        seg = ann.get("segmentation", None)
        img_id = int(ann["image_id"])
        if seg is None:
            skipped += 1
            new_annotations.append(ann)
            continue

        # 已经是 COCO RLE
        if isinstance(seg, dict) and "counts" in seg and "size" in seg:
            already_coco += 1
            new_annotations.append(ann)
            continue

        # polygon（list of lists）：保留即可（mmdet 也能用）
        if isinstance(seg, list) and seg and isinstance(seg[0], (list, tuple)):
            kept_poly += 1
            new_annotations.append(ann)
            continue

        H, W = id2size.get(img_id, (None, None))
        if H is None or W is None:
            raise KeyError(f"image_id={img_id} not found in images list for sizes")

        # 按 start-length 解析
        try:
            pairs = parse_startlen_list(seg)
        except Exception as e:
            raise RuntimeError(f"ann id={ann.get('id')} parse segmentation failed: {e}")

        if not pairs:
            # 空或无法解析，保留
            skipped += 1
            new_annotations.append(ann)
            continue

        mask = decode_startlen_to_mask(pairs, H, W)
        if mask.max() == 0:
            skipped += 1
            new_annotations.append(ann)
            continue

        coco_rle = mask_to_coco_rle(mask)
        ann_new = dict(ann)
        ann_new["segmentation"] = coco_rle
        ann_new["area"] = float(maskUtils.area(coco_rle))
        x, y, w, h = [float(v) for v in maskUtils.toBbox(coco_rle)]
        ann_new["bbox"] = [x, y, w, h]
        # 如需强制 crowd，可打开下一行
        # ann_new["iscrowd"] = 1

        new_annotations.append(ann_new)
        changed += 1

    data["annotations"] = new_annotations

    os.makedirs(os.path.dirname(out_json_path), exist_ok=True)
    with open(out_json_path, "w") as f:
        json.dump(data, f)

    print(
        f"[CONVERT] {os.path.basename(in_json_path)} -> {os.path.basename(out_json_path)} | "
        f"converted={changed}, kept_rle={already_coco}, kept_poly={kept_poly}, skipped={skipped}"
    )

    return data


# ------------------------- 可视化：new_annotations → vis/{split} -------------------------

def _random_colors(n: int, seed: int = 2025) -> np.ndarray:
    """
    生成 n 个随机 BGR 颜色（uint8），可复现
    """
    rng = np.random.default_rng(seed)
    cols = rng.integers(0, 256, size=(n, 3), dtype=np.uint8)
    # OpenCV 使用 BGR，本函数直接给 BGR
    return cols


def _decode_seg_to_mask(seg: Any, H: int, W: int) -> np.ndarray:
    """
    解码 segmentation（COCO RLE 或 polygon）到 (H,W) uint8 掩码。
    这里假设已经是 new_annotations：多数为 RLE；若仍有 polygon 也支持。
    """
    if isinstance(seg, dict) and "counts" in seg and "size" in seg:
        # RLE
        m = maskUtils.decode(seg)
        return m.astype(np.uint8)[:, :]  # (H,W,1) -> (H,W)
    elif isinstance(seg, list):
        # polygon(s)
        rles = maskUtils.frPyObjects(seg, H, W)
        m = maskUtils.decode(rles)  # (H,W, N)
        if m.ndim == 3:
            m = (m.max(axis=2) > 0).astype(np.uint8)
        else:
            m = m.astype(np.uint8)
        return m
    else:
        raise TypeError(f"Unsupported segmentation type for visualization: {type(seg)}")


def visualize_split(
    converted_data: Dict[str, Any],
    root: str,
    split: str,
    out_vis_dir: str,
    alpha: float = 0.5,
    draw_contour: bool = False,
) -> None:
    """
    对 new_annotations 的一个 split 进行可视化，按图片聚合实例，给每个实例不同半透明颜色。
    输出到 vis/{split}/
    """
    images = converted_data.get("images", [])
    anns = converted_data.get("annotations", [])

    # image_id -> file_name, H, W
    id2info: Dict[int, Tuple[str, int, int]] = {}
    for im in images:
        id2info[int(im["id"])] = (im["file_name"], int(im["height"]), int(im["width"]))

    # image_id -> [ann,...]
    img2anns: Dict[int, List[Dict[str, Any]]] = {}
    for ann in anns:
        img2anns.setdefault(int(ann["image_id"]), []).append(ann)

    out_seg_dir  = os.path.join(out_vis_dir, "segmentation")
    out_det_dir  = os.path.join(out_vis_dir, "detection")
    os.makedirs(out_seg_dir, exist_ok=True)
    os.makedirs(out_det_dir, exist_ok=True)

    # os.makedirs(out_vis_dir, exist_ok=True)

    # 颜色池：每张图最多用到该图实例数量的颜色
    # 为可复现，每张图用相同 seed + image_id 做偏移
    for img_id, ann_list in img2anns.items():
        file_name, H, W = id2info[img_id]
        img_path = os.path.join(root, split, file_name)
        img = cv2.imread(img_path)
        if img is None:
            print(f"[VIS][WARN] cannot read image: {img_path}")
            continue

        overlay_seg = img.copy()
        overlay_det = img.copy()
        N = len(ann_list)
        colors = _random_colors(N, seed=2025 + img_id)  # 每张图可复现但不同

        for idx, ann in enumerate(ann_list):
            seg = ann.get("segmentation")
            if seg is None:
                continue
            try:
                mask = _decode_seg_to_mask(seg, H, W)  # (H,W) 0/1
            except Exception as e:
                print(f"[VIS][WARN] decode seg failed: ann_id={ann.get('id')} err={e}")
                continue

            if mask.shape[:2] != img.shape[:2]:
                # 以图像尺寸为准
                mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

            m_bool = mask.astype(bool)
            color = colors[idx].astype(np.uint8)

            # 半透明叠加
            overlay_seg[m_bool] = (
                overlay_seg[m_bool] * (1 - alpha) + color.astype(np.float32) * alpha
            ).astype(np.uint8)

            if draw_contour:
                cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(overlay_seg, cnts, -1, (int(color[0]), int(color[1]), int(color[2])), 1)

            # 绘制 bbox
            x, y, w, h = [int(v) for v in ann.get("bbox", [0, 0, 0, 0])]
            cv2.rectangle(overlay_det, (x, y), (x + w, y + h), (int(color[0]), int(color[1]), int(color[2])), 2)

        out_seg_path = os.path.join(out_seg_dir, os.path.splitext(os.path.basename(file_name))[0] + ".png")
        cv2.imwrite(out_seg_path, overlay_seg)
        out_det_path = os.path.join(out_det_dir, os.path.splitext(os.path.basename(file_name))[0] + ".png")
        cv2.imwrite(out_det_path, overlay_det)

    print(f"[VIS] saved visualizations to: {out_vis_dir}")


# ------------------------- 主流程：转换 + 可视化 -------------------------

def main():
    # COCO 数据集根目录（含 annotations/train2017/val2017/test2017）
    root = os.path.expanduser("~/datasets/wheats_rle_coco/coco")
    in_anno_dir = os.path.join(root, "annotations")
    out_anno_dir = os.path.join(root, "new_annotations")
    out_vis_root = os.path.join(root, "vis")
    splits = ["train2017", "val2017", "test2017"]

    for split in splits:
        in_json = os.path.join(in_anno_dir, f"instances_{split}.json")
        if not os.path.exists(in_json):
            print(f"[WARN] missing {in_json}, skip")
            continue

        out_json = os.path.join(out_anno_dir, f"instances_{split}.json")
        converted = convert_one_annotations_file(in_json, out_json)

        # 可视化
        # out_vis_dir = os.path.join(out_vis_root, split)
        # visualize_split(converted, root, split, out_vis_dir, alpha=0.5, draw_contour=False)


if __name__ == "__main__":
    main()
