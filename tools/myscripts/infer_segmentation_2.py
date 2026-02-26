#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import os.path as osp
import json
import glob
import csv
from typing import List, Tuple, Dict, Any

import numpy as np
import cv2
import mmcv
from mmdet.apis import DetInferencer
from pycocotools import mask as coco_mask

# ===================== 路径与数据目录 =====================
PROJECT_ROOT = '/public/home/zncui/pycharm_project/mmdetection/'

# 输入两个数据集目录（按你的说明）
DATASET_COCO_DIR = "/public/home/zncui/datasets/zhaoxian/wheats_rle_coco/coco"  # 包含 test2017 与 annotations
DATASET_YOLO_DIR = "/public/home/zncui/datasets/wheat-guo-dataset/yolo_wheat_big_image"  # images/, labels/ 各含 train/val/test

# 输出 CSV 文件（当前目录）
RESULT_CSV = "./mask_result.csv"

# ===================== 推理模型预设 =====================
PRESET = {
    'resnet101_maskrcnn': dict(
        model='work_dirs/mask-rcnn_x101-64x4d_fpn_ms-poly_3x_coco/mask-rcnn_x101-64x4d_fpn_ms-poly_3x_coco (副本).py',
        weights='work_dirs/mask-rcnn_x101-64x4d_fpn_ms-poly_3x_coco/epoch_12.pth',
    )
}

def build_inferencer(choice: str) -> DetInferencer:
    assert choice in PRESET, f'Unknown preset {choice}, available: {list(PRESET.keys())}'
    cfg = PRESET[choice]
    # 替换为绝对路径
    cfg = {k: (v if osp.isabs(v) else osp.join(PROJECT_ROOT, v)) for k, v in cfg.items()}
    inferencer = DetInferencer(**cfg)
    return inferencer

# ===================== 掩码与预测解析工具 =====================
def _to_bool_masks(masks, H: int, W: int) -> np.ndarray:
    """将各种可能的 mask 表示统一成 (N,H,W) 的 bool ndarray。"""
    if masks is None:
        return np.zeros((0, H, W), dtype=bool)

    if isinstance(masks, np.ndarray):
        arr = masks
        if arr.ndim == 3:
            if arr.shape[0] == H and arr.shape[1] == W and arr.shape[2] != 0:
                arr = arr.transpose(2, 0, 1)
        elif arr.ndim == 2:
            arr = arr[None, ...]
        arr = arr.astype(bool)
        if arr.size > 0 and (arr.shape[1] != H or arr.shape[2] != W):
            resized = []
            for m in arr:
                m = cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
                resized.append(m)
            arr = np.stack(resized, axis=0) if resized else np.empty((0, H, W), dtype=bool)
        return arr

    if isinstance(masks, list):
        if len(masks) == 0:
            return np.empty((0, H, W), dtype=bool)
        if isinstance(masks[0], dict) and 'counts' in masks[0]:
            decoded = coco_mask.decode(masks)
            if decoded.ndim == 2:
                decoded = decoded[..., None]
            return np.ascontiguousarray(decoded.transpose(2, 0, 1)).astype(bool)

        bools = []
        for poly in masks:
            m = np.zeros((H, W), dtype=np.uint8)
            parts = poly if isinstance(poly[0], (list, tuple, np.ndarray)) else [poly]
            for p in parts:
                pts = np.array(p, dtype=np.int32).reshape(-1, 2)
                cv2.fillPoly(m, [pts], 1)
            bools.append(m.astype(bool))
        return np.stack(bools, 0) if bools else np.empty((0, H, W), dtype=bool)

    return np.empty((0, H, W), dtype=bool)

def _get_first_non_none(d: Dict[str, Any], keys: List[str]):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None

def _extract_single_pred_dict(pred: Dict[str, Any], H: int, W: int):
    """统一解析 DetInferencer 单图输出: 返回 bboxes,scores,labels,masks_bool"""
    if 'pred_instances' in pred:
        inst = pred['pred_instances']
        bboxes = np.asarray(_get_first_non_none(inst, ['bboxes', 'bbox']) or [], dtype=np.float32)
        scores = np.asarray(_get_first_non_none(inst, ['scores', 'score']) or [], dtype=np.float32).reshape(-1)
        labels = np.asarray(inst.get('labels', []), dtype=np.int64).reshape(-1)
        masks  = inst.get('masks', None)
    else:
        bboxes = np.asarray(_get_first_non_none(pred, ['bboxes', 'bbox']) or [], dtype=np.float32)
        scores = np.asarray(_get_first_non_none(pred, ['scores', 'score']) or [], dtype=np.float32).reshape(-1)
        labels = np.asarray(pred.get('labels', []), dtype=np.int64).reshape(-1)
        masks  = pred.get('masks', None)

    if bboxes.ndim == 1 and bboxes.size == 4:
        bboxes = bboxes[None, :]
    masks_bool = None
    if masks is not None:
        masks_bool = _to_bool_masks(masks, H, W)
    return bboxes, scores, labels, masks_bool

# ===================== 滑窗生成 =====================
def gen_sliding_windows(H: int, W: int, tile: int = 800, overlap: int = 256):
    assert tile > overlap, "tile 必须大于 overlap"
    stride = tile - overlap
    xs = list(range(0, max(W - tile, 0) + 1, stride))
    ys = list(range(0, max(H - tile, 0) + 1, stride))
    if xs[-1] != W - tile:
        xs.append(W - tile)
    if ys[-1] != H - tile:
        ys.append(H - tile)
    for y in ys:
        for x in xs:
            yield [x, y, x + tile, y + tile]

# =====================（旧）bbox NMS：保留参考 =====================
def nms_per_class(boxes: np.ndarray, scores: np.ndarray, iou_thr: float = 0.5) -> List[int]:
    if boxes.size == 0:
        return []
    order = scores.argsort()[::-1]
    keep = []
    b = boxes.astype(np.float32)
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(b[i, 0], b[order[1:], 0])
        yy1 = np.maximum(b[i, 1], b[order[1:], 1])
        xx2 = np.minimum(b[i, 2], b[order[1:], 2])
        yy2 = np.minimum(b[i, 3], b[order[1:], 3])
        inter_w = np.maximum(0.0, xx2 - xx1)
        inter_h = np.maximum(0.0, yy2 - yy1)
        inter = inter_w * inter_h
        area_i = (b[i, 2] - b[i, 0]) * (b[i, 3] - b[i, 1])
        area_o = (b[order[1:], 2] - b[order[1:], 0]) * (b[order[1:], 3] - b[order[1:], 1])
        iou = inter / (area_i + area_o - inter + 1e-6)
        remain = np.where(iou <= iou_thr)[0]
        order = order[remain + 1]
    return keep

# ===================== 新：基于 mask 的 IoMin（交 / min(面积)） =====================
def _sparse_mask_area(sm: Dict[str, Any]) -> int:
    if sm is None:
        return 0
    m = sm['mask']
    return int(np.count_nonzero(m))

def _roi_intersection(a: List[int], b: List[int]) -> Tuple[int, int, int, int]:
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    return x1, y1, x2, y2

def _sparse_masks_intersection(sm_i: Dict[str, Any], sm_j: Dict[str, Any]) -> int:
    if sm_i is None or sm_j is None:
        return 0
    mi, mj = sm_i['mask'].astype(bool), sm_j['mask'].astype(bool)
    oi, oj = sm_i['offset'], sm_j['offset']  # [x1,y1,x2,y2]
    x1, y1, x2, y2 = _roi_intersection(oi, oj)
    if x1 >= x2 or y1 >= y2:
        return 0
    si_x1 = x1 - oi[0]; si_x2 = x2 - oi[0]
    si_y1 = y1 - oi[1]; si_y2 = y2 - oi[1]
    sj_x1 = x1 - oj[0]; sj_x2 = x2 - oj[0]
    sj_y1 = y1 - oj[1]; sj_y2 = y2 - oj[1]
    sub_i = mi[si_y1:si_y2, si_x1:si_x2]
    sub_j = mj[sj_y1:sj_y2, sj_x1:sj_x2]
    inter = np.count_nonzero(sub_i & sub_j)
    return int(inter)

def mask_iomin(sm_i: Dict[str, Any], area_i: int,
               sm_j: Dict[str, Any], area_j: int) -> float:
    if sm_i is None or sm_j is None:
        return 0.0
    if area_i <= 0 or area_j <= 0:
        return 0.0
    inter = _sparse_masks_intersection(sm_i, sm_j)
    denom = min(area_i, area_j)
    return float(inter) / float(max(denom, 1))

def mask_nms_keep_indices_global(idxs: np.ndarray,
                                 sparse_masks: List[Dict[str, Any]],
                                 scores: np.ndarray,
                                 iomin_thr: float) -> List[int]:
    if idxs.size == 0:
        return []
    order = idxs[np.argsort(scores[idxs])[::-1]]
    areas = {int(i): _sparse_mask_area(sparse_masks[int(i)]) for i in idxs}
    keep: List[int] = []
    for j in order:
        j = int(j)
        sm_j = sparse_masks[j]
        area_j = areas[j]
        suppressed = False
        for i in keep:
            sm_i = sparse_masks[i]
            area_i = areas[i]
            if sm_i is not None and sm_j is not None:
                iomin = mask_iomin(sm_i, area_i, sm_j, area_j)
                if iomin > iomin_thr:
                    suppressed = True
                    break
        if not suppressed:
            keep.append(j)
    return keep

# ===================== 滑窗推理并全局合并 =====================
def infer_image_sliding(
    inferencer: DetInferencer,
    image_bgr: np.ndarray,
    tile: int = 800,
    overlap: int = 256,
    score_thr: float = 0.05,
    nms_iou_thr: float = 0.5,   # IoMin 阈值
    drop_edge_touch: int = 0,
) -> Dict[str, Any]:
    H, W = image_bgr.shape[:2]
    all_boxes = []
    all_scores = []
    all_labels = []
    sparse_masks = []

    for (x1, y1, x2, y2) in gen_sliding_windows(H, W, tile, overlap):
        tile_img = image_bgr[y1:y2, x1:x2]
        out = inferencer(tile_img, return_vis=False)
        pred = out.get('predictions', [out])[0]
        bboxes_t, scores_t, labels_t, masks_t = _extract_single_pred_dict(pred, y2 - y1, x2 - x1)
        if bboxes_t is None or bboxes_t.size == 0:
            continue

        keep = scores_t >= score_thr if scores_t.size > 0 else np.zeros((bboxes_t.shape[0],), dtype=bool)
        if not np.any(keep):
            continue

        b = bboxes_t[keep].copy()
        s = scores_t[keep].copy()
        l = labels_t[keep].copy()

        b[:, [0, 2]] += x1
        b[:, [1, 3]] += y1

        if drop_edge_touch > 0 and masks_t is not None and masks_t.shape[0] == bboxes_t.shape[0]:
            kept_idx = []
            kept_local_masks = []
            for i, k in enumerate(np.where(keep)[0]):
                m_local = masks_t[k].astype(np.uint8)
                edge = drop_edge_touch
                touch = (
                    m_local[:edge, :].any() or m_local[-edge:, :].any() or
                    m_local[:, :edge].any() or m_local[:, -edge:].any()
                )
                if not touch:
                    kept_idx.append(i)
                    kept_local_masks.append(m_local.astype(bool))
            if len(kept_idx) == 0:
                continue
            b = b[kept_idx]
            s = s[kept_idx]
            l = l[kept_idx]
            masks_local_kept = kept_local_masks
        else:
            masks_local_kept = []
            if masks_t is not None:
                for m in masks_t[keep]:
                    masks_local_kept.append(m.astype(bool))

        all_boxes.append(b)
        all_scores.append(s)
        all_labels.append(l)
        if len(masks_local_kept) > 0:
            for mloc in masks_local_kept:
                sparse_masks.append({'mask': mloc, 'offset': [x1, y1, x2, y2]})
        else:
            for _ in range(b.shape[0]):
                sparse_masks.append(None)

    if len(all_boxes) == 0:
        return {'predictions': [{
            'bboxes': np.zeros((0, 4), dtype=np.float32),
            'scores': np.zeros((0,), dtype=np.float32),
            'labels': np.zeros((0,), dtype=np.int64),
            'masks_sparse': []
        }]}

    all_boxes = np.concatenate(all_boxes, axis=0)
    all_scores = np.concatenate(all_scores, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    assert len(sparse_masks) == all_boxes.shape[0]

    final_boxes = []
    final_scores = []
    final_labels = []
    final_sparse_masks = []

    for cls in np.unique(all_labels):
        idx = np.where(all_labels == cls)[0]
        keep_idx_global = mask_nms_keep_indices_global(idx, sparse_masks, all_scores, iomin_thr=nms_iou_thr)
        if len(keep_idx_global) == 0:
            continue
        keep_idx_global = np.asarray(keep_idx_global, dtype=int)
        final_boxes.append(all_boxes[keep_idx_global])
        final_scores.append(all_scores[keep_idx_global])
        final_labels.append(all_labels[keep_idx_global])
        for gi in keep_idx_global:
            final_sparse_masks.append(sparse_masks[int(gi)])

    if len(final_boxes) == 0:
        return {'predictions': [{
            'bboxes': np.zeros((0, 4), dtype=np.float32),
            'scores': np.zeros((0,), dtype=np.float32),
            'labels': np.zeros((0,), dtype=np.int64),
            'masks_sparse': []
        }]}

    final_boxes = np.concatenate(final_boxes, axis=0)
    final_scores = np.concatenate(final_scores, axis=0)
    final_labels = np.concatenate(final_labels, axis=0)

    return {'predictions': [{
        'bboxes': final_boxes,
        'scores': final_scores,
        'labels': final_labels,
        'masks_sparse': final_sparse_masks
    }]}

# ===================== 可视化（可选） =====================
def save_mask_overlay(image_path: str, prediction: dict, out_dir: str, alpha: float = 0.5, draw_contour: bool = True) -> str:
    img = mmcv.imread(image_path)  # BGR
    H, W = img.shape[:2]
    pred = prediction.get('predictions', [prediction])[0]

    if 'pred_instances' in pred:
        inst = pred['pred_instances']
        scores = _get_first_non_none(inst, ['scores', 'score'])
        masks = inst.get('masks', None)
        masks_sparse = pred.get('masks_sparse', None)
        bboxes = _get_first_non_none(inst, ['bboxes', 'bbox'])
        labels = inst.get('labels', None)
    else:
        scores = _get_first_non_none(pred, ['scores', 'score'])
        masks = pred.get('masks', None)
        masks_sparse = pred.get('masks_sparse', None)
        bboxes = _get_first_non_none(pred, ['bboxes', 'bbox'])
        labels = pred.get('labels', None)

    if scores is not None:
        scores = np.array(scores).reshape(-1)
        N = scores.shape[0]
        order = scores.argsort()
    elif bboxes is not None:
        bboxes_np = np.asarray(bboxes)
        N = bboxes_np.shape[0] if bboxes_np.ndim == 2 else (1 if bboxes_np.size == 4 else 0)
        order = np.arange(N)
    else:
        N = 0
        order = np.array([], dtype=int)

    os.makedirs(out_dir, exist_ok=True)
    out_path = osp.join(out_dir, osp.splitext(osp.basename(image_path))[0] + '_maskvis.jpg')

    if N == 0:
        mmcv.imwrite(img, out_path)
        return out_path

    rng = np.random.default_rng(12345)
    colors = rng.integers(0, 256, size=(N, 3), dtype=np.uint8)
    overlay = img.copy()

    if masks_sparse is not None and len(masks_sparse) == N:
        for idx in order:
            color = colors[idx].tolist()
            sm = masks_sparse[idx]
            if sm is None:
                continue
            local_m = sm['mask'].astype(bool)
            x1, y1, x2, y2 = sm['offset']
            roi = overlay[y1:y2, x1:x2]
            roi_mask = local_m
            roi[roi_mask] = (roi[roi_mask] * (1.0 - alpha) + np.array(color, dtype=np.float32) * alpha).astype(np.uint8)
            overlay[y1:y2, x1:x2] = roi

            if draw_contour:
                cnts, _ = cv2.findContours(local_m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for c in cnts:
                    c = c + np.array([[x1, y1]])
                    cv2.drawContours(overlay, [c], -1, color, 2)

        mmcv.imwrite(overlay, out_path)
        return out_path

    if isinstance(masks, list) and len(masks) > 0 and isinstance(masks[0], dict) and 'counts' in masks[0]:
        for idx in order:
            color = colors[idx].tolist()
            m = coco_mask.decode(masks[idx]).astype(bool)
            overlay[m] = (overlay[m] * (1.0 - alpha) + np.array(color, dtype=np.float32) * alpha).astype(np.uint8)
            if draw_contour:
                cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(overlay, cnts, -1, color, 2)
        mmcv.imwrite(overlay, out_path)
        return out_path

    bool_masks = _to_bool_masks(masks, H, W) if masks is not None else np.zeros((0, H, W), dtype=bool)
    if bool_masks.shape[0] == 0:
        mmcv.imwrite(img, out_path)
        return out_path

    for idx in order:
        m = bool_masks[idx]
        color = colors[idx].tolist()
        if m.shape != (H, W):
            m = cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
        overlay[m] = (overlay[m] * (1.0 - alpha) + np.array(color, dtype=np.float32) * alpha).astype(np.uint8)
        if draw_contour:
            cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, cnts, -1, color, 2)

    mmcv.imwrite(overlay, out_path)
    return out_path

# ===================== 数据集工具：收集图片与标注计数 =====================
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

def list_images(root: str) -> List[str]:
    files = []
    for ext in IMG_EXTS:
        files.extend(glob.glob(osp.join(root, f"*{ext}")))
    files.sort()
    return files

def build_coco_ins_count_map(coco_dir: str) -> Dict[str, int]:
    """读取 COCO annotations（优先尝试 test2017 标注；如果没有，则返回空映射）"""
    ann_dir = osp.join(coco_dir, "annotations")
    # 常见文件名尝试：instances_test2017.json / instances_val2017.json（以防只有 val）
    candidates = [
        osp.join(ann_dir, "instances_test2017.json"),
        osp.join(ann_dir, "instances_val2017.json"),
        osp.join(ann_dir, "instances_train2017.json"),
    ]
    ann_path = None
    for p in candidates:
        if osp.isfile(p):
            ann_path = p
            break
    if ann_path is None:
        print(f"[COCO] 未找到 annotations（{ann_dir}），将对 COCO 图片的 ins_counts 记为 0。")
        return {}

    with open(ann_path, "r") as f:
        data = json.load(f)

    # image_id -> file_name
    id2fname = {img["id"]: img["file_name"] for img in data.get("images", [])}

    # 统计每张图的实例数
    count_map_rel = {fn: 0 for fn in id2fname.values()}
    for ann in data.get("annotations", []):
        img_id = ann.get("image_id")
        fn = id2fname.get(img_id)
        if fn is not None:
            count_map_rel[fn] = count_map_rel.get(fn, 0) + 1

    # 转成绝对路径键：拼接 test2017 目录
    test_root = osp.join(coco_dir, "test2017")
    abs_map: Dict[str, int] = {}
    for rel, cnt in count_map_rel.items():
        abs_path = osp.join(test_root, rel)
        abs_map[osp.abspath(abs_path)] = int(cnt)
    return abs_map

def yolo_label_to_count(label_path: str) -> int:
    """YOLO txt：每行一个实例；忽略空行与注释行。"""
    if not osp.isfile(label_path):
        return 0
    count = 0
    with open(label_path, "r") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            # 保险起见，检查至少包含类别与 4 个框字段
            parts = s.split()
            if len(parts) >= 5:
                count += 1
    return count

# ===================== 主流程 =====================
def main():
    # 1) 构建推理器（按需切换预设名）
    inferencer = build_inferencer('resnet101_maskrcnn')
    print("[INFO] Inferencer:", inferencer)

    # 2) 收集 COCO 与 YOLO 的测试图片清单
    coco_test_dir = osp.join(DATASET_COCO_DIR, "test2017")
    yolo_test_img_dir = osp.join(DATASET_YOLO_DIR, "images", "test")
    yolo_test_lbl_dir = osp.join(DATASET_YOLO_DIR, "labels", "test")

    coco_images = list_images(coco_test_dir)
    yolo_images = list_images(yolo_test_img_dir)

    print(f"[INFO] COCO test images: {len(coco_images)}")
    print(f"[INFO] YOLO test images: {len(yolo_images)}")

    # 3) 构建 COCO 标注计数字典（若无 annotations 则为 {}）
    coco_ins_map = build_coco_ins_count_map(DATASET_COCO_DIR)

    # 4) 逐图推理并记录 CSV
    rows = []  # 每行: (filename, ins_counts, pred_counts)

    # ---- 先处理 COCO ----
    for idx, img_path in enumerate(coco_images, 1):
        img_abs = osp.abspath(img_path)
        print(f"[COCO] ({idx}/{len(coco_images)}) {img_abs}")

        # GT 实例数（无标注文件则为 0）
        ins_cnt = coco_ins_map.get(img_abs, 0)

        # 推理（滑窗）
        image_bgr = mmcv.imread(img_abs)
        pred = infer_image_sliding(
            inferencer,
            image_bgr,
            tile=800,
            overlap=256,
            score_thr=0.05,
            nms_iou_thr=0.1,   # IoMin 阈值（与你示例一致）
            drop_edge_touch=0,
        )
        pred_inst = pred['predictions'][0]
        pred_cnt = int(pred_inst['bboxes'].shape[0]) if isinstance(pred_inst.get('bboxes', None), np.ndarray) else 0

        rows.append((img_abs, ins_cnt, pred_cnt))

        # 如需可视化，请取消注释：
        save_mask_overlay(img_abs, pred, out_dir="./vis_coco", alpha=0.5, draw_contour=True)

    # ---- 再处理 YOLO ----
    for idx, img_path in enumerate(yolo_images, 1):
        img_abs = osp.abspath(img_path)
        print(f"[YOLO] ({idx}/{len(yolo_images)}) {img_abs}")

        # 寻找对应的 label 路径
        stem = osp.splitext(osp.basename(img_abs))[0]
        lbl_path = osp.join(yolo_test_lbl_dir, stem + ".txt")
        ins_cnt = yolo_label_to_count(lbl_path)

        # 推理（滑窗）
        image_bgr = mmcv.imread(img_abs)
        pred = infer_image_sliding(
            inferencer,
            image_bgr,
            tile=800,
            overlap=256,
            score_thr=0.05,
            nms_iou_thr=0.1,
            drop_edge_touch=0,
        )
        pred_inst = pred['predictions'][0]
        pred_cnt = int(pred_inst['bboxes'].shape[0]) if isinstance(pred_inst.get('bboxes', None), np.ndarray) else 0

        rows.append((img_abs, ins_cnt, pred_cnt))

        # 如需可视化，请取消注释：
        # save_mask_overlay(img_abs, pred, out_dir="./vis_yolo", alpha=0.5, draw_contour=True)

    # 5) 写出 CSV
    os.makedirs(osp.dirname(osp.abspath(RESULT_CSV)) or ".", exist_ok=True)
    with open(RESULT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "ins_counts", "pred_counts"])
        for r in rows:
            writer.writerow([r[0], r[1], r[2]])

    print(f"[DONE] Wrote CSV: {osp.abspath(RESULT_CSV)}  (rows={len(rows)})")

if __name__ == '__main__':
    main()
