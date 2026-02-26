#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import os.path as osp
import numpy as np
import cv2
import mmcv
from typing import List, Tuple, Dict, Any
from mmdet.apis import DetInferencer
from pycocotools import mask as coco_mask

# ====== 路径与预设 ======
PROJECT_ROOT = '/public/home/zncui/pycharm_project/mmdetection/'
TEST2017_ROOT = '/public/home/zncui/datasets/wheats_rle_coco/coco/test2017'

PRESET = {
    'resnet101_maskrcnn': dict(
        model='work_dirs/mask-rcnn_x101-64x4d_fpn_ms-poly_3x_coco/mask-rcnn_x101-64x4d_fpn_ms-poly_3x_coco (副本).py',
        weights='work_dirs/mask-rcnn_x101-64x4d_fpn_ms-poly_3x_coco/epoch_12.pth',
    ),
    'dinov3_vit7b': dict(
        model='work_dirs/mask-rcnn_x101-64x4d_fpn_ms-poly_3x_coco_1/mask-rcnn_x101-64x4d_fpn_ms-poly_3x_coco.py',
        weights='work_dirs/mask-rcnn_x101-64x4d_fpn_ms-poly_3x_coco_1/epoch_12.pth',
    ),
    'dinov3_convnext_tiny_maskrcnn': dict(
        model='work_dirs/mask-rcnn_x101-64x4d_fpn_ms-poly_3x_coco_2/mask-rcnn_x101-64x4d_fpn_ms-poly_3x_coco.py',
        weights='work_dirs/mask-rcnn_x101-64x4d_fpn_ms-poly_3x_coco_2/epoch_12.pth',
    ),
}

def build_inferencer(choice: str) -> DetInferencer:
    assert choice in PRESET, f'Unknown preset {choice}, available: {list(PRESET.keys())}'
    cfg = PRESET[choice]
    # 替换为绝对路径
    cfg = {k: (v if osp.isabs(v) else osp.join(PROJECT_ROOT, v)) for k, v in cfg.items()}
    inferencer = DetInferencer(**cfg)
    return inferencer

# ====== 工具：统一把各种 mask -> (N,H,W) bool（仅在非流式场景使用）======
def _to_bool_masks(masks, H: int, W: int) -> np.ndarray:
    """将各种可能的 mask 表示统一成 (N,H,W) 的 bool ndarray。"""
    if masks is None:
        return np.zeros((0, H, W), dtype=bool)

    # numpy 数组：常见为 (N,H,W) 或 (H,W,N)
    if isinstance(masks, np.ndarray):
        arr = masks
        if arr.ndim == 3:
            # 若是 (H,W,N) -> (N,H,W)
            if arr.shape[0] == H and arr.shape[1] == W and arr.shape[2] != 0:
                arr = arr.transpose(2, 0, 1)
        elif arr.ndim == 2:
            arr = arr[None, ...]  # 单张 -> (1,H,W)
        arr = arr.astype(bool)
        # 尺寸不匹配则逐个 resize
        if arr.size > 0 and (arr.shape[1] != H or arr.shape[2] != W):
            resized = []
            for m in arr:
                m = cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
                resized.append(m)
            arr = np.stack(resized, axis=0) if resized else np.empty((0, H, W), dtype=bool)
        return arr

    # list：可能是 RLE 或 polygon
    if isinstance(masks, list):
        if len(masks) == 0:
            return np.empty((0, H, W), dtype=bool)

        # COCO RLE：元素为 dict 且含 'counts'
        if isinstance(masks[0], dict) and 'counts' in masks[0]:
            decoded = coco_mask.decode(masks)  # (H,W) 或 (H,W,N)
            if decoded.ndim == 2:
                decoded = decoded[..., None]
            # (H,W,N) -> (N,H,W)
            return np.ascontiguousarray(decoded.transpose(2, 0, 1)).astype(bool)

        # polygon：list[list[float]] 或多段 polygon
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

# ====== 安全取字段的小工具（避免对 ndarray 用“or”）======
def _get_first_non_none(d: Dict[str, Any], keys: List[str]):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None

# ====== 辅助：从 inferencer 的单图输出中抽取 numpy 结果 ======
def _extract_single_pred_dict(pred: Dict[str, Any], H: int, W: int):
    """
    统一解析 DetInferencer 单张图的输出:
    返回：bboxes (N,4, xyxy), scores (N,), labels (N,), masks_bool (N,H,W) or None
    """
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

    # 统一为 (N,4)
    if bboxes.ndim == 1 and bboxes.size == 4:
        bboxes = bboxes[None, :]
    # 统一为 (N,H,W) bool
    masks_bool = None
    if masks is not None:
        masks_bool = _to_bool_masks(masks, H, W)
    return bboxes, scores, labels, masks_bool

# ====== 生成滑窗 ======
def gen_sliding_windows(H: int, W: int, tile: int = 800, overlap: int = 256):
    """生成 [x1,y1,x2,y2] 的窗口坐标（含最后一块对齐到右/下边）"""
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

# ====== （旧）bbox NMS，保留但不再使用 ======
def nms_per_class(boxes: np.ndarray, scores: np.ndarray, iou_thr: float = 0.5) -> List[int]:
    """对单类 boxes 做 NMS，返回保留索引（相对 boxes 的局部索引）。本项目已不使用，仅留作参考。"""
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

# ====== 新：基于 mask 的 IoMin（交 / min(面积)） ======

def _sparse_mask_area(sm: Dict[str, Any]) -> int:
    """计算稀疏掩码的像素面积"""
    if sm is None:
        return 0
    # sm['mask'] 为 bool 或 0/1
    m = sm['mask']
    return int(np.count_nonzero(m))

def _roi_intersection(a: List[int], b: List[int]) -> Tuple[int, int, int, int]:
    """两个 ROI（x1,y1,x2,y2）的交集，返回 (x1,y1,x2,y2)。若无交集则 x1>=x2 或 y1>=y2。"""
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    return x1, y1, x2, y2

def _sparse_masks_intersection(sm_i: Dict[str, Any], sm_j: Dict[str, Any]) -> int:
    """计算两个稀疏掩码的交集像素数（在全局坐标对齐后再 AND）"""
    # 任意一方缺失，则认为交集为 0（因为我们不使用 bbox）
    if sm_i is None or sm_j is None:
        return 0

    mi, mj = sm_i['mask'].astype(bool), sm_j['mask'].astype(bool)
    oi, oj = sm_i['offset'], sm_j['offset']  # [x1,y1,x2,y2]

    # 先求 ROI 交集，若无交集直接返回 0
    x1, y1, x2, y2 = _roi_intersection(oi, oj)
    if x1 >= x2 or y1 >= y2:
        return 0

    # 截取各自 ROI 内的重叠子块
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
    """
    IoMin = |Mi ∩ Mj| / min(|Mi|, |Mj|)，若某个 mask 缺失，则返回 0（不进行压制）。
    """
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
    """
    对同一类别的候选（给定它们的全局索引 idxs）执行 mask-NMS（IoMin）。
    返回：保留的**全局**索引列表。
    规则：
      - 使用分数从高到低遍历；
      - 若与任何已保留实例的 IoMin > 阈值，则压制；
      - 任一对中存在缺失 mask（None），不进行压制（因为不使用 bbox 做近似）。
    """
    if idxs.size == 0:
        return []

    # 按分数降序排序（索引是全局）
    order = idxs[np.argsort(scores[idxs])[::-1]]

    # 预先计算参与者的面积，避免重复 sum
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
            # 仅当二者都有掩码才计算 IoMin，否则不压制
            if sm_i is not None and sm_j is not None:
                iomin = mask_iomin(sm_i, area_i, sm_j, area_j)
                if iomin > iomin_thr:
                    suppressed = True
                    break
        if not suppressed:
            keep.append(j)

    return keep

# ====== 滑窗推理并全局合并（NMS 改为 mask-IoMin） ======
def infer_image_sliding(
    inferencer: DetInferencer,
    image_bgr: np.ndarray,
    tile: int = 800,
    overlap: int = 256,
    score_thr: float = 0.05,
    nms_iou_thr: float = 0.5,   # 这里的阈值被用于 IoMin（交/小面积）
    drop_edge_touch: int = 0,   # >0 时丢弃在 tile 边缘接触宽度>=该值的实例（减少跨窗重复，按需使用）
) -> Dict[str, Any]:
    """
    返回 DetInferencer 风格的结果字典：
    {
      'predictions': [{
          'bboxes': (N,4) xyxy,
          'scores': (N,),
          'labels': (N,),
          # 稀疏掩码（与最终实例一一对应）：
          # 每个元素：{'mask': <bool 2D tile_mask>, 'offset': [x1,y1,x2,y2]} 或 None
          'masks_sparse': List[Dict]
      }]
    }
    """
    H, W = image_bgr.shape[:2]

    all_boxes = []
    all_scores = []
    all_labels = []
    sparse_masks = []  # 与 all_* 同步追加（逐实例）

    for (x1, y1, x2, y2) in gen_sliding_windows(H, W, tile, overlap):
        tile_img = image_bgr[y1:y2, x1:x2]

        out = inferencer(tile_img, return_vis=False)

        # 取第1张（单图）
        pred = out.get('predictions', [out])[0]
        bboxes_t, scores_t, labels_t, masks_t = _extract_single_pred_dict(pred, y2 - y1, x2 - x1)

        if bboxes_t is None or bboxes_t.size == 0:
            continue

        # score 过滤
        keep = scores_t >= score_thr if scores_t.size > 0 else np.zeros((bboxes_t.shape[0],), dtype=bool)
        if not np.any(keep):
            continue

        b = bboxes_t[keep].copy()
        s = scores_t[keep].copy()
        l = labels_t[keep].copy()

        # 偏移到全局坐标
        b[:, [0, 2]] += x1
        b[:, [1, 3]] += y1

        # 可选：丢弃贴边实例，减少跨窗重复（根据需要打开）
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
                    kept_idx.append(i)  # 相对 keep 后的索引
                    kept_local_masks.append(m_local.astype(bool))
            if len(kept_idx) == 0:
                continue
            b = b[kept_idx]
            s = s[kept_idx]
            l = l[kept_idx]
            masks_local_kept = kept_local_masks
        else:
            # 直接使用对应 local masks
            masks_local_kept = []
            if masks_t is not None:
                for m in masks_t[keep]:
                    masks_local_kept.append(m.astype(bool))

        # 累积
        all_boxes.append(b)
        all_scores.append(s)
        all_labels.append(l)
        if len(masks_local_kept) > 0:
            for mloc in masks_local_kept:
                sparse_masks.append({'mask': mloc, 'offset': [x1, y1, x2, y2]})
        else:
            for _ in range(b.shape[0]):
                # 没有 mask（例如仅检测到 bbox）
                sparse_masks.append(None)

    if len(all_boxes) == 0:
        return {'predictions': [{
            'bboxes': np.zeros((0, 4), dtype=np.float32),
            'scores': np.zeros((0,), dtype=np.float32),
            'labels': np.zeros((0,), dtype=np.int64),
            'masks_sparse': []
        }]}

    # 拼接
    all_boxes = np.concatenate(all_boxes, axis=0)
    all_scores = np.concatenate(all_scores, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    assert len(sparse_masks) == all_boxes.shape[0]

    # ---- 按类做全局 Mask-NMS（IoMin）----
    final_boxes = []
    final_scores = []
    final_labels = []
    final_sparse_masks = []

    for cls in np.unique(all_labels):
        idx = np.where(all_labels == cls)[0]  # 该类的全局索引
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

# ====== 可视化（支持流式稀疏掩码，避免一次性解码 NxHxW）======
def save_mask_overlay(image_path: str, prediction: dict, out_dir: str, alpha: float = 0.5, draw_contour: bool = True) -> str:
    """将实例 mask 叠加到原图并保存到同目录，返回输出路径。支持 masks_sparse 流式可视化。"""
    img = mmcv.imread(image_path)  # BGR
    H, W = img.shape[:2]

    pred = prediction.get('predictions', [prediction])[0]

    # 读取分数与掩码源（避免对 ndarray 使用 or）
    if 'pred_instances' in pred:
        inst = pred['pred_instances']
        scores = _get_first_non_none(inst, ['scores', 'score'])
        masks = inst.get('masks', None)
        masks_sparse = pred.get('masks_sparse', None)  # 允许两者并存
        bboxes = _get_first_non_none(inst, ['bboxes', 'bbox'])
        labels = inst.get('labels', None)
    else:
        scores = _get_first_non_none(pred, ['scores', 'score'])
        masks = pred.get('masks', None)
        masks_sparse = pred.get('masks_sparse', None)
        bboxes = _get_first_non_none(pred, ['bboxes', 'bbox'])
        labels = pred.get('labels', None)

    # 生成绘制顺序（低分先、高分后）
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
        print('No instances. Save original image.')
        mmcv.imwrite(img, out_path)
        print('Saved:', out_path)
        return out_path

    rng = np.random.default_rng(12345)
    colors = rng.integers(0, 256, size=(N, 3), dtype=np.uint8)

    overlay = img.copy()

    # ========== 优先使用稀疏掩码（内存友好）==========
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
        print('Saved:', out_path)
        return out_path

    # ========== 否则：兼容旧路径（可能是 RLE / polygon / (N,H,W)）==========
    if isinstance(masks, list) and len(masks) > 0 and isinstance(masks[0], dict) and 'counts' in masks[0]:
        for idx in order:
            color = colors[idx].tolist()
            m = coco_mask.decode(masks[idx]).astype(bool)  # 单个流式解码
            overlay[m] = (overlay[m] * (1.0 - alpha) + np.array(color, dtype=np.float32) * alpha).astype(np.uint8)
            if draw_contour:
                cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(overlay, cnts, -1, color, 2)

        mmcv.imwrite(overlay, out_path)
        print('Saved:', out_path)
        return out_path

    # 兜底：原来的批量路径（小数量时 OK）
    bool_masks = _to_bool_masks(masks, H, W) if masks is not None else np.zeros((0, H, W), dtype=bool)
    if bool_masks.shape[0] == 0:
        mmcv.imwrite(img, out_path)
        print('Saved:', out_path)
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
    print('Saved:', out_path)
    return out_path

# ====== 主入口 ======
def main():
    # 1) 构建推理器（按需修改预设名）
    inferencer = build_inferencer('resnet101_maskrcnn')
    print(inferencer)

    # 2) 指定待测图片
    # image_path = osp.join(TEST2017_ROOT, 'D6-017-20230529.jpg')
    image_path = osp.join('/public/home/zncui/datasets/wheat-guo-dataset/wheat-guo-square', '21CA01.jpg')
    if not osp.isfile(image_path):
        raise FileNotFoundError(image_path)

    img = mmcv.imread(image_path)  # BGR

    # 3) 选择是否使用滑窗
    USE_SLIDING = True

    if USE_SLIDING:
        prediction = infer_image_sliding(
            inferencer,
            img,
            tile=800,
            overlap=256,
            score_thr=0.05,
            nms_iou_thr=0.1,   # 这里现在是 IoMin 阈值
            drop_edge_touch=0,  # 可设 8~16 以减少跨窗重复
        )
    else:
        # 非滑窗：不改动原生输出；如需同样的 mask-NMS，可把整图 masks 转成 masks_sparse 后复用上面的 NMS 逻辑
        prediction = inferencer(image_path, return_vis=False)

    # 4) 保存叠加可视化结果（流式贴图，内存友好）
    out_dir = './vis'
    save_mask_overlay(image_path, prediction, out_dir=out_dir, alpha=0.5, draw_contour=True)

if __name__ == '__main__':
    main()
