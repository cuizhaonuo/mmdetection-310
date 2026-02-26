#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Sliding-window inference on COCO(test2017) & YOLO(test) with MMDetection.
- Windowing: size=800, overlap_ratio=0.15, pad_value=[114,114,114], filter_to_valid_region=True
- Post-NMS: custom IoU_min (intersection / min(areaA, areaB))
- Outputs: result.csv with columns: filename, ins_counts, pred_counts
"""

import os
import json
import math
import glob
import warnings
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import cv2
import numpy as np
from tqdm import tqdm

# ========= 1) 配置 =========
@dataclass
class ModelConfig:
    base_dir: str
    config_rel: str
    checkpoint_rel: str
    device: str = "cuda:0"

@dataclass
class SWConfig:
    window_size: int = 800
    overlap_ratio: float = 0.15
    pad_value: Tuple[int, int, int] = (114, 114, 114)
    filter_to_valid_region: bool = True

@dataclass
class NMSConfig:
    iou_type: str = "IoU_min"   # 固定使用 IoU_min
    score_thr: float = 0.4
    nms_iou: float = 0.655
    class_agnostic: bool = True  # 单类场景可设为 True

# ---- 按你提供的路径与参数填写 ----
MODEL_CFG = ModelConfig(
    base_dir="/public/home/zncui/pycharm_project/mmdetection",
    config_rel="work_dirs/co_dino_guo_result/co_dino_5scale_swin_l_16xb1_16e_o365tococo.py",
    checkpoint_rel="work_dirs/co_dino_guo_result/epoch_16.pth",
    device="cuda:0",
)

DATASET_COCO_DIR = "/public/home/zncui/datasets/zhaoxian/wheats_rle_coco/coco"  # 包含 test2017 与 annotations
DATASET_YOLO_DIR = "/public/home/zncui/datasets/wheat-guo-dataset/yolo_wheat_big_image"  # images/, labels/ 各含 train/val/test

SW_CFG = SWConfig(
    window_size=800,
    overlap_ratio=0.15,
    pad_value=(114, 114, 114),
    filter_to_valid_region=True,
)

NMS_CFG = NMSConfig(
    iou_type="IoU_min",
    score_thr=0.4,
    nms_iou=0.655,
    class_agnostic=False,
)

OUTPUT_CSV = "mmdet_all_result.csv"
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


# ========= 2) 工具函数 =========
def ensure_abs_path(base_dir: str, rel_path: str) -> str:
    if os.path.isabs(rel_path):
        return rel_path
    return os.path.join(base_dir, rel_path)

def list_images_in_dir(d: str) -> List[str]:
    files = []
    for ext in IMG_EXTS:
        files.extend(glob.glob(os.path.join(d, f"*{ext}")))
    files.sort()
    return files

def load_mmdet_model(mc: ModelConfig):
    from mmdet.apis import init_detector
    cfg_path = ensure_abs_path(mc.base_dir, mc.config_rel)
    ckpt_path = ensure_abs_path(mc.base_dir, mc.checkpoint_rel)
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    model = init_detector(cfg_path, ckpt_path, device=mc.device)
    return model

def run_mmdet_inference(model, img_bgr: np.ndarray) -> Dict:
    """
    统一提取 MMDetection 输出为：
    {
        'bboxes': Nx5 ndarray [x1,y1,x2,y2,score],
        'labels': Nx1 ndarray (int)
    }
    兼容：
      - MMDet 3.x: 返回 DetDataSample 或 List[DetDataSample]
      - MMDet 2.x: 返回 (bbox_results, segm_results) 或 仅 bbox_results（按类的 list[np.ndarray]）
    """
    from mmdet.apis import inference_detector

    result = inference_detector(model, img_bgr)

    # ====== 分支 A: MMDet 3.x (DetDataSample / List[DetDataSample]) ======
    try:
        # 惰性导入这些类型，避免在 2.x 环境下 ImportError
        from mmdet.structures import DetDataSample  # MMDet 3.x
        import torch

        def extract_from_datasample(ds):
            # ds: DetDataSample
            pred = getattr(ds, "pred_instances", None)
            if pred is None:
                # 有些模型可能用 pred_det_instances / pred_instances_xxx
                for k in ["pred_det_instances", "pred_instances"]:
                    pred = getattr(ds, k, None) or pred
            if pred is None:
                return None

            # bboxes, scores, labels 都是 InstanceData(Tensor)
            bboxes = getattr(pred, "bboxes", None)
            scores = getattr(pred, "scores", None)
            labels = getattr(pred, "labels", None)
            if bboxes is None or scores is None or labels is None:
                return None

            # 转 numpy
            if isinstance(bboxes, torch.Tensor):
                bboxes = bboxes.detach().cpu().numpy()
            if isinstance(scores, torch.Tensor):
                scores = scores.detach().cpu().numpy()
            if isinstance(labels, torch.Tensor):
                labels = labels.detach().cpu().numpy()

            if bboxes.size == 0:
                return {"bboxes": np.zeros((0, 5), dtype=np.float32),
                        "labels": np.zeros((0,), dtype=np.int32)}

            bboxes = bboxes.astype(np.float32)
            scores = scores.astype(np.float32).reshape(-1, 1)
            labels = labels.astype(np.int32)
            bboxes5 = np.concatenate([bboxes, scores], axis=1)  # [x1,y1,x2,y2,score]
            return {"bboxes": bboxes5, "labels": labels}

        # 情况 1：单张图 -> DetDataSample
        if isinstance(result, DetDataSample):
            out = extract_from_datasample(result)
            if out is not None:
                return out

        # 情况 2：可能返回 List[DetDataSample]（比如 TTA）
        if isinstance(result, (list, tuple)) and len(result) > 0 and isinstance(result[0], DetDataSample):
            # 这里简单地取第一个；若做 TTA 融合可在此加权合并
            out = extract_from_datasample(result[0])
            if out is not None:
                return out
    except Exception:
        # 如果不是 3.x 或类型导入失败，走 2.x 分支
        pass

    # ====== 分支 B: MMDet 2.x（老接口，list per class 或 (bbox, segm) tuple）======
    bboxes_list = []
    labels_list = []

    if isinstance(result, tuple):
        bbox_results = result[0]
    else:
        bbox_results = result

    # bbox_results: list of (n_i x 5) arrays per class
    if isinstance(bbox_results, (list, tuple)):
        for cls_idx, arr in enumerate(bbox_results):
            if arr is None or len(arr) == 0:
                continue
            bboxes_list.append(arr)
            labels_list.append(np.full((arr.shape[0],), cls_idx, dtype=np.int32))

        if len(bboxes_list) == 0:
            return {"bboxes": np.zeros((0, 5), dtype=np.float32),
                    "labels": np.zeros((0,), dtype=np.int32)}

        bboxes = np.concatenate(bboxes_list, axis=0).astype(np.float32)
        labels = np.concatenate(labels_list, axis=0).astype(np.int32)
        return {"bboxes": bboxes, "labels": labels}

    # 无法识别的返回类型
    raise TypeError(f"Unsupported inference result type: {type(result)}")


def compute_inter_min_iou(boxA: np.ndarray, boxB: np.ndarray) -> float:
    """
    IoU_min = intersection / min(areaA, areaB)
    box: [x1,y1,x2,y2]
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter_w = max(0.0, xB - xA)
    inter_h = max(0.0, yB - yA)
    inter = inter_w * inter_h
    if inter <= 0:
        return 0.0
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    denom = max(1e-9, min(areaA, areaB))
    return inter / denom

def nms_iou_min(dets: np.ndarray,
                scores: np.ndarray,
                labels: np.ndarray,
                iou_thr: float,
                class_agnostic: bool = True) -> np.ndarray:
    """
    dets: Nx4 (x1,y1,x2,y2)
    scores: Nx
    labels: Nx
    返回：保留索引的数组
    """
    if dets.shape[0] == 0:
        return np.array([], dtype=np.int64)

    if class_agnostic:
        order = scores.argsort()[::-1]
        keep = []
        suppressed = np.zeros(dets.shape[0], dtype=bool)
        for i in order:
            if suppressed[i]:
                continue
            keep.append(i)
            for j in order:
                if j <= i or suppressed[j]:
                    continue
                iou = compute_inter_min_iou(dets[i], dets[j])
                if iou >= iou_thr:
                    suppressed[j] = True
        return np.array(keep, dtype=np.int64)
    else:
        keep_all = []
        for cls in np.unique(labels):
            idxs = np.where(labels == cls)[0]
            o = idxs[np.argsort(scores[idxs])[::-1]]
            suppressed = np.zeros(o.shape[0], dtype=bool)
            keep_local = []
            for m in range(o.shape[0]):
                i = o[m]
                if suppressed[m]:
                    continue
                keep_local.append(i)
                for n in range(m + 1, o.shape[0]):
                    j = o[n]
                    if suppressed[n]:
                        continue
                    iou = compute_inter_min_iou(dets[i], dets[j])
                    if iou >= iou_thr:
                        suppressed[n] = True
            keep_all.extend(keep_local)
        return np.array(keep_all, dtype=np.int64)

def gen_windows(H: int, W: int, win: int, overlap: float) -> List[Tuple[int, int, int, int]]:
    """
    生成窗口列表: [(x1,y1,x2,y2), ...]
    """
    stride = int(round(win * (1.0 - overlap)))
    stride = max(1, min(win, stride))
    xs = list(range(0, max(1, W - win + 1), stride))
    ys = list(range(0, max(1, H - win + 1), stride))
    if len(xs) == 0:
        xs = [0]
    if len(ys) == 0:
        ys = [0]
    if xs[-1] + win < W:
        xs.append(W - win)
    if ys[-1] + win < H:
        ys.append(H - win)
    boxes = []
    for y in ys:
        for x in xs:
            x2 = min(x + win, W)
            y2 = min(y + win, H)
            boxes.append((x, y, x2, y2))
    return boxes

def crop_and_pad(img: np.ndarray, box: Tuple[int,int,int,int], win: int, pad_val: Tuple[int,int,int]) -> Tuple[np.ndarray, Tuple[int,int,int,int]]:
    """
    将 box 范围裁剪并右下补齐到 win x win（若不足），返回补齐后的图与有效区域 box_rel（相对于窗口坐标）
    """
    x1, y1, x2, y2 = box
    crop = img[y1:y2, x1:x2, :]
    h, w = crop.shape[:2]
    if h == win and w == win:
        return crop, (0, 0, w, h)
    canvas = np.zeros((win, win, 3), dtype=img.dtype)
    canvas[:, :, 0] = pad_val[0]
    canvas[:, :, 1] = pad_val[1]
    canvas[:, :, 2] = pad_val[2]
    canvas[0:h, 0:w, :] = crop
    return canvas, (0, 0, w, h)

def filter_to_inner_region(bboxes: np.ndarray, inner_box: Tuple[int,int,int,int]) -> np.ndarray:
    """
    根据预测框中心是否落在有效区域(inner_box)内进行过滤
    inner_box: (x1,y1,x2,y2) in window coords
    """
    if bboxes.shape[0] == 0:
        return np.array([], dtype=np.int64)
    x1, y1, x2, y2 = inner_box
    centers_x = 0.5 * (bboxes[:,0] + bboxes[:,2])
    centers_y = 0.5 * (bboxes[:,1] + bboxes[:,3])
    keep = np.where((centers_x >= x1) & (centers_x < x2) &
                    (centers_y >= y1) & (centers_y < y2))[0]
    return keep

def sliding_window_predict_single(model,
                                  img_bgr: np.ndarray,
                                  swc: SWConfig,
                                  nmsc: NMSConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    返回最终保留的 boxes, scores, labels （全图坐标）
    """
    H, W = img_bgr.shape[:2]
    windows = gen_windows(H, W, swc.window_size, swc.overlap_ratio)

    all_boxes = []
    all_scores = []
    all_labels = []

    for (x1, y1, x2, y2) in windows:
        crop, inner = crop_and_pad(img_bgr, (x1, y1, x2, y2), swc.window_size, swc.pad_value)
        pred = run_mmdet_inference(model, crop)

        b = pred["bboxes"].astype(np.float32)  # Nx5
        l = pred["labels"].astype(np.int32)

        if b.shape[0] == 0:
            continue

        # 分数阈值
        score_mask = b[:, 4] >= nmsc.score_thr
        b = b[score_mask]
        l = l[score_mask]
        if b.shape[0] == 0:
            continue

        # 过滤到有效区域（去掉补齐区域产生的框）
        if swc.filter_to_valid_region:
            keep = filter_to_inner_region(b[:, :4], inner)
            if keep.size == 0:
                continue
            b = b[keep]
            l = l[keep]

        # 平移到全图坐标
        b[:, 0] += x1
        b[:, 2] += x1
        b[:, 1] += y1
        b[:, 3] += y1

        all_boxes.append(b[:, :4])
        all_scores.append(b[:, 4])
        all_labels.append(l)

    if len(all_boxes) == 0:
        return (np.zeros((0, 4), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
                np.zeros((0,), dtype=np.int32))

    boxes = np.concatenate(all_boxes, axis=0)
    scores = np.concatenate(all_scores, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    keep = nms_iou_min(boxes, scores, labels, iou_thr=nmsc.nms_iou, class_agnostic=nmsc.class_agnostic)
    return boxes[keep], scores[keep], labels[keep]

# ========= 3) 数据集读取 & GT 计数 =========
def coco_load_annotations(ann_json_path: str) -> Tuple[Dict[int, int], Dict[str, int]]:
    """
    返回:
    - imgid_to_count: image_id -> 标注实例数量
    - filename_to_count: file_name -> 标注实例数量
    若文件不存在，返回空字典
    """
    if not os.path.isfile(ann_json_path):
        warnings.warn(f"COCO annotation not found: {ann_json_path}")
        return {}, {}

    with open(ann_json_path, "r") as f:
        data = json.load(f)

    # images: list of {id, file_name, ...}
    # annotations: list of {image_id, bbox/segmentation, ...}
    imgid_to_count = {}
    filename_to_count = {}

    id_to_filename = {}
    for img in data.get("images", []):
        img_id = img["id"]
        file_name = img["file_name"]
        id_to_filename[img_id] = file_name
        imgid_to_count[img_id] = 0

    for ann in data.get("annotations", []):
        img_id = ann["image_id"]
        if img_id in imgid_to_count:
            imgid_to_count[img_id] += 1

    for img_id, cnt in imgid_to_count.items():
        fn = id_to_filename.get(img_id, None)
        if fn is not None:
            filename_to_count[fn] = cnt

    return imgid_to_count, filename_to_count

def yolo_count_instances_for_image(label_txt_path: str) -> int:
    """
    YOLO txt 每行一个实例：cls cx cy w h [score...]
    直接统计行数；若文件不存在返回 0
    """
    if not os.path.isfile(label_txt_path):
        return 0
    with open(label_txt_path, "r") as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]
    return len(lines)

# ========= 4) 主流程：推理 & 统计 =========
def infer_coco_all(model, coco_root: str,
                    swc: SWConfig, nmsc: NMSConfig) -> List[Tuple[str, int, int]]:
    """
    在 coco_root 下推理 test2017、val2017、train2017 三个分区。
    GT 统计：优先读取各分区对应的 instances_{partition}.json（若无，则 ins_counts=0）
    """
    partitions = ["test2017", "val2017", "train2017"]
    results = []
    for part in partitions:
        part_dir = os.path.join(coco_root, part)
        images = list_images_in_dir(part_dir)

        # 找 annotations
        ann_json_candidates = [
            os.path.join(coco_root, "annotations", f"instances_{part}.json"),
            os.path.join(coco_root, "annotations", "instances_test.json"),
        ]
        fn_count_map = {}
        for p in ann_json_candidates:
            _, fn_map = coco_load_annotations(p)
            if fn_map:
                fn_count_map = fn_map
                break  # 用第一个存在的标注文件

        for img_path in tqdm(images, desc=f"COCO {part}"):
            img = cv2.imread(img_path)
            if img is None:
                warnings.warn(f"Failed to read image: {img_path}")
                continue
            boxes, scores, labels = sliding_window_predict_single(model, img, swc, nmsc)
            pred_counts = int(boxes.shape[0])

            # 绘制结果时可用：
            img_out = img.copy()
            for i in range(boxes.shape[0]):
                x1, y1, x2, y2 = boxes[i].astype(int)
                cv2.rectangle(img_out, (x1, y1), (x2, y2), (0, 255, 0), 2)
            vis_path = './vis' + os.path.basename(img_path)
            os.makedirs(os.path.dirname(vis_path), exist_ok=True)
            cv2.imwrite(vis_path, img_out)

            file_name = os.path.basename(img_path)
            ins_counts = int(fn_count_map.get(file_name, 0))
            results.append((img_path, ins_counts, pred_counts))
    return results


def infer_yolo_test(model, yolo_root: str,
                    swc: SWConfig, nmsc: NMSConfig) -> List[Tuple[str, int, int]]:
    """
    在 yolo_root/images/test 上推理。
    GT 统计来自 yolo_root/labels/test 中同名 .txt
    """
    img_dir = os.path.join(yolo_root, "images", "test")
    lbl_dir = os.path.join(yolo_root, "labels", "test")
    images = list_images_in_dir(img_dir)

    results = []
    for img_path in tqdm(images, desc="YOLO test"):
        img = cv2.imread(img_path)
        if img is None:
            warnings.warn(f"Failed to read image: {img_path}")
            continue
        boxes, scores, labels = sliding_window_predict_single(model, img, swc, nmsc)
        pred_counts = int(boxes.shape[0])

        # 绘制结果时可用：
        img_out = img.copy()
        for i in range(boxes.shape[0]):
            x1, y1, x2, y2 = boxes[i].astype(int)
            cv2.rectangle(img_out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        vis_path = './vis' + os.path.basename(img_path)
        os.makedirs(os.path.dirname(vis_path), exist_ok=True)
        cv2.imwrite(vis_path, img_out)

        stem = os.path.splitext(os.path.basename(img_path))[0]
        label_txt = os.path.join(lbl_dir, f"{stem}.txt")
        ins_counts = yolo_count_instances_for_image(label_txt)
        results.append((img_path, ins_counts, pred_counts))
    return results

def write_csv(rows: List[Tuple[str, int, int]], out_csv: str):
    import csv
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["filename", "ins_counts", "pred_counts"])
        for fn, ins_cnt, pred_cnt in rows:
            w.writerow([fn, ins_cnt, pred_cnt])


# ========= 5) 入口 =========
def main():
    # 抑制一些不必要的 Warning
    warnings.filterwarnings("once")

    # 1) 加载模型
    print("Loading MMDetection model ...")
    model = load_mmdet_model(MODEL_CFG)

    # 2) 推理 COCO test2017
    coco_rows = infer_coco_all(model, DATASET_COCO_DIR, SW_CFG, NMS_CFG)

    # 3) 推理 YOLO test
    # yolo_rows = infer_yolo_test(model, DATASET_YOLO_DIR, SW_CFG, NMS_CFG)
    yolo_rows = []

    # 4) 汇总并输出 CSV
    all_rows = coco_rows + yolo_rows
    write_csv(all_rows, OUTPUT_CSV)
    print(f"Done. Wrote {OUTPUT_CSV} with {len(all_rows)} rows.")


if __name__ == "__main__":
    main()
