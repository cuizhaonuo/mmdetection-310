#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Wheat instance segmentation on MMDetection with sliding-window + global Mask-NMS.
Modes:
- search: grid search (iou_type x score_thr x nms_iou_thr) on val then test with best
- infer : skip search, directly use (iou_type, score_thr, nms_iou_thr) on a split (default: test)

Template Method pattern:
- SegInferenceTemplate: defines the workflow skeleton (run_search / run_infer_only)
- MmdetSegInference: concrete implementation using MMDetection
"""

import os
import sys
import math
import glob
import argparse
import random
from enum import Enum
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any

import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from pycocotools import mask as coco_mask

# ====== MMDetection ======
from mmdet.apis import init_detector, inference_detector


# -----------------------------
# Config dataclasses
# -----------------------------
@dataclass
class DatasetConfig:
    yolo_root: str

@dataclass
class ModelConfig:
    project_root: str
    config_rel: str
    checkpoint_rel: str
    device: str = "cuda:0"

@dataclass
class SlidingWindowConfig:
    tile: int = 800
    overlap: int = 256
    pad_value: Tuple[int, int, int] = (114, 114, 114)  # BGR

@dataclass
class SearchSpace:
    iou_types: Tuple[str, ...] = ("IOU", "IOU_Min")
    score_thr_range: Tuple[float, float, int] = (0.1, 0.8, 30)
    nms_iou_thr_range: Tuple[float, float, int] = (0.1, 0.8, 30)
    base_keep_score: float = 0.01

@dataclass
class OutputOptions:
    vis: bool = True
    excel: bool = True
    draw_scatter: bool = True
    out_dir: str = "./vis_wheat_seg"


# -----------------------------
# IO helpers
# -----------------------------
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def list_images_in_dir(d: str) -> List[str]:
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff")
    files = []
    for e in exts:
        files.extend(glob.glob(os.path.join(d, e)))
    files.sort()
    return files

def yolo_split_paths(yolo_root: str, split: str) -> Tuple[str, str]:
    return os.path.join(yolo_root, "images", split), os.path.join(yolo_root, "labels", split)

def yolo_label_path_for_image(img_path: str, yolo_root: str) -> str:
    rel = os.path.relpath(img_path, os.path.join(yolo_root, "images"))
    split = rel.split(os.sep)[0]
    name = os.path.splitext(os.path.basename(img_path))[0]
    return os.path.join(yolo_root, "labels", split, f"{name}.txt")

def yolo_count_instances(label_path: str) -> int:
    if not os.path.exists(label_path):
        return 0
    c = 0
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            c += 1
    return c

def linspace_inclusive(start: float, end: float, num: int) -> List[float]:
    if num <= 1:
        return [start]
    step = (end - start) / (num - 1)
    return [start + i * step for i in range(num)]


# -----------------------------
# Proposal & IoU / NMS
# -----------------------------
@dataclass
class DetProposal:
    bbox: np.ndarray
    score: float
    label: int
    mask: np.ndarray    # cropped to bbox (h,w) bool
    top: int            # global y of bbox top
    left: int           # global x of bbox left
    area: int           # mask.sum()

class IOUType(Enum):
    IOU = "IOU"
    IOU_Min = "IOU_Min"

def mask_iou_between(a: DetProposal, b: DetProposal, iou_type: IOUType) -> float:
    xA = max(a.left, b.left); yA = max(a.top, b.top)
    xB = min(int(round(a.left + a.mask.shape[1])), int(round(b.left + b.mask.shape[1])))
    yB = min(int(round(a.top + a.mask.shape[0])), int(round(b.top + b.mask.shape[0])))
    if xB <= xA or yB <= yA:
        return 0.0
    a_roi = a.mask[yA - a.top:yB - a.top, xA - a.left:xB - a.left]
    b_roi = b.mask[yA - b.top:yB - b.top, xA - b.left:xB - b.left]
    inter = np.logical_and(a_roi, b_roi).sum()
    if inter <= 0:
        return 0.0
    if iou_type == IOUType.IOU:
        union = a.area + b.area - inter
        return inter / union if union > 0 else 0.0
    else:
        return inter / min(a.area, b.area) if min(a.area, b.area) > 0 else 0.0

def mask_nms(proposals: List[DetProposal], iou_thr: float, iou_type: IOUType) -> List[DetProposal]:
    if not proposals:
        return []
    order = sorted(range(len(proposals)), key=lambda i: proposals[i].score, reverse=True)
    keep: List[int] = []
    suppressed = np.zeros(len(proposals), dtype=bool)
    for i_idx in order:
        if suppressed[i_idx]:
            continue
        keep.append(i_idx)
        for j_idx in order:
            if suppressed[j_idx] or j_idx == i_idx:
                continue
            if mask_iou_between(proposals[i_idx], proposals[j_idx], iou_type) >= iou_thr:
                suppressed[j_idx] = True
    return [proposals[i] for i in keep]


# -----------------------------
# Template base
# -----------------------------
class SegInferenceTemplate:
    def __init__(self, dataset: DatasetConfig, model: ModelConfig, sw: SlidingWindowConfig,
                 search: SearchSpace, outputs: OutputOptions):
        self.dataset = dataset
        self.model_cfg = model
        self.sw = sw
        self.search = search
        self.outputs = outputs
        self.model = None

    # hooks
    def load_model(self): raise NotImplementedError
    def infer_tile(self, tile_img: np.ndarray) -> List[Dict[str, Any]]: raise NotImplementedError

    # ====== New: two modes ======
    def run_search(self):
        """Grid-search on val, then test with best params."""
        ensure_dir(self.outputs.out_dir)
        random.seed(0); np.random.seed(0)
        print("==> Loading model ..."); self.load_model()

        print("==> Collecting raw proposals on VAL ...")
        val_imgs_dir, _ = yolo_split_paths(self.dataset.yolo_root, "val")
        val_images = list_images_in_dir(val_imgs_dir)
        val_cache, val_gt_counts = {}, {}
        for img_path in tqdm(val_images, desc="VAL images"):
            val_cache[img_path] = self._proposals_for_image(img_path)
            val_gt_counts[img_path] = yolo_count_instances(yolo_label_path_for_image(img_path, self.dataset.yolo_root))

        print("==> Grid searching best parameters on VAL ...")
        best_cfg, best_sum_diff = self._grid_search(val_cache, val_gt_counts)
        print(f"[Best on VAL] iou_type={best_cfg['iou_type'].value}, "
              f"score_thr={best_cfg['score_thr']:.4f}, nms_iou_thr={best_cfg['nms_iou_thr']:.4f}, "
              f"sum|GT-Pred|={best_sum_diff}")

        print("==> Running TEST with best parameters ...")
        self._run_on_split("test", best_cfg['iou_type'], best_cfg['score_thr'], best_cfg['nms_iou_thr'])

    def run_infer_only(self, split: str, iou_type: IOUType, score_thr: float, nms_iou_thr: float):
        """Direct inference with fixed params on a chosen split (test/val/train)."""
        ensure_dir(self.outputs.out_dir)
        random.seed(0); np.random.seed(0)
        print("==> Loading model ..."); self.load_model()
        print(f"==> Running {split.upper()} with fixed params: "
              f"iou_type={iou_type.value}, score_thr={score_thr}, nms_iou_thr={nms_iou_thr}")
        self._run_on_split(split, iou_type, score_thr, nms_iou_thr)

    # ====== shared helpers ======
    def _run_on_split(self, split: str, iou_type: IOUType, score_thr: float, nms_iou_thr: float):
        imgs_dir, _ = yolo_split_paths(self.dataset.yolo_root, split)
        images = list_images_in_dir(imgs_dir)

        rows = [("filename", "ins_counts", "pred_counts")]
        xs, ys = [], []
        for img_path in tqdm(images, desc=f"{split.upper()} images"):
            proposals = self._proposals_for_image(img_path)
            final_dets = self._apply_thresh_nms(proposals, score_thr, nms_iou_thr, iou_type)
            pred_count = len(final_dets)
            gt_count = yolo_count_instances(yolo_label_path_for_image(img_path, self.dataset.yolo_root))
            rows.append((os.path.basename(img_path), gt_count, pred_count))
            xs.append(gt_count); ys.append(pred_count)
            if self.outputs.vis:
                self._visualize_and_save(img_path, final_dets, self.outputs.out_dir)

        if self.outputs.excel:
            csv_path = os.path.join(self.outputs.out_dir, f"{split}_pred_vs_gt.csv")
            with open(csv_path, "w", encoding="utf-8") as f:
                for r in rows: f.write(f"{r[0]},{r[1]},{r[2]}\n")
            print(f"[Saved] CSV -> {csv_path}")

        if self.outputs.draw_scatter and xs:
            self._draw_scatter(xs, ys, os.path.join(self.outputs.out_dir, f"scatter_{split}.png"))
            print(f"[Saved] Scatter -> {os.path.join(self.outputs.out_dir, f'scatter_{split}.png')}")

        print(f"{split.upper()} finished.")

    def _proposals_for_image(self, img_path: str) -> List[DetProposal]:
        img = cv2.imread(img_path)
        if img is None: raise FileNotFoundError(img_path)
        H, W = img.shape[:2]
        tile, overlap = self.sw.tile, self.sw.overlap
        stride = max(1, tile - overlap); padv = self.sw.pad_value

        proposals: List[DetProposal] = []

        y_list = list(range(0, max(1, H - tile + 1), stride))
        if y_list[-1] != max(0, H - tile): y_list.append(max(0, H - tile))
        x_list = list(range(0, max(1, W - tile + 1), stride))
        if x_list[-1] != max(0, W - tile): x_list.append(max(0, W - tile))

        for y0 in y_list:
            for x0 in x_list:
                tile_img = np.full((tile, tile, 3), padv, dtype=np.uint8)
                y1, x1 = min(y0 + tile, H), min(x0 + tile, W)
                h, w = y1 - y0, x1 - x0
                tile_img[:h, :w] = img[y0:y1, x0:x1]
                valid_mask = np.zeros((tile, tile), dtype=bool); valid_mask[:h, :w] = True

                insts = self.infer_tile(tile_img)
                for ins in insts:
                    score = float(ins["score"])
                    if score < self.search.base_keep_score:
                        continue
                    bbox = np.array(ins["bbox"], dtype=float)
                    mask = np.logical_and(ins["mask"].astype(bool), valid_mask)

                    x1t = max(0, min(tile, int(math.floor(bbox[0]))))
                    y1t = max(0, min(tile, int(math.floor(bbox[1]))))
                    x2t = max(0, min(tile, int(math.ceil(bbox[2]))))
                    y2t = max(0, min(tile, int(math.ceil(bbox[3]))))
                    if x2t <= x1t or y2t <= y1t:
                        continue

                    mask_crop = mask[y1t:y2t, x1t:x2t]
                    area = int(mask_crop.sum())
                    if area <= 0:
                        continue

                    gx1, gy1 = x0 + x1t, y0 + y1t
                    gx2, gy2 = x0 + x2t, y0 + y2t
                    gx1 = max(0, min(W, gx1)); gx2 = max(0, min(W, gx2))
                    gy1 = max(0, min(H, gy1)); gy2 = max(0, min(H, gy2))
                    if gx2 <= gx1 or gy2 <= gy1:
                        continue

                    proposals.append(DetProposal(
                        bbox=np.array([gx1, gy1, gx2, gy2], dtype=float),
                        score=score,
                        label=int(ins.get("label", 0)),
                        mask=mask_crop,
                        top=int(gy1),
                        left=int(gx1),
                        area=area
                    ))
        return proposals

    def _grid_search(self, val_cache: Dict[str, List[DetProposal]],
                     val_gt_counts: Dict[str, int]) -> Tuple[Dict[str, Any], int]:
        score_list = linspace_inclusive(*self.search.score_thr_range)
        nms_list = linspace_inclusive(*self.search.nms_iou_thr_range)
        iou_types = [IOUType(t) for t in self.search.iou_types]
        best_cfg, best_sum = None, 1 << 62
        total = len(score_list) * len(nms_list) * len(iou_types)
        pbar = tqdm(total=total, desc="Grid search")
        for iout in iou_types:
            for s_thr in score_list:
                for n_thr in nms_list:
                    sum_diff = 0
                    for img_path, props in val_cache.items():
                        pred = len(self._apply_thresh_nms(props, s_thr, n_thr, iout))
                        sum_diff += abs(val_gt_counts[img_path] - pred)
                    if sum_diff < best_sum:
                        best_sum = sum_diff
                        best_cfg = {"iou_type": iout, "score_thr": s_thr, "nms_iou_thr": n_thr}
                    pbar.update(1)
        pbar.close()
        return best_cfg, best_sum

    def _apply_thresh_nms(self, proposals: List[DetProposal],
                          score_thr: float, nms_thr: float, iou_type: IOUType) -> List[DetProposal]:
        fil = [p for p in proposals if p.score >= score_thr]
        return mask_nms(fil, nms_thr, iou_type) if fil else []

    def _visualize_and_save(self, img_path: str, dets: List[DetProposal],
                            out_dir: str, alpha: float = 0.8, draw_contour: bool = False):
        img = cv2.imread(img_path)
        if img is None: return
        overlay = img.copy()
        rng = np.random.default_rng(12345)
        for d in dets:
            color = rng.integers(0, 255, size=3, dtype=np.uint8).tolist()
            y1, y2 = d.top, int(d.top + d.mask.shape[0])
            x1, x2 = d.left, int(d.left + d.mask.shape[1])
            roi = overlay[y1:y2, x1:x2]
            m = d.mask
            roi[m] = (0.6 * roi[m] + 0.4 * np.array(color, dtype=np.uint8)).astype(np.uint8)
            if draw_contour:
                m8 = (m.astype(np.uint8) * 255)
                contours, _ = cv2.findContours(m8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(img[y1:y2, x1:x2], contours, -1, color, 2)
        vis = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
        ensure_dir(out_dir)
        cv2.imwrite(os.path.join(out_dir, os.path.basename(img_path)), vis)

    def _draw_scatter(self, xs: List[int], ys: List[int], out_path: str):
        plt.figure(figsize=(6, 6))
        plt.scatter(xs, ys, s=10)
        mn, mx = min(xs + ys), max(xs + ys)
        plt.plot([mn, mx], [mn, mx], linestyle="--")
        plt.xlabel("Ground Truth Count"); plt.ylabel("Predicted Count")
        plt.title("GT vs Pred")
        plt.tight_layout(); plt.savefig(out_path, dpi=160); plt.close()


# -----------------------------
# MMDetection impl
# -----------------------------
class MmdetSegInference(SegInferenceTemplate):
    def load_model(self):
        cfg_path = os.path.join(self.model_cfg.project_root, self.model_cfg.config_rel)
        ckpt_path = os.path.join(self.model_cfg.project_root, self.model_cfg.checkpoint_rel)
        self.model = init_detector(cfg_path, ckpt_path, device=self.model_cfg.device)

    def infer_tile(self, tile_img: np.ndarray) -> List[Dict[str, Any]]:
        result = inference_detector(self.model, tile_img)

        # MMDet 3.x: DetDataSample
        if hasattr(result, "pred_instances") or (isinstance(result, list) and len(result) > 0 and hasattr(result[0], "pred_instances")):
            data_sample = result[0] if isinstance(result, list) else result
            inst = data_sample.pred_instances
            if inst is None or len(inst) == 0:
                return []
            bboxes = inst.bboxes.cpu().numpy() if hasattr(inst.bboxes, "cpu") else inst.bboxes.numpy()
            scores = inst.scores.cpu().numpy() if hasattr(inst.scores, "cpu") else inst.scores.numpy()
            labels = inst.labels.cpu().numpy() if hasattr(inst.labels, "cpu") else inst.labels.numpy()
            if hasattr(inst.masks, "to_ndarray"):
                masks = inst.masks.to_ndarray()
            else:
                masks = inst.masks
                if hasattr(masks, "cpu"):
                    masks = masks.cpu().numpy()
            masks = masks.astype(bool)
            out = []
            for i in range(bboxes.shape[0]):
                x1, y1, x2, y2 = bboxes[i].tolist()
                out.append({"bbox": [x1, y1, x2, y2],
                            "score": float(scores[i]),
                            "label": int(labels[i]),
                            "mask": masks[i]})
            return out

        # MMDet 2.x: (bbox_results, segm_results)
        if isinstance(result, (list, tuple)) and len(result) == 2:
            bbox_results, segm_results = result
            out = []
            for cls_idx, (bcls, scls) in enumerate(zip(bbox_results, segm_results)):
                if len(bcls) == 0:
                    continue
                for k in range(len(bcls)):
                    x1, y1, x2, y2, score = bcls[k].tolist()
                    m = scls[k]
                    if isinstance(m, dict) or (isinstance(m, list) and isinstance(m[0], dict)):
                        mask = coco_mask.decode(m).astype(bool)
                    else:
                        mask = np.array(m).astype(bool)
                    out.append({"bbox": [x1, y1, x2, y2],
                                "score": float(score),
                                "label": int(cls_idx),
                                "mask": mask})
            return out

        return []


# -----------------------------
# CLI
# -----------------------------
def build_argparser():
    p = argparse.ArgumentParser(description="Wheat Segmentation (Template Method) with search/infer modes")

    # Modes
    p.add_argument("--mode", choices=["search", "infer"], default="infer",
                   help="search: grid search on val then test; infer: direct inference with fixed params")

    # Dataset
    # /public/home/zncui/datasets/wheat-guo-dataset/yolo_wheat_big_image
    # /public/home/zncui/pycharm_project/mmdetection/data/yolo
    p.add_argument("--yolo_root", type=str,
                   default="/public/home/zncui/pycharm_project/mmdetection/data/yolo")

    # Model
    p.add_argument("--project_root", type=str,
                   default="/public/home/zncui/pycharm_project/mmdetection/")
    p.add_argument("--model_config_rel", type=str,
                   default="work_dirs/mask-rcnn_x101-64x4d_fpn_ms-poly_3x_coco/mask-rcnn_x101-64x4d_fpn_ms-poly_3x_coco (副本).py")
    p.add_argument("--checkpoint_rel", type=str,
                   default="work_dirs/mask-rcnn_x101-64x4d_fpn_ms-poly_3x_coco/epoch_12.pth")
    p.add_argument("--device", type=str, default="cuda:0")

    # Sliding window
    p.add_argument("--tile", type=int, default=1024)
    p.add_argument("--overlap", type=int, default=256)

    # Search space (used only in --mode search)
    p.add_argument("--score_start", type=float, default=0.1)
    p.add_argument("--score_end", type=float, default=0.8)
    p.add_argument("--score_num", type=int, default=30)
    p.add_argument("--nms_start", type=float, default=0.1)
    p.add_argument("--nms_end", type=float, default=0.8)
    p.add_argument("--nms_num", type=int, default=30)
    p.add_argument("--iou_types", type=str, nargs="+", default=["IOU", "IOU_Min"])
    p.add_argument("--base_keep_score", type=float, default=0.01)

    # Infer-only fixed params
    p.add_argument("--infer_split", choices=["test", "val", "train"], default="test")
    p.add_argument("--fix_iou_type", choices=["IOU", "IOU_Min"], default="IOU")
    p.add_argument("--fix_score_thr", type=float, default=0.1)
    p.add_argument("--fix_nms_iou_thr", type=float, default=0.1)

    # Outputs
    p.add_argument("--out_dir", type=str, default="./vis_wheat_seg_iou_0.1_0.1_1024")
    p.add_argument("--no_vis", action="store_true")
    p.add_argument("--no_excel", action="store_true")
    p.add_argument("--no_scatter", action="store_true")

    return p

def main():
    args = build_argparser().parse_args()

    dataset = DatasetConfig(yolo_root=args.yolo_root)
    model = ModelConfig(
        project_root=args.project_root,
        config_rel=args.model_config_rel,
        checkpoint_rel=args.checkpoint_rel,
        device=args.device
    )
    sw = SlidingWindowConfig(tile=args.tile, overlap=args.overlap, pad_value=(114, 114, 114))
    search = SearchSpace(
        iou_types=tuple(args.iou_types),
        score_thr_range=(args.score_start, args.score_end, args.score_num),
        nms_iou_thr_range=(args.nms_start, args.nms_end, args.nms_num),
        base_keep_score=args.base_keep_score
    )
    outputs = OutputOptions(
        vis=not args.no_vis,
        excel=not args.no_excel,
        draw_scatter=not args.no_scatter,
        out_dir=args.out_dir
    )

    runner = MmdetSegInference(dataset, model, sw, search, outputs)

    if args.mode == "search":
        runner.run_search()
    else:
        runner.run_infer_only(
            split=args.infer_split,
            iou_type=IOUType(args.fix_iou_type),
            score_thr=args.fix_score_thr,
            nms_iou_thr=args.fix_nms_iou_thr
        )

if __name__ == "__main__":
    main()
