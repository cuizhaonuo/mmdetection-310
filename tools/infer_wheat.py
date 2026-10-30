#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, csv, math, time, json, shutil, logging
from dataclasses import dataclass
from typing import List, Tuple, Iterable, Optional, Dict, Any

import pandas as pd
import numpy as np
import torch, mmcv, cv2
from tqdm import tqdm

from mmengine.structures import InstanceData
from mmdet.structures import DetDataSample
from mmdet.apis import init_detector, inference_detector
# from mmcv.ops import soft_nms as mmcv_soft_nms, nms as mmcv_hard_nms  # 如需官方 CUDA NMS

# --------------------------
# Logging
# --------------------------
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s', datefmt='%H:%M:%S'
)
logger = logging.getLogger('wheat-pipeline')
torch.backends.cudnn.benchmark = True


# =============================================================================
# Config dataclasses
# =============================================================================
@dataclass
class ModelConfig:
    base_dir: str
    config_rel: str
    checkpoint_rel: str
    device: str = "cuda:0"

    @property
    def config(self) -> str:
        return os.path.join(self.base_dir, self.config_rel)

    @property
    def checkpoint(self) -> str:
        return os.path.join(self.base_dir, self.checkpoint_rel)


@dataclass
class DataConfig:
    dataset_dir: str
    input_rel: str
    output_rel: str
    image_exts: Tuple[str, ...] = (".jpg", ".jpeg", ".png")

    def __post_init__(self):
        if isinstance(self.image_exts, list):
            self.image_exts = tuple(self.image_exts)

    @property
    def input_dir(self) -> str:
        p = os.path.join(self.dataset_dir, self.input_rel)
        os.makedirs(p, exist_ok=True)
        return p

    @property
    def output_dir(self) -> str:
        p = os.path.join(self.dataset_dir, self.output_rel)
        os.makedirs(p, exist_ok=True)
        return p


@dataclass
class SWParams:
    window_size: int = 1024
    overlap_ratio: float = 0.15
    pad_value: Tuple[int, int, int] = (114, 114, 114)
    filter_to_valid_region: bool = True


@dataclass
class NMSParams:
    score_thr: float = 0.3
    nms_iou: float = 0.5
    iou_type: str = "IoU"  # IoU | IoU_min | GIoU | DIoU | CIoU | EIoU
    use_soft_nms: bool = False
    soft_sigma: float = 0.5
    soft_score_thr: float = 0.001


@dataclass
class RunSave:
    ori: bool = True
    txt: bool = True
    rect_vis: bool = False
    circle_vis: bool = False
    index_vis: bool = False
    normalized_result: bool = False

@dataclass
class RunMetrics:
    calc_prf: bool = False
    label_root_rel: str = "yolo_wheat_big_image_v2split/labels/val"


@dataclass
class GridParams:
    nms_range: Tuple[float, float, int] = (0.2, 0.6, 10)   # start, stop, num
    score_range: Tuple[float, float, int] = (0.2, 0.6, 8)
    iou_types: Tuple[str, ...] = ("IoU_min",)


@dataclass
class RunConfig:
    mode: str = "run"  # run | grid
    save: RunSave = RunSave()
    metrics: RunMetrics = RunMetrics()
    grid: GridParams = GridParams()


@dataclass
class FullConfig:
    model: ModelConfig
    data: DataConfig
    sliding_window: SWParams
    nms: NMSParams
    run: RunConfig


# =============================================================================
# Config loader (yaml/json/toml)
# =============================================================================
def load_config(path_or_dict: Any) -> FullConfig:
    """
    支持三种方式：
    - str: 指向 .yaml/.yml/.json/.toml 文件
    - dict: 直接给 dict
    - dataclass: 已实例化的 FullConfig（直接返回）
    """
    if isinstance(path_or_dict, FullConfig):
        return path_or_dict

    if isinstance(path_or_dict, dict):
        cfg_dict = path_or_dict
    elif isinstance(path_or_dict, str):
        ext = os.path.splitext(path_or_dict)[1].lower()
        if ext in (".yaml", ".yml"):
            try:
                import yaml
            except ImportError:
                raise RuntimeError("需要 PyYAML。安装：pip install pyyaml")
            with open(path_or_dict, "r", encoding="utf-8") as f:
                cfg_dict = yaml.safe_load(f)
        elif ext == ".json":
            with open(path_or_dict, "r", encoding="utf-8") as f:
                cfg_dict = json.load(f)
        elif ext == ".toml":
            try:
                import tomllib  # py>=3.11
            except Exception:
                try:
                    import tomli as tomllib  # py<3.11
                except ImportError:
                    raise RuntimeError("需要 tomli/tomllib。安装：pip install tomli")
            with open(path_or_dict, "rb") as f:
                cfg_dict = tomllib.load(f)
        else:
            raise ValueError(f"不支持的配置扩展名: {ext}")
    else:
        raise TypeError("load_config 仅接受 FullConfig、dict 或 str 路径")

    # ---- 构造成 dataclass ----
    def _get(key, default=None, cast=dict):
        v = cfg_dict.get(key, None)
        if v is None:
            return cast() if default is None else default
        return v

    model = ModelConfig(**cfg_dict["model"])
    data = DataConfig(**cfg_dict["data"])
    sw = SWParams(**cfg_dict.get("sliding_window", {}))
    nms = NMSParams(**cfg_dict.get("nms", {}))

    save = RunSave(**_get("run", {}).get("save", {}))
    metrics = RunMetrics(**_get("run", {}).get("metrics", {}))
    grid = GridParams(**_get("run", {}).get("grid", {}))
    run = RunConfig(
        mode=_get("run", {}).get("mode", "run"),
        save=save, metrics=metrics, grid=grid
    )
    return FullConfig(model=model, data=data, sliding_window=sw, nms=nms, run=run)


# =============================================================================
# IoU / NMS 策略（与你原实现等价）
# =============================================================================
def _custom_soft_nms(bboxes: torch.Tensor,
                     scores: torch.Tensor,
                     iou_threshold: float = 0.3,
                     sigma: float = 0.5,
                     score_threshold: float = 0.001) -> torch.Tensor:
    if bboxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64)
    x1, y1, x2, y2 = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    _, order = scores.sort(descending=True)
    keep = []
    while order.numel() > 0:
        i = order[0].item()
        keep.append(i)
        if order.numel() == 1:
            break
        idxs = order[1:]
        xx1 = torch.maximum(x1[i], x1[idxs])
        yy1 = torch.maximum(y1[i], y1[idxs])
        xx2 = torch.minimum(x2[i], x2[idxs])
        yy2 = torch.minimum(y2[i], y2[idxs])
        w = torch.clamp(xx2 - xx1 + 1, min=0)
        h = torch.clamp(yy2 - yy1 + 1, min=0)
        inter = w * h
        iou = inter / (areas[i] + areas[idxs] - inter + 1e-7)
        weights = torch.exp(-(iou ** 2) / sigma)
        scores[idxs] *= weights
        remain = (scores[idxs] > score_threshold).nonzero(as_tuple=False).squeeze()
        order = idxs[remain] if remain.numel() > 0 else torch.empty(0, dtype=torch.long)
    return torch.tensor(keep, dtype=torch.int64)


def _custom_hard_nms(bboxes: torch.Tensor,
                     scores: torch.Tensor,
                     iou_threshold: float,
                     iou_type: str = "IoU") -> torch.Tensor:
    if bboxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=bboxes.device)
    x1, y1, x2, y2 = bboxes.unbind(dim=1)
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    _, order = scores.sort(descending=True)

    keep = []
    eps = 1e-7
    while order.numel() > 0:
        i = order[0].item()
        keep.append(i)
        if order.numel() == 1:
            break
        idxs = order[1:]
        xx1 = torch.maximum(x1[i], x1[idxs])
        yy1 = torch.maximum(y1[i], y1[idxs])
        xx2 = torch.minimum(x2[i], x2[idxs])
        yy2 = torch.minimum(y2[i], y2[idxs])
        w = (xx2 - xx1 + 1).clamp(min=0)
        h = (yy2 - yy1 + 1).clamp(min=0)
        inter = w * h
        union = areas[i] + areas[idxs] - inter
        base_iou = inter / union.clamp(min=eps)

        if iou_type == 'IoU':
            ious = base_iou
        elif iou_type == 'IoU_min':
            ious = inter / torch.min(areas[i], areas[idxs]).clamp(min=eps)
        elif iou_type == 'GIoU':
            cx1 = torch.minimum(x1[i], x1[idxs])
            cy1 = torch.minimum(y1[i], y1[idxs])
            cx2 = torch.maximum(x2[i], x2[idxs])
            cy2 = torch.maximum(y2[i], y2[idxs])
            cw = (cx2 - cx1 + 1).clamp(min=0)
            ch = (cy2 - cy1 + 1).clamp(min=0)
            c_area = cw * ch
            ious = base_iou - (c_area - union) / c_area.clamp(min=eps)
        elif iou_type in ('DIoU', 'CIoU', 'EIoU'):
            bxi = (x1[i] + x2[i]) / 2; byi = (y1[i] + y2[i]) / 2
            bxj = (x1[idxs] + x2[idxs]) / 2; byj = (y1[idxs] + y2[idxs]) / 2
            dist_sq = (bxi - bxj) ** 2 + (byi - byj) ** 2
            ex1 = torch.minimum(x1[i], x1[idxs]); ey1 = torch.minimum(y1[i], y1[idxs])
            ex2 = torch.maximum(x2[i], x2[idxs]); ey2 = torch.maximum(y2[i], y2[idxs])
            diag_sq = ((ex2 - ex1) ** 2 + (ey2 - ey1) ** 2).clamp(min=eps)
            if iou_type == 'DIoU':
                ious = base_iou - dist_sq / diag_sq
            elif iou_type == 'CIoU':
                wi = x2[i] - x1[i] + 1; hi = y2[i] - y1[i] + 1
                wj = x2[idxs] - x1[idxs] + 1; hj = y2[idxs] - y1[idxs] + 1
                v = (4 / (math.pi ** 2)) * (torch.atan(wj / (hj + eps)) - torch.atan(wi / (hi + eps))) ** 2
                alpha = v / ((1 - base_iou) + v + eps)
                ious = base_iou - (dist_sq / diag_sq + alpha * v)
            else:  # EIoU
                cw = (torch.maximum(x2[i], x2[idxs]) - torch.minimum(x1[i], x1[idxs]) + 1).clamp(min=eps)
                ch = (torch.maximum(y2[i], y2[idxs]) - torch.minimum(y1[i], y1[idxs]) + 1).clamp(min=eps)
                wi = x2[i] - x1[i] + 1; hi = y2[i] - y1[i] + 1
                wj = x2[idxs] - x1[idxs] + 1; hj = y2[idxs] - y1[idxs] + 1
                w_diff = (wj - wi) ** 2; h_diff = (hj - hi) ** 2
                ious = base_iou - (dist_sq/diag_sq + w_diff/(cw**2) + h_diff/(ch**2))
        else:
            raise ValueError(f"Unknown iou_type: {iou_type}")

        mask = ious <= iou_threshold
        order = idxs[mask]

    return torch.as_tensor(keep, dtype=torch.int64, device=bboxes.device)


def batched_nms(bboxes: torch.Tensor,
                scores: torch.Tensor,
                labels: torch.Tensor,
                p: NMSParams) -> np.ndarray:
    keep_indices: List[int] = []
    unique_labels = labels.unique()
    for cls_id in unique_labels:
        cls_mask = (labels == cls_id)
        cls_bboxes = bboxes[cls_mask]
        cls_scores = scores[cls_mask]
        if cls_bboxes.shape[0] == 0:
            continue
        if p.use_soft_nms:
            inds = _custom_soft_nms(cls_bboxes.clone(), cls_scores.clone(),
                                    iou_threshold=p.nms_iou,
                                    sigma=p.soft_sigma,
                                    score_threshold=p.soft_score_thr)
        else:
            inds = _custom_hard_nms(cls_bboxes, cls_scores,
                                    iou_threshold=p.nms_iou,
                                    iou_type=p.iou_type)
        mask_indices = cls_mask.nonzero(as_tuple=False)[:, 0]
        keep_indices.extend(mask_indices[inds].cpu().numpy())

    order = np.argsort(-scores.cpu().numpy()[keep_indices])
    return np.array(keep_indices)[order]


# =============================================================================
# 核心组件：Detector / Tiler / Exporter / Evaluator / Runner
# =============================================================================
class Detector:
    def __init__(self, cfg: ModelConfig):
        self.model = init_detector(cfg.config, cfg.checkpoint, device=cfg.device)
        self._meta = getattr(self.model, "dataset_meta", None)
        self._vis = None

    @torch.no_grad()
    def infer_crop(self, crop: np.ndarray) -> DetDataSample:
        return inference_detector(self.model, crop)

    def get_visualizer(self):
        if self._vis is None:
            from mmdet.visualization import DetLocalVisualizer
            self._vis = DetLocalVisualizer()
            self._vis.dataset_meta = self._meta
        return self._vis


class Tiler:
    def __init__(self, sw: SWParams):
        self.sw = sw

    def _tiles(self, h: int, w: int) -> Iterable[Tuple[int, int, int, int]]:
        stride = int(self.sw.window_size * (1 - self.sw.overlap_ratio))
        for y in range(0, h, stride):
            for x in range(0, w, stride):
                yield y, min(y + self.sw.window_size, h), x, min(x + self.sw.window_size, w)

    def infer_image(self, detector: Detector, img: np.ndarray, nms_params: NMSParams) -> DetDataSample:
        h, w = img.shape[:2]
        all_instances: List[InstanceData] = []

        for y, y_end, x, x_end in self._tiles(h, w):
            crop = img[y:y_end, x:x_end, :]
            ch, cw = crop.shape[:2]
            if ch != self.sw.window_size or cw != self.sw.window_size:
                crop = cv2.copyMakeBorder(
                    crop, 0, self.sw.window_size - ch, 0, self.sw.window_size - cw,
                    borderType=cv2.BORDER_CONSTANT, value=self.sw.pad_value
                )
            result = detector.infer_crop(crop)
            inst = result.pred_instances
            if inst is None or len(inst) == 0:
                continue

            valid = inst.scores > nms_params.score_thr
            inst = inst[valid]
            if len(inst) == 0:
                continue

            if self.sw.filter_to_valid_region:
                b = inst.bboxes
                cx = (b[:, 0] + b[:, 2]) * 0.5
                cy = (b[:, 1] + b[:, 3]) * 0.5
                in_valid = (cx < cw) & (cy < ch)
                inst = inst[in_valid]
                if len(inst) == 0:
                    continue

            inst = inst.cpu()
            inst.bboxes[:, 0::2] += x
            inst.bboxes[:, 1::2] += y
            inst.bboxes[:, 0::2] = torch.clamp(inst.bboxes[:, 0::2], 0, w)
            inst.bboxes[:, 1::2] = torch.clamp(inst.bboxes[:, 1::2], 0, h)
            all_instances.append(inst)

        if not all_instances:
            empty = InstanceData(
                bboxes=torch.zeros((0, 4), dtype=torch.float32),
                scores=torch.zeros((0,), dtype=torch.float32),
                labels=torch.zeros((0,), dtype=torch.int64)
            )
            out = DetDataSample(); out.pred_instances = empty
            return out

        merged = InstanceData.cat(all_instances)
        keep = batched_nms(merged.bboxes, merged.scores, merged.labels, nms_params)
        final = DetDataSample(); final.pred_instances = merged[keep[:]].cpu()
        return final


class Exporter:
    @staticmethod
    def copy_original(src: str, out_root: str):
        out_dir = os.path.join(out_root, "ori"); os.makedirs(out_dir, exist_ok=True)
        shutil.copy2(src, os.path.join(out_dir, os.path.basename(src)))

    @staticmethod
    def save_yolo_txt(out_root: str, image_basename: str,
                      bboxes_xyxy: np.ndarray, H: int, W: int,
                      subdir: Optional[str] = None, class_id: int = 0):
        out_dir = os.path.join(out_root, "txt")
        if subdir:
            out_dir = os.path.join(out_dir, subdir)
        os.makedirs(out_dir, exist_ok=True)
        txt = os.path.join(out_dir, os.path.splitext(image_basename)[0] + ".txt")
        with open(txt, "w") as f:
            for x1, y1, x2, y2 in bboxes_xyxy:
                xc, yc = (x1 + x2) / 2 / W, (y1 + y2) / 2 / H
                w, h = (x2 - x1) / W, (y2 - y1) / H
                f.write(f"{class_id} {xc} {yc} {w} {h}\n")

    @staticmethod
    def save_rect_vis(det: Detector, img: np.ndarray, result: DetDataSample, out_path: str):
        vis = det.get_visualizer()
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        vis.add_datasample('detection', img, result, draw_pred=True, draw_gt=False, show=False, out_file=out_path)

    @staticmethod
    def save_circle_vis(img: np.ndarray, bboxes: np.ndarray, out_path: str):
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        draw = img.copy()
        for x1, y1, x2, y2 in bboxes:
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
            cv2.circle(draw, (cx, cy), 2, (255, 0, 0), -1, lineType=cv2.LINE_AA)
        cv2.imwrite(out_path, cv2.cvtColor(draw, cv2.COLOR_RGB2BGR))

    @staticmethod
    def save_index_vis(img: np.ndarray, bboxes: np.ndarray, out_path: str):
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        draw = img.copy()
        for i, (x1, y1, x2, y2) in enumerate(bboxes):
            x1i, y1i, x2i, y2i = map(int, (x1, y1, x2, y2))
            cv2.rectangle(draw, (x1i, y1i), (x2i, y2i), (0, 255, 0), 1, lineType=cv2.LINE_AA)
            cv2.putText(draw, str(i), (x1i, max(y1i - 5, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        cv2.imwrite(out_path, cv2.cvtColor(draw, cv2.COLOR_RGB2BGR))

    @staticmethod
    def to_coco_placeholder(*args, **kwargs):
        pass  # 预留：把你之前注释的 json.dump 放进来即可

    @staticmethod
    def normalized_result(img_path: str, bboxes: np.ndarray, out_path: str):
        """
        输出每个中心正方形(t=0.50..0.95)的：
        - _cnt：与该窗有交集的检测框数量 N_t
        - _den：密度 N_t / area_real(t)  (单位：个/平方米；整图=1 m^2)
        """
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        img = mmcv.imread(img_path)
        H, W = img.shape[:2]

        thresholds = [round(x, 2) for x in np.arange(0.50, 0.95 + 1e-9, 0.05)]

        if bboxes is None or len(bboxes) == 0:
            bboxes = np.zeros((0, 4), dtype=np.float32)
        else:
            bboxes = np.asarray(bboxes, dtype=np.float32).reshape(-1, 4)

        side_base = float(min(H, W))
        cx, cy = W / 2.0, H / 2.0

        def rect_intersects(a, b):
            x1 = max(a[0], b[0]);
            y1 = max(a[1], b[1])
            x2 = min(a[2], b[2]);
            y2 = min(a[3], b[3])
            return (x2 - x1) > 0 and (y2 - y1) > 0

        row = {"filename": os.path.basename(img_path), "H": int(H), "W": int(W)}

        for t in thresholds:
            # 中心正方形（像素）
            side = t * side_base
            half = side / 2.0
            sq = (
                max(0.0, cx - half),
                max(0.0, cy - half),
                min(float(W), cx + half),
                min(float(H), cy + half),
            )

            # 实际面积：整图 = 1.0 m^2
            real_area = (side * side) / (H * W)  # m^2

            # 计数
            count = 0
            for box in bboxes:
                x1, y1, x2, y2 = map(float, box)
                if x2 <= x1 or y2 <= y1:
                    continue
                if rect_intersects((x1, y1, x2, y2), sq):
                    count += 1

            density = (count / real_area) if real_area > 0 else 0.0

            # 两列同时写入
            k_cnt = f"{t:.2f}_cnt"
            k_den = f"{t:.2f}_den"
            row[k_cnt] = int(count)
            row[k_den] = float(density)

        df_new = pd.DataFrame([row])

        # 追加或新建 —— 注意把所有 *_cnt 与 *_den 列都纳入列对齐
        if os.path.exists(out_path):
            df_old = pd.read_excel(out_path)
            # 目标列集合（保持 filename,H,W 在最前）
            metric_cols = []
            for t in thresholds:
                metric_cols += [f"{t:.2f}_cnt", f"{t:.2f}_den"]
            target_cols = ["filename", "H", "W"] + metric_cols

            # 确保旧表包含所有目标列；缺失则填 NaN
            for c in target_cols:
                if c not in df_old.columns:
                    df_old[c] = np.nan
            # 确保新表也有旧表里非目标的其他列（通常不需要，但为稳妥合并）
            for c in df_old.columns:
                if c not in df_new.columns:
                    df_new[c] = np.nan

            df_out = pd.concat([df_old[target_cols], df_new[target_cols]], ignore_index=True)
        else:
            df_out = df_new

        df_out.to_excel(out_path, index=False)


class Evaluator:
    @staticmethod
    def _yolo_to_xyxy(xc, yc, w, h):
        return xc - w/2, yc - h/2, xc + w/2, yc + h/2

    @staticmethod
    def _iou_xyxy(b1, b2):
        x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
        x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        a1 = max(0, b1[2]-b1[0]) * max(0, b1[3]-b1[1])
        a2 = max(0, b2[2]-b2[0]) * max(0, b2[3]-b2[1])
        return inter / (a1 + a2 - inter + 1e-7)

    @staticmethod
    def prf_from_yolo_abs(pred_xywh: List[List[float]],
                          gt_xywh: List[List[float]],
                          iou_thr: float) -> Tuple[int, int, int]:
        pred = [Evaluator._yolo_to_xyxy(*b) for b in pred_xywh]
        gt = [Evaluator._yolo_to_xyxy(*b) for b in gt_xywh]
        matched = set(); tp = 0; fp = 0
        for pb in pred:
            best, best_i = 0.0, -1
            for i, gb in enumerate(gt):
                if i in matched: continue
                iou = Evaluator._iou_xyxy(pb, gb)
                if iou > best: best, best_i = iou, i
            if best >= iou_thr:
                tp += 1; matched.add(best_i)
            else:
                fp += 1
        fn = len(gt) - len(matched)
        return tp, fp, fn


class Runner:
    def __init__(self, cfg: FullConfig):
        self.cfg = cfg
        self.det = Detector(cfg.model)
        self.tiler = Tiler(cfg.sliding_window)

    def _collect_images(self) -> List[str]:
        files = []
        for root, _, names in os.walk(self.cfg.data.input_dir):
            for f in names:
                if f.lower().endswith(self.cfg.data.image_exts):
                    files.append(os.path.join(root, f))
        files.sort()
        return files

    def _save_all(self, img_path: str, img: np.ndarray,
                  result: DetDataSample, bboxes: np.ndarray):
        save = self.cfg.run.save
        out_root = self.cfg.data.output_dir
        if save.ori:
            Exporter.copy_original(img_path, out_root)
        if save.txt:
            subdir = f"nms_{self.cfg.nms.nms_iou}_score_{self.cfg.nms.score_thr}"
            Exporter.save_yolo_txt(out_root, os.path.basename(img_path),
                                   bboxes, img.shape[0], img.shape[1], subdir=subdir)
        if save.rect_vis:
            p = os.path.join(out_root, "rect", os.path.basename(img_path))
            Exporter.save_rect_vis(self.det, img, result, p)
        if save.circle_vis:
            p = os.path.join(out_root, "circle", os.path.basename(img_path))
            Exporter.save_circle_vis(img, bboxes, p)
        if save.index_vis:
            p = os.path.join(out_root, "index", os.path.basename(img_path))
            Exporter.save_index_vis(img, bboxes, p)
        if save.normalized_result:
            p = os.path.join(out_root, 'normalized_result.xlsx')
            Exporter.normalized_result(img_path, bboxes, p)


    def run_once(self):
        paths = self._collect_images()
        if not paths:
            logger.warning("未发现图像。"); return
        logger.info(f"开始处理 {len(paths)} 张图片 ...")
        t0 = time.time()
        counts: List[Tuple[str, int]] = []
        for p in tqdm(paths, desc="Processing"):
            img = mmcv.imread(p, channel_order='rgb')
            result = self.tiler.infer_image(self.det, img, self.cfg.nms)
            inst = result.pred_instances
            bboxes = inst.bboxes.cpu().numpy() if len(inst) else np.zeros((0,4))
            self._save_all(p, img, result, bboxes)
            counts.append((os.path.basename(p), len(bboxes)))

        # 输出预测计数 CSV：文件名, 预测数量
        csv_path = os.path.join(self.cfg.data.output_dir, "counts.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["filename", "count"])
            w.writerows(counts)
            w.writerow(["TOTAL", sum(c for _, c in counts)])
        logger.info(f"计数 CSV 已保存: {csv_path}")
        logger.info(f"完成，用时 {time.time()-t0:.2f}s")

    def _load_yolo_abs(self, label_file: str, W: int, H: int) -> List[List[float]]:
        out = []
        with open(label_file, "r") as f:
            for line in f:
                sp = line.strip().split()
                if len(sp) >= 5:
                    _, xc, yc, w, h = map(float, sp[:5])
                    out.append([xc*W, yc*H, w*W, h*H])
        return out

    def grid_search(self):
        """以“逐图计数差之和最小”为目标（与你当前策略一致）。"""
        run = self.cfg.run
        paths = self._collect_images()
        if not paths:
            logger.warning("未发现图像。"); return

        label_root = os.path.join(self.cfg.data.dataset_dir, run.metrics.label_root_rel)
        best = dict(min_total_diff=float('inf'), nms=None, score=None, iou_type=None)
        t_all = time.time()

        def linspace(start: float, stop: float, num: int) -> List[float]:
            return list(np.linspace(start, stop, num))

        nms_start, nms_stop, nms_num = run.grid.nms_range
        sc_start, sc_stop, sc_num = run.grid.score_range
        nms_vals = linspace(float(nms_start), float(nms_stop), int(nms_num))
        sc_vals = linspace(float(sc_start), float(sc_stop), int(sc_num))

        for iou_t in run.grid.iou_types:
            for nms_iou in nms_vals:
                for score_thr in sc_vals:
                    cur = NMSParams(
                        score_thr=float(score_thr),
                        nms_iou=float(nms_iou),
                        iou_type=iou_t,
                        use_soft_nms=self.cfg.nms.use_soft_nms,
                        soft_sigma=self.cfg.nms.soft_sigma,
                        soft_score_thr=self.cfg.nms.soft_score_thr
                    )

                    total_diff = 0; total_gt = 0; total_pred = 0
                    agg_tp = agg_fp = agg_fn = 0
                    subdir = f"nms_{cur.nms_iou}_score_{cur.score_thr}"
                    out_txt_dir = os.path.join(self.cfg.data.output_dir, "txt", subdir)
                    os.makedirs(out_txt_dir, exist_ok=True)

                    for p in tqdm(paths, desc=f"[{iou_t}] nms={cur.nms_iou:.2f}, score={cur.score_thr:.2f}"):
                        img = mmcv.imread(p, channel_order='rgb')
                        H, W = img.shape[:2]
                        result = self.tiler.infer_image(self.det, img, cur)
                        inst = result.pred_instances
                        bboxes = inst.bboxes.cpu().numpy() if len(inst) else np.zeros((0,4))

                        # 保存当前参数下的预测 YOLO
                        Exporter.save_yolo_txt(self.cfg.data.output_dir,
                                               os.path.basename(p),
                                               bboxes, H, W, subdir=subdir)

                        pred_cnt = len(bboxes)
                        label_file = os.path.join(label_root, os.path.splitext(os.path.basename(p))[0] + ".txt")
                        if not os.path.exists(label_file):
                            logger.warning(f"缺少标签: {label_file}")
                            continue
                        gt_abs = self._load_yolo_abs(label_file, W, H)
                        gt_cnt = len(gt_abs)

                        total_diff += abs(pred_cnt - gt_cnt)
                        total_gt += gt_cnt; total_pred += pred_cnt

                        if run.metrics.calc_prf:
                            pred_abs_xywh = [[(x1+x2)/2, (y1+y2)/2, (x2-x1), (y2-y1)] for (x1,y1,x2,y2) in bboxes]
                            tp, fp, fn = Evaluator.prf_from_yolo_abs(pred_abs_xywh, gt_abs, iou_thr=cur.nms_iou)
                            agg_tp += tp; agg_fp += fp; agg_fn += fn

                    avg_diff = total_diff / max(len(paths), 1)
                    logger.info(f"[{iou_t}] nms={cur.nms_iou:.2f}, score={cur.score_thr:.2f} "
                                f"=> 逐图差总={total_diff}, 平均={avg_diff:.2f}, GT={total_gt}, PRED={total_pred}")

                    log_txt = os.path.join(self.cfg.data.output_dir, "total_diff_counts_log.txt")
                    with open(log_txt, "a") as f:
                        f.write(f"iou_type: {iou_t}, nms: {cur.nms_iou}, score: {cur.score_thr}\n")
                        f.write(f"总预测: {total_pred}, 总真实: {total_gt}, 总差距: {abs(total_gt-total_pred)}\n")
                        f.write(f"逐图差总: {total_diff}, 平均逐图差: {avg_diff}\n")

                    if total_diff < best["min_total_diff"]:
                        best.update(min_total_diff=total_diff, nms=cur.nms_iou, score=cur.score_thr, iou_type=iou_t)
                        logger.info(f"<<< 更新最优: {best} >>>")

        logger.info(f"网格搜索完成，用时 {time.time()-t_all:.2f}s")
        logger.info(f"最优配置: {best}")

    def run(self):
        mode = self.cfg.run.mode.lower()
        if mode == "run":
            self.run_once()
        elif mode == "grid":
            self.grid_search()
        else:
            raise ValueError(f"未知 run.mode: {self.cfg.run.mode}")


# =============================================================================
# Public API
# =============================================================================
def build_from_config(cfg_like: Any) -> Runner:
    """
    传入 path(str)/dict/FullConfig，返回 Runner。
    """
    cfg = load_config(cfg_like)
    # 一点点健壮性检查：
    if not os.path.exists(cfg.model.config):
        raise FileNotFoundError(cfg.model.config)
    if not os.path.exists(cfg.model.checkpoint):
        raise FileNotFoundError(cfg.model.checkpoint)
    if not os.path.isdir(cfg.data.input_dir):
        raise FileNotFoundError(cfg.data.input_dir)
    os.makedirs(cfg.data.output_dir, exist_ok=True)
    return Runner(cfg)


# 可选：提供一个 main() 方便直接跑（仍然不是 CLI，只是默认读取 config.yaml）
if __name__ == "__main__":
    cfg_path = os.environ.get("WHEAT_CFG", "config.yaml")
    runner = build_from_config(cfg_path)
    runner.run()
