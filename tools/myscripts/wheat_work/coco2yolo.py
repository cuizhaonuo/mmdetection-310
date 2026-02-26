#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
COCO(train2017 + val2017) -> YOLO 格式转换 & 8:1:1 切分 & 可视化 (防串标修正版)

修复与强化：
1) 以图片“绝对路径”(absolute path) 作为唯一键，不使用 image_id 做跨文件合并，防止 key 冲突。
2) 分别加载每份 COCO 标注，构建 {abs_image_path -> anns[]} 映射，再在合并阶段按“键=绝对路径”合并。
3) 每张图可视化前都重新读取图像，不复用全局 state；OpenCV 绘制不需要 plt.clear/close。
4) 标签写入为覆盖写，不做 append，防止并发/复用导致累积。

输入目录（默认）:
  /public/home/zncui/pycharm_project/mmdetection/data/coco2017/
    ├─ train2017/
    ├─ val2017/
    └─ annotations/
        ├─ instances_train2017.json
        └─ instances_val2017.json

输出目录（默认）:
  /public/home/zncui/pycharm_project/mmdetection/data/yolo/
    ├─ images/{train,val,test}/
    ├─ labels/{train,val,test}/
    ├─ vis/   # 可视化
    └─ data.yaml
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
import random
import shutil

import cv2
from tqdm import tqdm


# ===== 默认路径 =====
DEFAULT_COCO_ROOT = "/public/home/zncui/pycharm_project/mmdetection/data/coco2017"
DEFAULT_YOLO_ROOT = "/public/home/zncui/pycharm_project/mmdetection/data/yolo"

VIS_MAX_PER_SPLIT = 200
RANDOM_SEED = 20251010


def ensure_dir(p: str | Path):
    Path(p).mkdir(parents=True, exist_ok=True)


def norm_abs(path: str) -> str:
    """规范化绝对路径，作为唯一键。"""
    return str(Path(path).resolve())


def load_one_coco_split(
    ann_json: str, img_dir: str
) -> Tuple[Dict[str, dict], Dict[str, List[dict]], Dict[int, str]]:
    """
    读取一份 COCO 标注，构建
      images_by_abs: {abs_img_path: {"file_name","width","height","path"}}
      anns_by_abs:   {abs_img_path: [ann,...]}   # 只保留有 bbox 的实例
      cat_id_to_name: {cat_id: name}
    """
    images_by_abs: Dict[str, dict] = {}
    anns_by_abs: Dict[str, List[dict]] = defaultdict(list)
    cat_id_to_name: Dict[int, str] = {}

    if not os.path.isfile(ann_json):
        return images_by_abs, anns_by_abs, cat_id_to_name

    with open(ann_json, "r", encoding="utf-8") as f:
        coco = json.load(f)

    # 建图片索引（以绝对路径为 key）
    id2abs: Dict[int, str] = {}
    for im in coco.get("images", []):
        file_name = im["file_name"]
        width, height = im.get("width"), im.get("height")
        abs_path = norm_abs(os.path.join(img_dir, file_name))
        id2abs[im["id"]] = abs_path
        images_by_abs[abs_path] = {
            "file_name": file_name,
            "width": width,
            "height": height,
            "path": abs_path,
        }

    # 收集标注（使用 abs_path 分桶，避免 image_id 冲突）
    for ann in coco.get("annotations", []):
        if "bbox" not in ann:
            continue
        img_id = ann.get("image_id")
        if img_id not in id2abs:
            continue
        x, y, w, h = ann["bbox"]
        if w <= 1e-6 or h <= 1e-6:
            continue
        abs_path = id2abs[img_id]
        anns_by_abs[abs_path].append(ann)

    # 类别表
    for c in coco.get("categories", []):
        cat_id_to_name[c["id"]] = c["name"]

    return images_by_abs, anns_by_abs, cat_id_to_name


def build_global_index(coco_root: str):
    """
    从 train/val 两个 split 构建全局索引（按“绝对路径”合并，不依赖 image_id）。
    """
    ann_dir = os.path.join(coco_root, "annotations")
    train_ann = os.path.join(ann_dir, "instances_train2017.json")
    val_ann = os.path.join(ann_dir, "instances_val2017.json")
    train_img_dir = os.path.join(coco_root, "train2017")
    val_img_dir = os.path.join(coco_root, "val2017")

    images_all: Dict[str, dict] = {}
    anns_all: Dict[str, List[dict]] = defaultdict(list)
    cat_all: Dict[int, str] = {}

    for ann_json, img_dir in [
        (train_ann, train_img_dir),
        (val_ann, val_img_dir),
    ]:
        imgs, anns, cats = load_one_coco_split(ann_json, img_dir)
        # 图片元数据：后加载的 split 同键会覆盖（基本同一文件不会重复）
        for k, v in imgs.items():
            images_all[k] = v
        # 标注：追加
        for k, v in anns.items():
            anns_all[k].extend(v)
        # 类别表：合并（COCO 的 id/name 一致；若不一致，以第一次出现为准）
        for cid, name in cats.items():
            if cid not in cat_all:
                cat_all[cid] = name

    return images_all, anns_all, cat_all


def build_contiguous_cat_map(cat_id_to_name: Dict[int, str]):
    """COCO 稀疏 cat_id -> 连续 new_id，返回 old2new 与 names 列表"""
    pairs = sorted(cat_id_to_name.items(), key=lambda kv: kv[0])
    old2new = {}
    names = []
    for i, (old, name) in enumerate(pairs):
        old2new[old] = i
        names.append(name)
    return old2new, names


def coco_bbox_to_yolo_line(bbox_xywh, img_w, img_h, cls_id_new: int) -> str:
    """COCO: [x,y,w,h]  -> YOLO: cls cx cy w h  (归一化)"""
    x, y, w, h = bbox_xywh
    # clamp 到图像范围，避免越界造成归一化错误
    x = max(0.0, min(x, img_w - 1.0))
    y = max(0.0, min(y, img_h - 1.0))
    w = max(0.0, min(w, img_w - x))
    h = max(0.0, min(h, img_h - y))

    cx = x + w / 2.0
    cy = y + h / 2.0
    nx, ny = cx / img_w, cy / img_h
    nw, nh = w / img_w, h / img_h

    # 二次 clamp
    nx = max(0.0, min(1.0, nx))
    ny = max(0.0, min(1.0, ny))
    nw = max(0.0, min(1.0, nw))
    nh = max(0.0, min(1.0, nh))

    return f"{cls_id_new} {nx:.6f} {ny:.6f} {nw:.6f} {nh:.6f}"


def write_yolo_label(path_txt: str, lines: List[str]):
    ensure_dir(Path(path_txt).parent)
    with open(path_txt, "w", encoding="utf-8") as f:
        if lines:
            f.write("\n".join(lines))
        # 无目标时写空文件也可（YOLO 会跳过）


def link_or_copy(src: str, dst: str):
    ensure_dir(Path(dst).parent)
    try:
        os.link(src, dst)   # 硬链接省空间
    except Exception:
        shutil.copy2(src, dst)


def draw_vis(abs_img_path: str, anns_for_img: List[dict],
             cat_old2new: Dict[int, int], cat_id_to_name: Dict[int, str],
             out_path_jpg: str):
    """每张图独立读取 & 绘制（不复用任何全局画布/列表）"""
    img = cv2.imread(abs_img_path)
    if img is None:
        return
    h, w = img.shape[:2]

    # 每个 ann 只画属于本图的框（这里 anns_for_img 已经按 abs_path 过滤）
    for ann in anns_for_img:
        if "bbox" not in ann:
            continue
        old_cid = ann.get("category_id")
        if old_cid not in cat_old2new:
            continue
        x, y, bw, bh = ann["bbox"]
        # clamp
        x = max(0, min(int(round(x)), w - 1))
        y = max(0, min(int(round(y)), h - 1))
        bw = max(1, min(int(round(bw)), w - x))
        bh = max(1, min(int(round(bh)), h - y))
        cv2.rectangle(img, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
        label = f"{cat_old2new[old_cid]}:{cat_id_to_name.get(old_cid, str(old_cid))}"
        cv2.putText(img, label, (x, max(0, y - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12), 1, cv2.LINE_AA)

    ensure_dir(Path(out_path_jpg).parent)
    cv2.imwrite(out_path_jpg, img)


def make_data_yaml(yolo_root: str, names: List[str]):
    data = {
        "path": str(Path(yolo_root).resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(names),
        "names": names,
    }
    yaml_path = os.path.join(yolo_root, "data.yaml")
    try:
        import yaml
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)
        print(f"[OK] data.yaml written: {yaml_path}")
    except Exception as e:
        print(f"[WARN] writing data.yaml failed (install pyyaml?): {e}")


def main():
    parser = argparse.ArgumentParser(description="COCO -> YOLO (8:1:1) with safe visualization")
    parser.add_argument("--coco_root", type=str, default=DEFAULT_COCO_ROOT)
    parser.add_argument("--yolo_root", type=str, default=DEFAULT_YOLO_ROOT)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--vis_max_per_split", type=int, default=VIS_MAX_PER_SPLIT)
    args = parser.parse_args()

    random.seed(args.seed)

    # 1) 建全局索引（按“绝对路径”键）
    print("[*] Building global index...")
    images_all, anns_all, cat_id_to_name = build_global_index(args.coco_root)
    if not images_all:
        raise RuntimeError("No images found. Check coco_root structure.")
    print(f"[*] Num images: {len(images_all)} | with anns: {sum(1 for k in images_all if k in anns_all)}")
    print(f"[*] Categories: {len(cat_id_to_name)}")

    # 2) 类别连续映射
    cat_old2new, names = build_contiguous_cat_map(cat_id_to_name)

    # 3) 切分（以“绝对路径”列表为对象）
    abs_paths = list(images_all.keys())
    random.shuffle(abs_paths)
    n_total = len(abs_paths)
    n_train = int(n_total * 0.8 + 0.5)
    n_val = int(n_total * 0.1 + 0.0)
    n_test = n_total - n_train - n_val

    split_paths = {
        "train": abs_paths[:n_train],
        "val": abs_paths[n_train:n_train + n_val],
        "test": abs_paths[n_train + n_val:],
    }
    print(f"[*] Split sizes -> train: {len(split_paths['train'])}, val: {len(split_paths['val'])}, test: {len(split_paths['test'])}")

    # 4) 目标目录
    for sub in ["images/train", "images/val", "images/test", "labels/train", "labels/val", "labels/test", "vis"]:
        ensure_dir(os.path.join(args.yolo_root, sub))

    # 5) 转换与可视化（逐图，按绝对路径唯一键，杜绝跨图累加）
    for split_name, paths in split_paths.items():
        print(f"[*] Processing split: {split_name} ({len(paths)})")
        vis_count = 0
        for abs_img_path in tqdm(paths):
            meta = images_all[abs_img_path]
            img_w, img_h = meta.get("width"), meta.get("height")
            if (img_w is None) or (img_h is None):
                # 兜底从文件读
                im = cv2.imread(abs_img_path)
                if im is None:
                    continue
                img_h, img_w = im.shape[:2]

            base = Path(abs_img_path).name
            stem = Path(base).stem
            dst_img = os.path.join(args.yolo_root, "images", split_name, base)
            dst_lbl = os.path.join(args.yolo_root, "labels", split_name, stem + ".txt")

            # 仅取这张图的 anns（键=绝对路径），无任何跨图累加
            anns_for_img = anns_all.get(abs_img_path, [])

            # 生成 YOLO 标签（覆盖写）
            lines = []
            for ann in anns_for_img:
                old_cid = ann.get("category_id")
                if old_cid not in cat_old2new:
                    continue
                new_id = cat_old2new[old_cid]
                line = coco_bbox_to_yolo_line(ann["bbox"], img_w, img_h, new_id)
                # 过滤 0 尺寸
                parts = line.split()
                if float(parts[3]) <= 0.0 or float(parts[4]) <= 0.0:
                    continue
                lines.append(line)
            write_yolo_label(dst_lbl, lines)

            # 复制/硬链图片（避免同名被覆盖：使用原 basename，且每个 split 独立目录）
            link_or_copy(abs_img_path, dst_img)

            # 可视化：每个 split 限额，防止整套渲染过慢
            if vis_count < args.vis_max_per_split:
                vis_out = os.path.join(args.yolo_root, "vis", f"{split_name}_{base}")
                draw_vis(abs_img_path, anns_for_img, cat_old2new, cat_id_to_name, vis_out)
                vis_count += 1

        print(f"[OK] {split_name} done. Visualized {vis_count} → {os.path.join(args.yolo_root, 'vis')}")

    # 6) data.yaml
    make_data_yaml(args.yolo_root, names)

    print("\n[ALL DONE]")
    print(f"YOLO root: {args.yolo_root}")
    print(f"  images/: train/ val/ test/")
    print(f"  labels/: train/ val/ test/")
    print(f"  vis/:    每个 split 前 {args.vis_max_per_split} 张可视化")
    print("  data.yaml 已生成，可直接用于 YOLO 训练")


if __name__ == "__main__":
    main()
