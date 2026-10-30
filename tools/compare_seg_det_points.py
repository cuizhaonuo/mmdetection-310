#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分割模型(A) 与 检测模型(B) 预测点差异比对。

输入:
  - A 分割: <seg_npy_dir>/<stem>.npy    逐像素实例标签图(uint16, 2D)
              0 = 小区外填充; 少数超大稀疏块 = 土壤/叶层; 其余每个 id = 一个麦穗实例
  - B 检测: <det_result_dir>/txt/<txt_subdir>/<stem>.txt   YOLO 归一化 (cls xc yc w h)
  - 原图:   <det_result_dir>/ori/<stem>.jpg

逻辑:
  1) A 侧: bincount 取各 id 面积 -> 剔除面积 > blob_area_thr 的大块 -> 其余实例取质心作代表点
  2) B 侧: YOLO 框中心作代表点
  3) 带门限的匈牙利一对一匹配(固定 20px 欧氏中心距)
  4) 未匹配点分类: A 独有 / B 独有(再按落点在 穗/大块/填充区 细分)

输出 (<out_dir>):
  - summary.csv            每图一行 + TOTAL 行
  - detail/<stem>.csv      每个匹配对 / A独有 / B独有 一行
  - overlay/<stem>.jpg     全尺寸标注图
  - thumbs/<stem>/*.jpg    未匹配点的 contact sheet 图集
"""

import os
import csv
import glob
import argparse
import logging

import numpy as np
import cv2
from scipy import ndimage
from scipy.optimize import linear_sum_assignment

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s', datefmt='%H:%M:%S'
)
logger = logging.getLogger('seg-det-cmp')


# =============================================================================
# 默认参数(模块级常量作 argparse default, 保证裸跑可用)
# =============================================================================
SEG_NPY_DIR    = "/public/home/qwang/DJI/guoweilong_wheat/20260817/test_compare/test_100_wang/test_100"
DET_RESULT_DIR = "/public/home/zncui/datasets/wheat50/wheat-8-17/test_100_zncui"
TXT_SUBDIR     = "nms_0.35_score_0.3"
OUT_DIR        = ""          # 空则取 <det_result_dir>/compare_wang_vs_ours
MATCH_DIST     = 20.0        # 匹配门限(px)
BLOB_AREA_THR  = 30000       # 面积超过此值视为土壤/叶层大块, 不算麦穗
CROP_SIZE      = 128
SHEET_COLS     = 10
SHEET_ROWS     = 10
BIG_COST       = 1e6         # 门限外的哨兵代价, 必须是有限值
JPEG_QUALITY   = 90
CAPTION_H      = 18          # 小图字幕条高度
HEADER_H       = 30          # contact sheet 表头高度

# 颜色 (BGR)
C_MATCH   = (0, 255, 0)      # 绿: 匹配上
C_A_ONLY  = (255, 0, 0)      # 蓝: A(分割)独有
C_B_ONLY  = (0, 0, 255)      # 红: B独有
C_BOX     = (0, 255, 255)    # 黄: 检测框

# 只用上面三种颜色。B独有的落点类别(穗/大块/填充区)不再用颜色区分,
# 只写在小图字幕和明细 CSV 的 b_center_category 列里。
CAT_ABBR  = {"spike": "spk", "big_blob": "blb", "padding": "pad"}
CAT_ORDER = {"big_blob": 0, "padding": 1, "spike": 2}   # sheet 内排序: 失效模式聚在前面


def parse_args():
    ap = argparse.ArgumentParser(description="分割模型与检测模型的预测点差异比对(匈牙利匹配)")
    ap.add_argument("--seg-npy-dir", default=SEG_NPY_DIR, help="分割模型 .npy 实例标签图目录")
    ap.add_argument("--det-result-dir", default=DET_RESULT_DIR, help="检测结果目录(含 ori/ 和 txt/)")
    ap.add_argument("--txt-subdir", default=TXT_SUBDIR, help="txt/ 下的子目录名")
    ap.add_argument("--out-dir", default=OUT_DIR, help="输出目录, 留空则为 <det-result-dir>/compare_wang_vs_ours")
    ap.add_argument("--match-dist", type=float, default=MATCH_DIST, help="匹配门限(px), 中心距超过即视为不匹配")
    ap.add_argument("--blob-area-thr", type=int, default=BLOB_AREA_THR, help="面积超过此值的实例视为大块并剔除")
    ap.add_argument("--crop-size", type=int, default=CROP_SIZE, help="未匹配点裁剪小图边长(px)")
    ap.add_argument("--sheet-cols", type=int, default=SHEET_COLS, help="contact sheet 列数")
    ap.add_argument("--sheet-rows", type=int, default=SHEET_ROWS, help="contact sheet 行数")
    ap.add_argument("--big-cost", type=float, default=BIG_COST, help="门限外的哨兵代价(必须有限)")
    ap.add_argument("--jpeg-quality", type=int, default=JPEG_QUALITY, help="输出 jpg 质量")
    ap.add_argument("--overlay-scale", type=float, default=1.0, help="叠加图保存前的缩放比例")
    ap.add_argument("--draw-boxes", action="store_true", help="叠加图上同时画出检测框")
    ap.add_argument("--no-vis", action="store_true", help="跳过叠加图和图集(纯数值快速迭代)")
    ap.add_argument("--no-thumbs", action="store_true", help="跳过未匹配点的裁剪小图")
    ap.add_argument("--only", action="append", default=None, help="只处理指定 stem, 可重复")
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 张, 0 表示不限")
    ap.add_argument("--debug-ungated", action="store_true", help="额外解一次无门限矩阵, 打印距离分位作回归探针")
    return ap.parse_args()


# =============================================================================
# 数据读取
# =============================================================================
def load_yolo_boxes(txt_path: str, W: int, H: int):
    """读 YOLO txt, 返回 list[(cls, xc_n, yc_n, w_n, h_n, cx_px, cy_px, x1,y1,x2,y2)]。

    照抄自 tools/visualize_area.py:56, 刻意不 import ——
    那个模块在 import 时会执行 `from infer_wheat import load_config`, 会把 YAML 配置耦合进来。
    """
    out = []
    with open(txt_path, 'r') as f:
        for line in f:
            sp = line.strip().split()
            if len(sp) < 5:
                continue
            cls = int(float(sp[0]))
            xc, yc, w, h = map(float, sp[1:5])
            cx, cy = xc * W, yc * H
            x1 = (xc - w / 2) * W
            y1 = (yc - h / 2) * H
            x2 = (xc + w / 2) * W
            y2 = (yc + h / 2) * H
            out.append((cls, xc, yc, w, h, cx, cy, x1, y1, x2, y2))
    return out


def seg_instance_points(label_map: np.ndarray, blob_area_thr: int):
    """实例标签图 -> (保留的 id, 质心 xy, 面积, 质心是否在自身掩膜内, 被剔除的大块 id)。"""
    H, W = label_map.shape
    cnt = np.bincount(label_map.ravel())          # 一次扫描拿到全部 id 的面积
    ids = np.nonzero(cnt)[0]
    ids = ids[ids != 0]                           # 0 = 小区外填充, 不是实例

    empty = (np.zeros(0, np.int64), np.zeros((0, 2), np.float64),
             np.zeros(0, np.int64), np.zeros(0, bool), np.zeros(0, np.int64))
    if ids.size == 0:
        return empty

    areas = cnt[ids]
    blob_ids = ids[areas > blob_area_thr]
    keep = ids[areas <= blob_area_thr]
    if keep.size == 0:
        return (np.zeros(0, np.int64), np.zeros((0, 2), np.float64),
                np.zeros(0, np.int64), np.zeros(0, bool), blob_ids)

    # 坑1: 第一个参数必须是常数数组。传 label_map 自身会按标签值加权, 得到"看着合理"的错误质心。
    # 坑2: index 必须来自 bincount 的实际 id, 不能用 range(1, max+1) ——
    #      标签 id 不连续(实测有缺号), center_of_mass 对不存在的 index 返回 nan,
    #      nan 进代价矩阵后 linear_sum_assignment 会抛 "matrix contains invalid numeric entries",
    #      报错点离病根很远, 极难定位。
    ones = np.ones(label_map.shape, dtype=bool)
    com = np.asarray(ndimage.center_of_mass(ones, label_map, list(keep)),
                     dtype=np.float64).reshape(-1, 2)   # (row, col) = (y, x)

    pts = np.stack([com[:, 1], com[:, 0]], axis=1)      # -> (x, y)

    # 质心是否落在自身掩膜内(遮挡造成的弯月/碎块形状会落在外面)
    rr = np.clip(np.round(com[:, 0]).astype(np.int64), 0, H - 1)
    cc = np.clip(np.round(com[:, 1]).astype(np.int64), 0, W - 1)
    inside = label_map[rr, cc] == keep

    return keep, pts, cnt[keep], inside, blob_ids


# =============================================================================
# 匹配
# =============================================================================
def match_points(A: np.ndarray, B: np.ndarray, thr: float, big_cost: float):
    """带门限的匈牙利一对一匹配, 返回 (配对索引 Nx2, A独有索引, B独有索引, 配对距离)。"""
    nA, nB = len(A), len(B)
    if nA == 0 or nB == 0:
        return (np.zeros((0, 2), np.int64), np.arange(nA), np.arange(nB), np.zeros(0))

    D = np.linalg.norm(A[:, None, :] - B[None, :, :], axis=2)   # |A| x |B|

    # ！！！门限必须在求解之前进代价矩阵, 不能先解后筛 ！！！
    # 先解后筛: 无门限的解会把配额浪费在上百像素外的假配对上, 挤掉本该成立的近邻配对。
    # 实测同一张图: 先解后筛得 473 匹配 / 50 A独有 / 55 B独有; 先门限后解得 497 / 26 / 31。
    # (同门限的贪心最近邻也得 497, 互相印证。)
    # 下面这行看着和后面的 ok 过滤重复, 其实不是 —— 删掉它一张图就少 24 个匹配。
    C = np.where(D > thr, big_cost, D)

    # 不能用 np.inf: 整行全 inf 时 linear_sum_assignment 抛 ValueError: cost matrix is infeasible。
    r, c = linear_sum_assignment(C)

    # 哨兵不会自动剔除配对: |A| != |B| 时求解器必然返回哨兵代价的配对,
    # 所以按真实距离过滤是必须的, 不是防御性代码。
    # big_cost=1e6 安全: 满配对总代价上限约 550*20=11000 << 1e6,
    # 求解器会先最小化哨兵配对数, 再最小化剩余距离和, 正是所需目标。
    ok = D[r, c] <= thr
    pairs = np.stack([r[ok], c[ok]], axis=1)
    dists = D[r[ok], c[ok]]

    a_only = np.setdiff1d(np.arange(nA), pairs[:, 0])
    b_only = np.setdiff1d(np.arange(nB), pairs[:, 1])
    return pairs, a_only, b_only, dists


def categorize_b_points(label_map: np.ndarray, B: np.ndarray, blob_set: set):
    """B 点中心落在哪类标签上 -> (标签值, 类别字符串)。

    用 np.round 而非 floor: 两者都"对", 但必须固定一种, 否则两次运行的分类数永远差几个。
    """
    H, W = label_map.shape
    if len(B) == 0:
        return np.zeros(0, np.int64), []
    bi = np.clip(np.round(B[:, 1]).astype(np.int64), 0, H - 1)
    bj = np.clip(np.round(B[:, 0]).astype(np.int64), 0, W - 1)
    lab = label_map[bi, bj].astype(np.int64)
    cats = []
    for v in lab:
        if v == 0:
            cats.append("padding")
        elif v in blob_set:
            cats.append("big_blob")
        else:
            cats.append("spike")
    return lab, cats


# =============================================================================
# 可视化
# =============================================================================
def draw_overlay(img, A, B, pairs, a_only, b_only, b_cats, dists, stat, draw_boxes, boxes):
    """在原图副本上画出匹配/未匹配点, 左上角带英文图例。"""
    vis = img.copy()

    if draw_boxes:
        for (x1, y1, x2, y2) in boxes:
            cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), C_BOX, 1, cv2.LINE_AA)

    # 先画匹配, 再画未匹配 —— 保证少数关键点不被压在下面
    for ai, bi in pairs:
        ax, ay = A[ai]
        bx, by = B[bi]
        cv2.circle(vis, (int(round(ax)), int(round(ay))), 9, C_MATCH, 2, cv2.LINE_AA)
        cv2.circle(vis, (int(round(bx)), int(round(by))), 3, C_MATCH, -1, cv2.LINE_AA)

    for ai in a_only:
        ax, ay = A[ai]
        cv2.circle(vis, (int(round(ax)), int(round(ay))), 10, C_A_ONLY, -1, cv2.LINE_AA)

    for bi in b_only:
        bx, by = B[bi]
        cv2.circle(vis, (int(round(bx)), int(round(by))), 10, C_B_ONLY, -1, cv2.LINE_AA)

    # 图例。注意: cv2.putText 的 Hershey 字体只有 ASCII, 中文会渲染成 ?, 所以这里只能用英文。
    lines = [
        ("A(seg)=%d  B(det)=%d  matched=%d" % (stat["n_A"], stat["n_B"], stat["n_matched"]), C_MATCH),
        ("A-only=%d (outside-mask %d)" % (stat["n_A_only"], stat["n_A_only_centroid_outside"]), C_A_ONLY),
        ("B-only=%d  spike=%d blob=%d pad=%d" % (
            stat["n_B_only"], stat["n_B_only_on_spike"],
            stat["n_B_only_on_blob"], stat["n_B_only_on_padding"]), C_B_ONLY),
        ("dist p50=%s p90=%s max=%s" % (stat["dist_median"], stat["dist_p90"], stat["dist_max"]), (200, 200, 200)),
    ]
    font, fs, ft = cv2.FONT_HERSHEY_SIMPLEX, 1.6, 3
    pad, lh, dot_x = 20, 62, 44
    panel_w = max(cv2.getTextSize(t, font, fs, ft)[0][0] for t, _ in lines) + dot_x + pad * 2
    panel_h = lh * len(lines) + pad * 2
    panel = vis[0:panel_h, 0:panel_w].copy()
    cv2.rectangle(panel, (0, 0), (panel_w, panel_h), (0, 0, 0), -1)
    vis[0:panel_h, 0:panel_w] = cv2.addWeighted(panel, 0.55, vis[0:panel_h, 0:panel_w], 0.45, 0)
    for i, (text, color) in enumerate(lines):
        y = pad + lh * i + 44
        cv2.circle(vis, (pad + 12, y - 14), 12, color, -1, cv2.LINE_AA)
        cv2.putText(vis, text, (dot_x + pad, y), font, fs, (255, 255, 255), ft, cv2.LINE_AA)
    return vis


def crop_patch(img, cx, cy, S):
    """以 (cx, cy) 为中心裁 SxS。越界时先开零画布再贴源切片, 避免 copyMakeBorder 的差一像素。"""
    H, W = img.shape[:2]
    half = S // 2
    x0, y0 = int(round(cx)) - half, int(round(cy)) - half
    x1, y1 = x0 + S, y0 + S
    canvas = np.zeros((S, S, 3), np.uint8)
    sx0, sy0 = max(0, x0), max(0, y0)
    sx1, sy1 = min(W, x1), min(H, y1)
    if sx1 > sx0 and sy1 > sy0:
        canvas[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = img[sy0:sy1, sx0:sx1]
    return canvas


def make_contact_sheets(img, items, out_dir, prefix, stem, S, cols, rows, quality):
    """把未匹配点拼成 contact sheet 图集。

    items: list[(cx, cy, color, caption)] —— 顺序即 cell 序号。
    返回 list[(sheet_basename, cell_index)], 与 items 一一对应。
    """
    if not items:
        return []
    os.makedirs(out_dir, exist_ok=True)
    per = cols * rows
    cell_w, cell_h = S, S + CAPTION_H
    assign = []
    n_sheet = (len(items) + per - 1) // per

    for si in range(n_sheet):
        chunk = items[si * per:(si + 1) * per]
        sheet = np.zeros((HEADER_H + rows * cell_h, cols * cell_w, 3), np.uint8)
        cv2.putText(sheet, "%s  %s  sheet %d/%d  (%d pts)" % (stem, prefix, si, n_sheet - 1, len(items)),
                    (6, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        name = "%s_sheet%02d.jpg" % (prefix, si)
        for k, (cx, cy, color, caption) in enumerate(chunk):
            patch = crop_patch(img, cx, cy, S)
            # 中心标记: 没有它, 128px 里三四个穗根本分不清哪个是主角
            c = S // 2
            cv2.circle(patch, (c, c), 6, color, 1, cv2.LINE_AA)
            cv2.line(patch, (c - 5, c), (c + 5, c), color, 1, cv2.LINE_AA)
            cv2.line(patch, (c, c - 5), (c, c + 5), color, 1, cv2.LINE_AA)
            cell = np.zeros((cell_h, cell_w, 3), np.uint8)
            cell[0:S, 0:S] = patch
            cv2.putText(cell, caption, (2, S + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (255, 255, 255), 1, cv2.LINE_AA)
            r, cc_ = divmod(k, cols)
            sheet[HEADER_H + r * cell_h:HEADER_H + (r + 1) * cell_h,
                  cc_ * cell_w:(cc_ + 1) * cell_w] = cell
            assign.append((name, si * per + k))
        cv2.imwrite(os.path.join(out_dir, name), sheet, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return assign


# =============================================================================
# 单图处理
# =============================================================================
def _fmt(v, nd=3):
    return "" if v is None else ("%.*f" % (nd, v))


def process_one(stem, args, paths, all_dists):
    npy_path, jpg_path, txt_path = paths
    label_map = np.load(npy_path)
    if label_map.ndim != 2:
        logger.error("%s: npy 不是 2 维 (ndim=%d), 跳过", stem, label_map.ndim)
        return None
    img = cv2.imread(jpg_path)
    if img is None:
        logger.warning("%s: 原图读取失败, 跳过", stem)
        return None
    H, W = img.shape[:2]
    if label_map.shape != (H, W):
        # 不做缩放对齐 —— 静默改坐标比跳过更糟; 也不用 assert, 否则整轮在中途炸掉
        logger.error("%s: npy 尺寸 %s 与原图 %s 不一致, 跳过", stem, label_map.shape, (H, W))
        return None

    # ---- A 侧 ----
    a_ids, A, a_areas, a_inside, blob_ids = seg_instance_points(label_map, args.blob_area_thr)
    blob_set = set(int(v) for v in blob_ids)
    max_kept = int(a_areas.max()) if a_areas.size else 0
    if max_kept > 10000:
        logger.warning("%s: 保留实例最大面积 %d px 偏大, 大块阈值可能漏网", stem, max_kept)

    # ---- B 侧 ----
    rows = load_yolo_boxes(txt_path, W, H)
    B = np.array([[r[5], r[6]] for r in rows], dtype=np.float64).reshape(-1, 2)
    boxes = [(r[7], r[8], r[9], r[10]) for r in rows]
    b_wh = [(r[9] - r[7], r[10] - r[8]) for r in rows]
    b_lab, b_cats = categorize_b_points(label_map, B, blob_set)

    # ---- 匹配 ----
    pairs, a_only, b_only, dists = match_points(A, B, args.match_dist, args.big_cost)
    all_dists.append(dists)

    if args.debug_ungated and len(A) and len(B):
        D = np.linalg.norm(A[:, None, :] - B[None, :, :], axis=2)
        ur, uc = linear_sum_assignment(D)
        ud = D[ur, uc]
        wrong = int((ud <= args.match_dist).sum())
        logger.info("[debug-ungated] %s 无门限距离分位: p50=%.3f p75=%.3f p90=%.3f p95=%.3f max=%.3f",
                    stem, *[np.percentile(ud, q) for q in (50, 75, 90, 95)], ud.max())
        logger.info("[debug-ungated] %s 先解后筛(错误做法)会得到 匹配=%d A独有=%d B独有=%d; "
                    "本脚本先门限后解得到 匹配=%d A独有=%d B独有=%d",
                    stem, wrong, len(A) - wrong, len(B) - wrong,
                    len(pairs), len(a_only), len(b_only))

    matched_a = set(int(v) for v in pairs[:, 0])
    b_cat_of = {int(i): b_cats[int(i)] for i in range(len(B))}

    # ---- 统计 ----
    n_A, n_B, n_m = len(A), len(B), len(pairs)
    stat = {
        "image": stem, "img_w": W, "img_h": H,
        "n_seg_nonzero_ids": int(len(a_ids) + len(blob_ids)),
        "n_big_blobs": int(len(blob_ids)),
        "max_kept_area_px": max_kept,
        "n_A": n_A, "n_B": n_B, "n_matched": n_m,
        "n_A_only": len(a_only), "n_B_only": len(b_only),
        "recall_like": _fmt(n_m / n_A, 4) if n_A else "",
        "precision_like": _fmt(n_m / n_B, 4) if n_B else "",
        "f1": _fmt(2 * n_m / (n_A + n_B), 4) if (n_A + n_B) else "",
        "dist_mean": _fmt(float(dists.mean()) if dists.size else None),
        "dist_median": _fmt(float(np.median(dists)) if dists.size else None),
        "dist_p90": _fmt(float(np.percentile(dists, 90)) if dists.size else None),
        "dist_max": _fmt(float(dists.max()) if dists.size else None),
        "n_A_centroid_outside": int((~a_inside).sum()),
        "n_A_only_centroid_outside": int(sum(1 for i in a_only if not a_inside[i])),
        "n_B_on_spike": sum(1 for c in b_cats if c == "spike"),
        "n_B_on_blob": sum(1 for c in b_cats if c == "big_blob"),
        "n_B_on_padding": sum(1 for c in b_cats if c == "padding"),
        "n_B_only_on_spike": sum(1 for i in b_only if b_cat_of[int(i)] == "spike"),
        "n_B_only_on_blob": sum(1 for i in b_only if b_cat_of[int(i)] == "big_blob"),
        "n_B_only_on_padding": sum(1 for i in b_only if b_cat_of[int(i)] == "padding"),
    }

    # ---- contact sheet (顺序决定 cell 序号) ----
    thumb_a, thumb_b = {}, {}
    if not args.no_vis and not args.no_thumbs:
        tdir = os.path.join(args.out_dir, "thumbs", stem)
        # A 独有: 质心在掩膜外的排前面
        a_sorted = sorted(a_only, key=lambda i: (bool(a_inside[i]), int(a_ids[i])))
        a_items = [(A[i][0], A[i][1], C_A_ONLY, "#%d id%d" % (k, int(a_ids[i])))
                   for k, i in enumerate(a_sorted)]
        for (name, cell), i in zip(
                make_contact_sheets(img, a_items, tdir, "A_only", stem,
                                    args.crop_size, args.sheet_cols, args.sheet_rows, args.jpeg_quality),
                a_sorted):
            thumb_a[int(i)] = (name, cell)
        # B 独有: 按 大块 -> 填充区 -> 穗 排序, 让失效模式聚在一起
        b_sorted = sorted(b_only, key=lambda i: (CAT_ORDER[b_cat_of[int(i)]], int(i)))
        b_items = [(B[i][0], B[i][1], C_B_ONLY,
                    "#%d b%d %s" % (k, int(i), CAT_ABBR[b_cat_of[int(i)]]))
                   for k, i in enumerate(b_sorted)]
        for (name, cell), i in zip(
                make_contact_sheets(img, b_items, tdir, "B_only", stem,
                                    args.crop_size, args.sheet_cols, args.sheet_rows, args.jpeg_quality),
                b_sorted):
            thumb_b[int(i)] = (name, cell)

    # ---- 明细 CSV (行序固定, 便于两次运行 diff) ----
    det_dir = os.path.join(args.out_dir, "detail")
    os.makedirs(det_dir, exist_ok=True)
    with open(os.path.join(det_dir, stem + ".csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image", "record_type", "distance_px",
                    "a_id", "a_cx", "a_cy", "a_area_px", "a_centroid_inside_mask",
                    "b_index", "b_cx", "b_cy", "b_w_px", "b_h_px",
                    "b_label_at_center", "b_center_category",
                    "thumb_sheet", "thumb_cell"])

        def a_cols(i):
            return [int(a_ids[i]), _fmt(A[i][0], 2), _fmt(A[i][1], 2),
                    int(a_areas[i]), bool(a_inside[i])]

        def b_cols(i):
            return [int(i), _fmt(B[i][0], 2), _fmt(B[i][1], 2),
                    _fmt(b_wh[i][0], 1), _fmt(b_wh[i][1], 1),
                    int(b_lab[i]), b_cats[i]]

        order = np.argsort([int(a_ids[ai]) for ai, _ in pairs]) if len(pairs) else []
        for k in order:
            ai, bi = int(pairs[k][0]), int(pairs[k][1])
            w.writerow([stem, "matched", _fmt(float(dists[k]))] + a_cols(ai) + b_cols(bi) + ["", ""])
        for i in sorted(a_only, key=lambda i: int(a_ids[i])):
            sh, ce = thumb_a.get(int(i), ("", ""))
            w.writerow([stem, "A_only", ""] + a_cols(int(i)) + ["", "", "", "", "", "", ""] + [sh, ce])
        for i in sorted(b_only):
            sh, ce = thumb_b.get(int(i), ("", ""))
            w.writerow([stem, "B_only", "", "", "", "", "", ""] + b_cols(int(i)) + [sh, ce])

    # ---- 叠加图 ----
    if not args.no_vis:
        ov_dir = os.path.join(args.out_dir, "overlay")
        os.makedirs(ov_dir, exist_ok=True)
        vis = draw_overlay(img, A, B, pairs, a_only, b_only, b_cats, dists, stat, args.draw_boxes, boxes)
        if args.overlay_scale != 1.0:
            vis = cv2.resize(vis, None, fx=args.overlay_scale, fy=args.overlay_scale,
                             interpolation=cv2.INTER_AREA)
        cv2.imwrite(os.path.join(ov_dir, stem + ".jpg"), vis, [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality])

    return stat


SUMMARY_COLS = ["image", "img_w", "img_h",
                "n_seg_nonzero_ids", "n_big_blobs", "max_kept_area_px",
                "n_A", "n_B", "n_matched", "n_A_only", "n_B_only",
                "recall_like", "precision_like", "f1",
                "dist_mean", "dist_median", "dist_p90", "dist_max",
                "n_A_centroid_outside", "n_A_only_centroid_outside",
                "n_B_on_spike", "n_B_on_blob", "n_B_on_padding",
                "n_B_only_on_spike", "n_B_only_on_blob", "n_B_only_on_padding"]

SUM_INT_COLS = ["n_seg_nonzero_ids", "n_big_blobs", "n_A", "n_B", "n_matched",
                "n_A_only", "n_B_only", "n_A_centroid_outside", "n_A_only_centroid_outside",
                "n_B_on_spike", "n_B_on_blob", "n_B_on_padding",
                "n_B_only_on_spike", "n_B_only_on_blob", "n_B_only_on_padding"]


def main():
    args = parse_args()
    if not args.out_dir:
        args.out_dir = os.path.join(args.det_result_dir, "compare_wang_vs_ours")
    os.makedirs(args.out_dir, exist_ok=True)
    logger.info("参数: seg=%s", args.seg_npy_dir)
    logger.info("参数: det=%s  txt_subdir=%s", args.det_result_dir, args.txt_subdir)
    logger.info("参数: out=%s", args.out_dir)
    logger.info("参数: match_dist=%.1fpx  blob_area_thr=%d  big_cost=%g",
                args.match_dist, args.blob_area_thr, args.big_cost)

    npys = sorted(glob.glob(os.path.join(args.seg_npy_dir, "*.npy")))
    if args.only:
        keep = set(args.only)
        npys = [p for p in npys if os.path.splitext(os.path.basename(p))[0] in keep]
    if args.limit > 0:
        npys = npys[:args.limit]
    if not npys:
        logger.warning("未发现 .npy 文件。")
        return

    logger.info("共 %d 张待处理 ...", len(npys))
    stats, skipped, all_dists = [], 0, []
    for p in npys:
        stem = os.path.splitext(os.path.basename(p))[0]
        jpg = os.path.join(args.det_result_dir, "ori", stem + ".jpg")
        txt = os.path.join(args.det_result_dir, "txt", args.txt_subdir, stem + ".txt")
        if not os.path.exists(jpg):
            logger.warning("%s: 缺少原图 %s, 跳过", stem, jpg); skipped += 1; continue
        if not os.path.exists(txt):
            logger.warning("%s: 缺少 txt %s, 跳过", stem, txt); skipped += 1; continue
        st = process_one(stem, args, (p, jpg, txt), all_dists)
        if st is None:
            skipped += 1; continue
        stats.append(st)
        logger.info("%s: A=%d B=%d 匹配=%d A独有=%d B独有=%d (B独有 穗%d/大块%d/填充%d)",
                    stem, st["n_A"], st["n_B"], st["n_matched"], st["n_A_only"], st["n_B_only"],
                    st["n_B_only_on_spike"], st["n_B_only_on_blob"], st["n_B_only_on_padding"])

    if not stats:
        logger.warning("没有成功处理的图像。")
        return

    # ---- 汇总 CSV: TOTAL 行的计数直接求和, 比率由求和后的计数重算(不是逐图平均) ----
    tot = {c: sum(int(s[c]) for s in stats) for c in SUM_INT_COLS}
    gd = np.concatenate(all_dists) if all_dists and sum(len(d) for d in all_dists) else np.zeros(0)
    total = {c: "" for c in SUMMARY_COLS}
    total.update(tot)
    total["image"] = "TOTAL"
    total["max_kept_area_px"] = max(int(s["max_kept_area_px"]) for s in stats)
    total["recall_like"] = _fmt(tot["n_matched"] / tot["n_A"], 4) if tot["n_A"] else ""
    total["precision_like"] = _fmt(tot["n_matched"] / tot["n_B"], 4) if tot["n_B"] else ""
    total["f1"] = _fmt(2 * tot["n_matched"] / (tot["n_A"] + tot["n_B"]), 4) if (tot["n_A"] + tot["n_B"]) else ""
    if gd.size:
        total["dist_mean"] = _fmt(float(gd.mean()))
        total["dist_median"] = _fmt(float(np.median(gd)))
        total["dist_p90"] = _fmt(float(np.percentile(gd, 90)))
        total["dist_max"] = _fmt(float(gd.max()))

    sum_path = os.path.join(args.out_dir, "summary.csv")
    with open(sum_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(SUMMARY_COLS)
        for s in stats:
            w.writerow([s[c] for c in SUMMARY_COLS])
        w.writerow([total[c] for c in SUMMARY_COLS])

    logger.info("=" * 60)
    logger.info("处理 %d 张, 跳过 %d 张", len(stats), skipped)
    logger.info("合计: A=%d B=%d 匹配=%d A独有=%d B独有=%d",
                tot["n_A"], tot["n_B"], tot["n_matched"], tot["n_A_only"], tot["n_B_only"])
    logger.info("B独有落点: 穗=%d 大块=%d 填充区=%d",
                tot["n_B_only_on_spike"], tot["n_B_only_on_blob"], tot["n_B_only_on_padding"])
    logger.info("汇总 CSV: %s", sum_path)
    logger.info("明细 CSV: %s", os.path.join(args.out_dir, "detail"))
    if not args.no_vis:
        logger.info("叠加图  : %s", os.path.join(args.out_dir, "overlay"))
        if not args.no_thumbs:
            logger.info("图集    : %s", os.path.join(args.out_dir, "thumbs"))


if __name__ == "__main__":
    main()
