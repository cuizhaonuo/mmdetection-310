#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import cv2
import math
import numpy as np

def sort_four_points(pts):
    """
    将四个点按 [tl, tr, br, bl] 排序
    """
    pts = pts.astype(np.float32)
    s   = pts.sum(axis=1)
    diff= np.diff(pts, axis=1).reshape(-1)
    tl  = pts[np.argmin(s)]
    br  = pts[np.argmax(s)]
    tr  = pts[np.argmin(diff)]
    bl  = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype=np.int32)


def find_inner_polygon(img):
    """
    返回红框“洞”轮廓的拟合多边形顶点（内接），不做裁剪／涂黑。
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # 分离红色
    lower1, upper1 = np.array([0,100,50]),  np.array([10,255,255])
    lower2, upper2 = np.array([160,100,50]), np.array([180,255,255])
    m1 = cv2.inRange(hsv, lower1, upper1)
    m2 = cv2.inRange(hsv, lower2, upper2)
    mask = cv2.bitwise_or(m1, m2)
    # 闭操作去噪
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # findContours + 层级
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None or not contours:
        raise RuntimeError("未检测到红色轮廓")
    hier = hierarchy[0]

    # 外轮廓：面积最大
    areas     = [cv2.contourArea(c) for c in contours]
    idx_outer = int(np.argmax(areas))
    # 子轮廓（洞）
    children  = [i for i,h in enumerate(hier) if h[3] == idx_outer]
    if not children:
        raise RuntimeError("未检测到红框内洞轮廓")
    idx_hole  = children[int(np.argmax([areas[i] for i in children]))]
    hole_cnt  = contours[idx_hole]

    # 内接多边形拟合（优先 approxPolyDP，否则 minAreaRect）
    eps    = 0.02 * cv2.arcLength(hole_cnt, True)
    approx = cv2.approxPolyDP(hole_cnt, eps, True).reshape(-1,2)
    if len(approx) == 4:
        pts_inner = approx
    else:
        rect      = cv2.minAreaRect(hole_cnt)
        pts_inner = cv2.boxPoints(rect).astype(np.int32)

    return sort_four_points(pts_inner)


def main():
    root      = '/public/home/zncui/datasets/'
    src_dir   = os.path.join(root, 'wheat-guo')
    out_dir   = os.path.join(root, 'wheat-guo-crop')
    os.makedirs(out_dir, exist_ok=True)

    exts       = {'.jpg', '.jpeg', '.png'}
    area_scale = 1.2   # 参数 a：目标矩形面积 = 原矩形面积 × a

    for fn in os.listdir(src_dir):
        name, ext = os.path.splitext(fn)
        if ext.lower() not in exts:
            continue

        path = os.path.join(src_dir, fn)
        img  = cv2.imread(path)
        if img is None:
            print(f"[WARN] 读图失败：{fn}")
            continue

        try:
            pts = find_inner_polygon(img)
            # 原轴对齐外接矩形
            x, y, w, h = cv2.boundingRect(pts)

            # 按面积扩展 a 倍 ⇒ 宽高扩展 sqrt(a) 倍
            factor = math.sqrt(area_scale)
            new_w  = int(w * factor)
            new_h  = int(h * factor)
            cx     = x + w / 2
            cy     = y + h / 2

            x0 = int(cx - new_w / 2)
            y0 = int(cy - new_h / 2)
            x1 = x0 + new_w
            y1 = y0 + new_h

            # 边界裁剪
            x0 = max(0, x0)
            y0 = max(0, y0)
            x1 = min(img.shape[1], x1)
            y1 = min(img.shape[0], y1)

            cropped = img[y0:y1, x0:x1]

        except Exception as e:
            print(f"[ERROR] 处理 {fn} 失败：{e}")
            continue

        dst = os.path.join(out_dir, fn)
        # cv2.imwrite(dst, cropped)
        print(f"[OK] {fn} → 已裁剪并保存到 {out_dir}")

if __name__ == '__main__':
    main()
