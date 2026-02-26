#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自动检测白色细边框四角，将中间麦田透视为正方形。
- 角点获取已改为：优先用“红/粉外框 -> 次大内圈四边形”（来自代码 B 的思路）；
  若失败再回退到原先的“白边框分割/直线法”以提高鲁棒性。
- 透视前做形状体检，避免出现自交/极瘦四边形导致的“V字黑屏”。
- 可用 keep_px 外扩/内缩以保留红框或仅取麦田（>0向外，<0向内）。
"""

import os, sys, glob, math, argparse
import cv2
import numpy as np

# ----------------------- 通用工具函数 -----------------------
def order_corners(pts: np.ndarray) -> np.ndarray:
    """将4点按 tl,tr,br,bl 排序"""
    pts = np.array(pts, dtype=np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).reshape(-1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]
    return np.array([tl, tr, br, bl], dtype=np.float32)

def poly_area(p: np.ndarray) -> float:
    x = p[:,0]; y = p[:,1]
    return 0.5 * float(np.dot(x, np.roll(y,-1)) - np.dot(y, np.roll(x,-1)))

def score_quad(quad: np.ndarray, H: int, W: int) -> float:
    """候选四边形打分：面积大、近似正方、居中"""
    quad = np.array(quad, dtype=np.float32)
    area = abs(poly_area(quad))
    if area < 100: return -1e9
    tl,tr,br,bl = order_corners(quad)
    w1 = np.linalg.norm(tr-tl); w2 = np.linalg.norm(br-bl)
    h1 = np.linalg.norm(bl-tl); h2 = np.linalg.norm(br-tr)
    w = 0.5*(w1+w2); h = 0.5*(h1+h2)
    aspect = min(w,h)/max(w,h)
    cx = quad[:,0].mean()/W; cy = quad[:,1].mean()/H
    center_score = 1.0 - np.hypot(cx-0.5, cy-0.5)*1.4
    return area * (0.5+0.5*aspect) * max(center_score, 0.2)

def approx_to_four(cnt, peri_ratio_list=(0.01,0.015,0.02,0.03,0.05,0.08)):
    """在多种epsilon下尝试逼近为4点"""
    peri = cv2.arcLength(cnt, True)
    for r in peri_ratio_list:
        approx = cv2.approxPolyDP(cnt, r*peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            return approx.reshape(-1,2).astype(np.float32)
    return None

def expand_quad(quad: np.ndarray, offset: float) -> np.ndarray:
    """
    沿法线外扩/内缩四边形 offset 像素（正=外扩，负=内缩），
    用“平移直线取交点”实现，鲁棒且不依赖中心缩放。
    """
    q = order_corners(quad).astype(np.float32)

    def line_from(p0, p1):
        v = (p1 - p0).astype(np.float32)
        n = np.array([-v[1], v[0]], np.float32)  # 法向
        n_norm = np.linalg.norm(n) + 1e-6
        n = n / n_norm
        # 让外法向一致：面积<0表示顺时针，翻转
        if poly_area(q) < 0:
            n = -n
        a, b = n[0], n[1]
        c = -(a*p0[0] + b*p0[1])
        return a, b, c

    def shift_line(l, d):
        a,b,c = l
        return (a,b,c - d*math.hypot(a,b))

    def intersect(l1, l2):
        a1,b1,c1 = l1; a2,b2,c2 = l2
        D = a1*b2 - a2*b1
        if abs(D) < 1e-6:
            return (0.5*(q[:,0].min()+q[:,0].max()), 0.5*(q[:,1].min()+q[:,1].max()))
        x = (b1*c2 - b2*c1)/D
        y = (c1*a2 - c2*a1)/D
        return (x,y)

    # ax + by + c = 0 形式 (a, b, c)
    lines = [line_from(q[i], q[(i+1)%4]) for i in range(4)]
    shifted = [shift_line(L, offset) for L in lines]
    out = [intersect(shifted[i], shifted[(i+1)%4]) for i in range(4)]
    return np.array(out, dtype=np.float32)

def warp_to_square(img, src_quad, out_side=None):
    src = order_corners(src_quad).astype(np.float32)
    w1 = np.linalg.norm(src[1]-src[0]); w2 = np.linalg.norm(src[2]-src[3])
    h1 = np.linalg.norm(src[3]-src[0]); h2 = np.linalg.norm(src[2]-src[1])
    side = int(out_side if out_side else max(32, int(0.5*(w1+w2+h1+h2)/2)))
    dst = np.array([[0,0],[side-1,0],[side-1,side-1],[0,side-1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    out = cv2.warpPerspective(img, M, (side, side), flags=cv2.INTER_CUBIC)
    return out

def is_valid_quad(q: np.ndarray, min_angle=60, max_angle=120) -> bool:
    """排除自交/极瘦四边形；角度在 [min_angle, max_angle]"""
    if q is None or len(q) != 4: return False
    q = order_corners(q).astype(np.float32)
    # 自交/非凸：相邻边叉积必须同号
    v = [q[(i+1)%4]-q[i] for i in range(4)]
    cross = [v[i][0]*v[(i+1)%4][1] - v[i][1]*v[(i+1)%4][0] for i in range(4)]
    if (np.min(cross) <= 1e-6 and np.max(cross) >= -1e-6):
        return False
    # 角度检查
    angs = []
    for i in range(4):
        a = q[(i-1)%4]-q[i]; b = q[(i+1)%4]-q[i]
        ca = np.dot(a,b) / (np.linalg.norm(a)*np.linalg.norm(b)+1e-6)
        ca = np.clip(ca, -1, 1)
        angs.append(np.degrees(np.arccos(ca)))
    return all(min_angle <= a <= max_angle for a in angs)

# ----------------------- （来自代码 B）红/粉外框分割 -----------------------
def clean_small_islands_and_holes(mask, area_white_max=2_000):
    """
    清除小白点、填充小黑洞（按面积阈值）。
    - area_white_max: 小于该面积的白色连通域将被抹黑
    """
    m = mask.copy().astype(np.uint8)

    # 1) 清除小白点
    bw = (m > 0).astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] < area_white_max:
            m[labels == i] = 0

    # 2) 保留最大两个轮廓，其它轮廓内部填白（方便“外-内”求环带）
    contours, _ = cv2.findContours((m > 0).astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours_sorted = sorted(contours, key=cv2.contourArea, reverse=True)
    mask_processed = m.copy()
    if len(contours_sorted) > 2:
        mask_fill = np.zeros_like(m)
        for i in range(2, len(contours_sorted)):
            cv2.drawContours(mask_fill, [contours_sorted[i]], -1, 255, -1)
        mask_processed[mask_fill > 0] = 255
    return mask_processed

def extract_red_frame_mask(img, debug=False):
    """
    提取红/粉外框，并返回“次大内圈”的四边形角点（优先近似4点，否则用最小外接矩形）。
    返回：
        mask: 仅保留外圈-内圈之间的环带（便于可视化）
        inside_quad_vertices: 内圈四边形角点（np.int32[4,2]），若失败为 None
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 红色两段
    lo1, hi1 = (0,   60, 40), (10, 255,255)
    lo2, hi2 = (160, 60, 40), (180,255,255)
    mask1 = cv2.inRange(hsv, np.uint8(lo1), np.uint8(hi1))
    mask2 = cv2.inRange(hsv, np.uint8(lo2), np.uint8(hi2))
    mask  = cv2.bitwise_or(mask1, mask2)

    # 去噪与形态学
    mask = cv2.medianBlur(mask, 3)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))

    contours, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours_sorted = sorted(contours, key=cv2.contourArea, reverse=True)

    # 取外圈-内圈环带
    if len(contours_sorted) >= 2:
        mask_outer = np.zeros_like(mask)
        mask_inner = np.zeros_like(mask)
        cv2.drawContours(mask_outer, [contours_sorted[0]], -1, 255, -1)
        cv2.drawContours(mask_inner, [contours_sorted[1]], -1, 255, -1)
        mask = cv2.bitwise_and(mask_outer, cv2.bitwise_not(mask_inner))

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=10)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=10)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=50)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=10)

    mask = clean_small_islands_and_holes(mask, area_white_max=1500)

    # 再次找轮廓，取“内圈”（次大）
    contours, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours_sorted = sorted(contours, key=cv2.contourArea, reverse=True)

    inside_quad_vertices = None
    if len(contours_sorted) >= 2:
        inside_contour = contours_sorted[1]
        epsilon = 0.02 * cv2.arcLength(inside_contour, True)
        approx = cv2.approxPolyDP(inside_contour, epsilon, True)

        if 3 <= len(approx) <= 6:
            if len(approx) != 4:
                rect = cv2.minAreaRect(inside_contour)
                inside_quad_vertices = np.int0(cv2.boxPoints(rect))
            else:
                inside_quad_vertices = approx.reshape(-1, 2)

            # 按与质心的极角排序，保证顺序一致
            M = cv2.moments(inside_quad_vertices)
            if M.get("m00", 0) != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                inside_quad_vertices = sorted(
                    inside_quad_vertices,
                    key=lambda p: np.arctan2(p[1]-cy, p[0]-cx)
                )
                inside_quad_vertices = np.array(inside_quad_vertices, dtype=np.int32)

    if debug:
        return mask, inside_quad_vertices, {"mask1": mask1, "mask2": mask2}
    return mask, inside_quad_vertices

# ----------------------- A 原有：白边/直线回退（保留作兜底） -----------------------
def detect_quad_via_white(img):
    H, W = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    S = hsv[:,:,1]; V = hsv[:,:,2]

    s_thr = int(np.clip(np.median(S)*0.9, 25, 80))
    v_thr = int(np.clip(np.percentile(V, 75), 150, 245))

    mask_white = cv2.inRange(hsv, (0, 0, v_thr-25), (180, s_thr, 255))
    k = 5
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k|1, k|1))
    top = cv2.morphologyEx(V, cv2.MORPH_TOPHAT, kernel)
    _, mask_top = cv2.threshold(top, 15, 255, cv2.THRESH_BINARY)
    mask = cv2.bitwise_or(mask_white, mask_top)

    mask_thick = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask_thick = cv2.dilate(mask_thick, kernel, iterations=1)

    cnts, _ = cv2.findContours(mask_thick, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None; best_score = -1e9
    for c in cnts:
        if cv2.contourArea(c) < 0.002*H*W:
            continue
        approx = approx_to_four(c)
        if approx is None:
            rect = cv2.minAreaRect(c)
            approx = cv2.boxPoints(rect).astype(np.float32)
        sc = score_quad(approx, H, W)
        if sc > best_score:
            best = approx; best_score = sc
    return best, mask, mask_thick

def detect_quad_via_lines(img):
    H, W = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(gray, 60, 180)

    k = max(3, int(min(H,W)*0.004))
    edges = cv2.dilate(edges, np.ones((k|1, k|1), np.uint8), iterations=1)

    lines = cv2.HoughLinesP(edges, 1, np.pi/180,
                            threshold=int(0.004*H*W**0.5),
                            minLineLength=int(0.25*min(H,W)),
                            maxLineGap=int(0.02*min(H,W)))
    if lines is None:
        return None, edges
    lines = lines[:,0,:]

    angs = np.arctan2(lines[:,3]-lines[:,1], lines[:,2]-lines[:,0])
    angs = (angs + np.pi) % np.pi

    a_med = float(np.median(angs))
    def ang_dist(a,b):
        d = abs(a-b);  return min(d, np.pi-d)

    group0, group1 = [], []
    for i,a in enumerate(angs):
        if ang_dist(a, a_med) <= ang_dist(a, (a_med + np.pi/2) % np.pi):
            group0.append(i)
        else:
            group1.append(i)
    if len(group0) < 2 or len(group1) < 2:
        return None, edges

    cx, cy = W*0.5, H*0.5
    def canon_line(l):
        x1,y1,x2,y2 = map(float, l)
        a = y2 - y1; b = x1 - x2; c = -(a*x1 + b*y1)
        n = math.hypot(a,b) + 1e-6
        a,b,c = a/n, b/n, c/n
        s = a*cx + b*cy + c
        if s < 0: a,b,c = -a,-b,-c
        d = -c
        return (a,b,c,d)

    def pick_two(ls_idx):
        abcds = np.array([canon_line(lines[i]) for i in ls_idx])
        idx = np.argsort(abcds[:,3])
        return abcds[idx[0]], abcds[idx[-1]]

    L0a, L0b = pick_two(group0)
    L1a, L1b = pick_two(group1)

    def intersect(L1, L2):
        a1,b1,c1,_ = L1; a2,b2,c2,_ = L2
        D = a1*b2 - a2*b1
        if abs(D) < 1e-6: return None
        x = (b1*c2 - b2*c1)/D
        y = (c1*a2 - c2*a1)/D
        return np.array([x,y], np.float32)

    p_tl = intersect(L0a, L1a)
    p_tr = intersect(L0a, L1b)
    p_br = intersect(L0b, L1b)
    p_bl = intersect(L0b, L1a)
    quad = np.array([p_tl, p_tr, p_br, p_bl], dtype=np.float32)
    if np.any(np.isnan(quad)):
        return None, edges
    return quad, edges

# ----------------------- 新的统一检测入口（优先红/粉外框法） -----------------------
def detect_quad(img, prefer_inner=True, debug_collect=True):
    """
    优先使用：红/粉外框 -> 次大内圈四边形（代码 B 方法）。
    若失败或不合法，再回退到 A 的白边/直线法。
    """
    dbg = {'mode': None, 'mask_red': None, 'mask_white': None, 'mask_white_thick': None, 'edges': None}

    # 1) 代码 B：红/粉外框提取 + 次大内圈四边形
    mask_red, inner_quad = extract_red_frame_mask(img, debug=False)
    dbg['mask_red'] = mask_red
    if inner_quad is not None and len(inner_quad) == 4:
        quad_b = inner_quad.astype(np.float32)
        if is_valid_quad(quad_b):
            dbg['mode'] = 'red_inner'
            return quad_b, dbg, 'red_inner'

    # 2) A 的白边框回退
    quad_w, mask_w, mask_thick = detect_quad_via_white(img)
    dbg['mask_white'] = mask_w
    dbg['mask_white_thick'] = mask_thick
    if quad_w is not None and is_valid_quad(quad_w):
        dbg['mode'] = 'white'
        return quad_w, dbg, 'white'

    # 3) A 的直线回退
    quad_l, edges = detect_quad_via_lines(img)
    dbg['edges'] = edges
    if quad_l is not None and is_valid_quad(quad_l):
        dbg['mode'] = 'lines'
        return quad_l, dbg, 'lines'

    dbg['mode'] = 'fail'
    return None, dbg, 'fail'

# ----------------------- 处理与可视化 -----------------------
def process_one(img_path, out_dir, out_side=None, keep_px=60, write_debug=False):
    base = os.path.splitext(os.path.basename(img_path))[0]
    img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        print(f"[WARN] 读图失败: {img_path}")
        return

    quad, dbg, mode = detect_quad(img)
    if quad is None:
        print(f"[FAIL] 未找到合法四角形: {img_path}")
        return

    # 根据 keep_px 外扩/内缩（>0 向外，<0 向内）
    quad_keep = expand_quad(quad, keep_px)
    out = warp_to_square(img, quad_keep, out_side)


    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{base}.jpg")
    cv2.imencode(".jpg", out)[1].tofile(out_path)

    if write_debug:
        # 1. 标记原图
        dbg_img = img.copy()
        q0 = order_corners(quad).astype(int)
        q1 = order_corners(quad_keep).astype(int)
        cv2.polylines(dbg_img, [q0], True, (0,255,0), 4)
        cv2.polylines(dbg_img, [q1], True, (255,0,0), 4)
        for p in q0: cv2.circle(dbg_img, tuple(p), 12, (0,255,0), -1)
        for p in q1: cv2.circle(dbg_img, tuple(p), 12, (255,0,0), -1)

        dbg_path = os.path.join(out_dir, f"{base}_debug_marked.jpg")
        cv2.imencode(".jpg", dbg_img)[1].tofile(dbg_path)

        # 2. 保存各个调试图像
        if dbg.get('mask_red') is not None:
            mr_path = os.path.join(out_dir, f"{base}_debug_mask_red.jpg")
            cv2.imencode(".jpg", dbg['mask_red'])[1].tofile(mr_path)

        if dbg.get('mask_white') is not None:
            mw_path = os.path.join(out_dir, f"{base}_debug_mask_white.jpg")
            cv2.imencode(".jpg", dbg['mask_white'])[1].tofile(mw_path)

        if dbg.get('mask_white_thick') is not None:
            mt_path = os.path.join(out_dir, f"{base}_debug_mask_thick.jpg")
            cv2.imencode(".jpg", dbg['mask_white_thick'])[1].tofile(mt_path)

        if dbg.get('edges') is not None:
            ed_path = os.path.join(out_dir, f"{base}_debug_edges.jpg")
            cv2.imencode(".jpg", dbg['edges'])[1].tofile(ed_path)

        # 3. 左上角拼接调试小图
        tiles = []
        if dbg.get('mask_red') is not None:
            tiles += [cv2.cvtColor(dbg['mask_red'], cv2.COLOR_GRAY2BGR)]
        if dbg.get('mask_white') is not None:
            tiles += [cv2.cvtColor(dbg['mask_white'], cv2.COLOR_GRAY2BGR)]
        if dbg.get('mask_white_thick') is not None:
            tiles += [cv2.cvtColor(dbg['mask_white_thick'], cv2.COLOR_GRAY2BGR)]
        if dbg.get('edges') is not None:
            tiles += [cv2.cvtColor(dbg['edges'], cv2.COLOR_GRAY2BGR)]

        if tiles:
            tiles = [cv2.resize(t, (t.shape[1]//2, t.shape[0]//2)) for t in tiles]
            stack = np.hstack(tiles)
            h, w = stack.shape[:2]
            dbg_img_combined = img.copy()
            dbg_img_combined[0:h, 0:w] = stack
            cv2.polylines(dbg_img_combined, [q0], True, (0,255,0), 4)
            cv2.polylines(dbg_img_combined, [q1], True, (255,0,0), 4)
            for p in q0: cv2.circle(dbg_img_combined, tuple(p), 12, (0,255,0), -1)
            for p in q1: cv2.circle(dbg_img_combined, tuple(p), 12, (255,0,0), -1)

            dbg_path2 = os.path.join(out_dir, f"{base}_debug_combined.jpg")
            cv2.imencode(".jpg", dbg_img_combined)[1].tofile(dbg_path2)

    print(f"[OK:{mode}] {img_path} -> {out_path}")

def is_image(f):
    ext = os.path.splitext(f)[1].lower()
    return ext in [".jpg",".jpeg",".png",".bmp",".tif",".tiff",".webp"]

def main():
    ap = argparse.ArgumentParser(description="自动检测白/粉框四角并将中间麦田透视为正方形（已改为代码B的角点法）")
    ap.add_argument("--in", dest="in_path", required=True, help="输入文件或目录")
    ap.add_argument("--out", dest="out_dir", required=True, help="输出目录")
    ap.add_argument("--square", type=int, default=None, help="输出正方形边长（像素），默认自适应")
    ap.add_argument("--keep-px", type=float, default=60.0, help=">0向外（含红框），=0到内圈边，<0仅麦田")
    ap.add_argument("--debug", action="store_true", help="保存调试图")
    args = ap.parse_args()

    if os.path.isdir(args.in_path):
        files = sorted([f for f in glob.glob(os.path.join(args.in_path, "*")) if is_image(f)])
    else:
        files = [args.in_path]

    if not files:
        print("[ERR] 未找到图片"); sys.exit(1)
    from tqdm import tqdm
    for f in tqdm(files):
        try:
            process_one(f, args.out_dir, args.square, args.keep_px, args.debug)
        except Exception as e:
            print(f"[ERR] 处理失败 {f}: {e}")

if __name__ == "__main__":
    main()
