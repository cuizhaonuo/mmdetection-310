#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自动检测白色细边框四角，将中间麦田透视为正方形。
- 先用“白边框分割”找四角；失败或四边形不合法时回退到“直线法”。
- 透视前做形状体检，避免出现自交/极瘦四边形导致的“V字黑屏”。
- 可用 keep_px 外扩/内缩以保留红框或仅取麦田。
"""

import os, sys, glob, math, argparse
import cv2
import numpy as np

# ----------------------- 工具函数 -----------------------
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

# ----------------------- 1) 白边框检测 -----------------------
def detect_quad_via_white(img):
    """
    低饱和高亮阈值 + 顶帽增强 + 形态学闭运算，找白色细边框。
    返回：quad, mask_raw, mask_thick
    """
    H, W = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    S = hsv[:,:,1]; V = hsv[:,:,2]

    # 自适应阈值（随曝光变化）
    s_thr = int(np.clip(np.median(S)*0.9, 25, 80))
    v_thr = int(np.clip(np.percentile(V, 75), 150, 245))

    # 基础白色掩码 + 顶帽增强细亮边
    mask_white = cv2.inRange(hsv, (0, 0, v_thr-25), (180, s_thr, 255))
    # k = max(3, int(min(H,W)*0.008))  # 略大一点，连接细白边
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
        if cv2.contourArea(c) < 0.002*H*W:  # 过滤小目标（文字标签等）
            continue
        approx = approx_to_four(c)
        if approx is None:
            rect = cv2.minAreaRect(c)
            approx = cv2.boxPoints(rect).astype(np.float32)
        sc = score_quad(approx, H, W)
        if sc > best_score:
            best = approx; best_score = sc
    return best, mask, mask_thick

# ----------------------- 2) 直线回退（改进） -----------------------
def detect_quad_via_lines(img):
    """
    改进版：将直线按角度聚成两簇（约90°），统一法向量指向画面外，
    各簇取“最外侧两条”，再两两求交点得到 tl,tr,br,bl。
    """
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
    lines = lines[:,0,:]  # (x1,y1,x2,y2)

    # 角度归一化到 [0, pi)
    angs = np.arctan2(lines[:,3]-lines[:,1], lines[:,2]-lines[:,0])
    angs = (angs + np.pi) % np.pi

    # 分成两簇（相差约90°）
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
        # 法向量统一“指向画面外”（远离中心），便于按距离排序
        s = a*cx + b*cy + c  # 中心点代入：>0表示中心在法向同侧    
        if s < 0:
            a,b,c = -a,-b,-c
        d = -c  # 到原点距离（归一化），同号可直接比较
        return (a,b,c,d)

    def pick_two(ls_idx):
        abcds = np.array([canon_line(lines[i]) for i in ls_idx])
        idx = np.argsort(abcds[:,3])
        return abcds[idx[0]], abcds[idx[-1]]  # 最外侧两条

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

# ----------------------- 3) 统一检测入口 -----------------------
def detect_quad(img):
    """先白边；若无或不合法，再用直线回退"""
    quad_w, mask_w, mask_thick = detect_quad_via_white(img)
    debug = {'mask_white': mask_w, 'mask_white_thick': mask_thick, 'edges': None, 'mode': None}

    if quad_w is not None and is_valid_quad(quad_w):
        debug['mode'] = 'white'
        return quad_w, debug, 'white'

    # 回退：线段
    quad_l, edges = detect_quad_via_lines(img)
    debug['edges'] = edges
    if quad_l is not None and is_valid_quad(quad_l):
        debug['mode'] = 'lines'
        return quad_l, debug, 'lines'

    # 都失败
    debug['mode'] = 'fail'
    return None, debug, 'fail'



# ----------------------- 4) 处理与可视化 -----------------------
def process_one(img_path, out_dir, out_side=None, keep_px=60, write_debug=False):
    base = os.path.splitext(os.path.basename(img_path))[0]
    img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        print(f"[WARN] 读图失败: {img_path}")
        return
    # img = paint_tag_red(img)
    # cv2.imwrite(os.path.join(out_dir, "output.jpg"), img)

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

        # 保存标记的原图
        dbg_path = os.path.join(out_dir, f"{base}_debug_marked.jpg")
        cv2.imencode(".jpg", dbg_img)[1].tofile(dbg_path)

        # 2. 保存各个调试图像
        if dbg.get('mask_white') is not None:
            mw_path = os.path.join(out_dir, f"{base}_debug_mask_white.jpg")
            cv2.imencode(".jpg", dbg['mask_white'])[1].tofile(mw_path)

            mt_path = os.path.join(out_dir, f"{base}_debug_mask_thick.jpg")
            cv2.imencode(".jpg", dbg['mask_white_thick'])[1].tofile(mt_path)

        if dbg.get('edges') is not None:
            ed_path = os.path.join(out_dir, f"{base}_debug_edges.jpg")
            cv2.imencode(".jpg", dbg['edges'])[1].tofile(ed_path)

        # 3. 左上角拼接调试小图（保持原功能）
        tiles = []
        if dbg.get('mask_white') is not None:
            mw = cv2.cvtColor(dbg['mask_white'], cv2.COLOR_GRAY2BGR)
            mt = cv2.cvtColor(dbg['mask_white_thick'], cv2.COLOR_GRAY2BGR)
            tiles += [mw, mt]
        if dbg.get('edges') is not None:
            ed = cv2.cvtColor(dbg['edges'], cv2.COLOR_GRAY2BGR)
            tiles += [ed]
        if tiles:
            tiles = [cv2.resize(t, (t.shape[1]//2, t.shape[0]//2)) for t in tiles]
            stack = np.hstack(tiles)
            h, w = stack.shape[:2]
            dbg_img_combined = img.copy()
            dbg_img_combined[0:h, 0:w] = stack
            # 在组合图上也绘制四边形标记
            cv2.polylines(dbg_img_combined, [q0], True, (0,255,0), 4)
            cv2.polylines(dbg_img_combined, [q1], True, (255,0,0), 4)
            for p in q0: cv2.circle(dbg_img_combined, tuple(p), 12, (0,255,0), -1)
            for p in q1: cv2.circle(dbg_img_combined, tuple(p), 12, (255,0,0), -1)

            dbg_path = os.path.join(out_dir, f"{base}_debug_combined.jpg")
            cv2.imencode(".jpg", dbg_img_combined)[1].tofile(dbg_path)

    print(f"[OK:{mode}] {img_path} -> {out_path}")

def is_image(f):
    ext = os.path.splitext(f)[1].lower()
    return ext in [".jpg",".jpeg",".png",".bmp",".tif",".tiff",".webp"]

def main():
    ap = argparse.ArgumentParser(description="自动检测白框四角并将中间麦田透视为正方形")
    ap.add_argument("--in", dest="in_path", required=True, help="输入文件或目录")
    ap.add_argument("--out", dest="out_dir", required=True, help="输出目录")
    ap.add_argument("--square", type=int, default=None, help="输出正方形边长（像素），默认自适应")
    ap.add_argument("--keep-px", type=float, default=60.0, help="从白框向外保留像素（>0含红框，=0到白框，<0只麦田）")
    ap.add_argument("--debug", action="store_true", help="保存调试图")
    args = ap.parse_args()

    if os.path.isdir(args.in_path):
        files = sorted([f for f in glob.glob(os.path.join(args.in_path, "*")) if is_image(f)])
    else:
        files = [args.in_path]

    if not files:
        print("[ERR] 未找到图片"); sys.exit(1)

    for f in files:
        try:
            process_one(f, args.out_dir, args.square, args.keep_px, args.debug)
        except Exception as e:
            print(f"[ERR] 处理失败 {f}: {e}")

# def create_error_img():
#     dataset_dir = '/public/home/zncui/datasets/wheat-guo-dataset'
#     error_square_dir = os.path.join(dataset_dir, 'wheat-guo-square-error')
#     error_ori_dir = os.path.join(dataset_dir, 'wheat-guo-crop')
#     target_dir = os.path.join(dataset_dir, 'wheat-guo-square-error/ori')
#     error_img_name = [f for f in os.listdir(error_square_dir) if is_image(f)]
#     os.makedirs(target_dir, exist_ok=True)
#
#     import shutil
#     for img_name in error_img_name:
#         shutil.copy(os.path.join(error_ori_dir, img_name), os.path.join(target_dir, img_name))


if __name__ == "__main__":
    # create_error_img()
    main()
