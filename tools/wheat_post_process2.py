import glob
import os.path

import cv2
import numpy as np
import math
import cv2
import numpy as np
from exceptiongroup import catch


def clean_small_islands_and_holes(mask, area_white_max=2_000):
    """
    清除小白点、填充小黑洞（按面积阈值）。
    - area_white_max:  小于该面积的白色连通域将被抹黑（去除白点/小白块）
    - area_black_max:  小于该面积的黑色连通域将被填白（填掉黑洞/小孔洞）
    """
    m = mask.copy().astype(np.uint8)

    # --- 1) 清除小白点：对白色连通域按面积阈值过滤 ---
    bw = (m > 0).astype(np.uint8)  # 白=1，黑=0
    num, labels, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    for i in range(1, num):  # 跳过背景0
        if stats[i, cv2.CC_STAT_AREA] < area_white_max:
            m[labels == i] = 0  # 抹黑

    # --- 2) 获取白色轮廓并处理：保留最大的两个轮廓，其余轮廓内部填充白色 ---
    contours, _ = cv2.findContours((m > 0).astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # 按面积排序轮廓
    contours_sorted = sorted(contours, key=cv2.contourArea, reverse=True)
    # 创建一个掩码，初始为与m相同
    mask_processed = m.copy()
    # 如果有超过2个轮廓，将除最大两个外的所有轮廓内部填充为白色
    if len(contours_sorted) > 2:
        # 创建一个临时掩码来绘制需要填充白色的区域
        mask_fill = np.zeros_like(m)
        # 绘制除了最大两个外的所有轮廓
        for i in range(2, len(contours_sorted)):
            cv2.drawContours(mask_fill, [contours_sorted[i]], -1, 255, -1)
        # 将这些区域填充为白色
        mask_processed[mask_fill > 0] = 255
    # 更新m
    m = mask_processed

    return m


def extract_red_frame_mask(img, debug=False):
    """
    提取红/粉边框的 mask。
    参数:
        img   : BGR 图 (cv2 风格)
        debug : True 时额外返回中间结果字典
    返回:
        mask  : np.uint8, 0/255，只有红色外框区域（内侧白线/麦田不在 mask 内）
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # --- 1) 红色在 HSV 上分为两个区间 ---
    # 这些阈值对偏粉/偏红都比较鲁棒，S/V 下限适度放宽以适应阴影与高光
    lo1, hi1 = (0,   60, 40), (10, 255,255)
    lo2, hi2 = (160, 60, 40), (180,255,255)

    mask1 = cv2.inRange(hsv, np.uint8(lo1), np.uint8(hi1))
    mask2 = cv2.inRange(hsv, np.uint8(lo2), np.uint8(hi2))
    mask  = cv2.bitwise_or(mask1, mask2)

    # --- 2) 轻度去噪，先进行10次开运算，再进行10次闭运算 ---
    mask = cv2.medianBlur(mask, 3)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    # 将最大轮廓和次大轮廓之间的区域填白
    # 获取轮廓
    # 获取轮廓
    contours, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours_sorted = sorted(contours, key=cv2.contourArea, reverse=True)

    # if len(contours_sorted) >= 1:
    #     # 创建一个掩码，初始为全黑
    #     mask_new = np.zeros_like(mask)
    #     # 只绘制最大轮廓内部为白色
    #     cv2.drawContours(mask_new, [contours_sorted[0]], -1, 255, -1)
    #     # 更新掩码，只保留最大轮廓
    #     mask = mask_new

    if len(contours_sorted) >= 2:
        # 创建一个掩码，初始为全黑
        mask_outer = np.zeros_like(mask)
        mask_inner = np.zeros_like(mask)
        # 绘制最大轮廓内部为白色
        cv2.drawContours(mask_outer, [contours_sorted[0]], -1, 255, -1)
        # 绘制次大轮廓内部为白色
        cv2.drawContours(mask_inner, [contours_sorted[1]], -1, 255, -1)
        # 最大轮廓内部减去次大轮廓内部，得到它们之间的区域
        mask = cv2.bitwise_and(mask_outer, cv2.bitwise_not(mask_inner))

    # 10次开运算
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=10)
    # 10次闭运算
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=10)

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=50)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=10)

    mask = clean_small_islands_and_holes(mask, area_white_max=1500)

    # 获取轮廓
    contours, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours_sorted = sorted(contours, key=cv2.contourArea, reverse=True)
    inside_contour = None
    inside_quad_vertices = None
    if len(contours_sorted) >= 2:
        inside_contour = contours_sorted[1]
        # 使用多边形近似找到四边形轮廓
        epsilon = 0.02 * cv2.arcLength(inside_contour, True)  # 精度参数
        approx = cv2.approxPolyDP(inside_contour, epsilon, True)

        # 如果近似出的点接近4个，则认为是四边形
        if len(approx) >= 3 and len(approx) <= 6:
            # 如果点数不等于4，强制取4个主要顶点
            if len(approx) != 4:
                # 找到轮廓的最小外接矩形
                rect = cv2.minAreaRect(inside_contour)
                # 获取矩形的四个顶点坐标
                inside_quad_vertices = np.int0(cv2.boxPoints(rect))
            else:
                inside_quad_vertices = approx.reshape(-1, 2)  # 转换为 [4,2] 形状的数组

            # 按照顺时针或逆时针排序顶点
            # 计算质心
            M = cv2.moments(inside_quad_vertices)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                # 根据点与质心的角度排序
                inside_quad_vertices = sorted(inside_quad_vertices, key=lambda p: np.arctan2(p[1]-cy, p[0]-cx))
                inside_quad_vertices = np.array(inside_quad_vertices, dtype=np.int32)

    if debug:
        return mask, {"mask1": mask1, "mask2": mask2}
    return mask, inside_quad_vertices

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

dataset_root = '/public/home/zncui/datasets/wheat-guo-dataset/wheat-guo-crop/'
img_paths = glob.glob(dataset_root + '*.jpg')
from tqdm import tqdm

for img_path in tqdm(img_paths):
    # if 'Li21.jpg' not in img_path:
    #     continue
    basename = os.path.basename(img_path)
    img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    mask, inside_quad_vertices = extract_red_frame_mask(img, debug=False)

    # # 将 mask 叠加在 img 上，显示为蓝色，并保存下来
    # img_masked = img.copy()
    # img_masked[mask > 0] = [255, 0, 0]  # 蓝色
    # # 将 inside_quad_vertices 画在 mask 上
    # if inside_quad_vertices is not None:
    #     # # 在 mask 上绘制白色边框
    #     # cv2.polylines(mask, [inside_quad_vertices], isClosed=True, color=255, thickness=8)
    #     # 在原始图像上绘制红色边框
    #     cv2.polylines(img_masked, [inside_quad_vertices], isClosed=True, color=(0, 0, 255), thickness=8)
    # cv2.imwrite(os.path.join('/public/home/zncui/datasets/wheat-guo-dataset/wheat-guo-square-error/result', basename), img_masked)

    quad = inside_quad_vertices
    keep_px = -60
    out_side = None

    # 根据 keep_px 外扩/内缩（>0 向外，<0 向内）
    quad_keep = expand_quad(quad, keep_px)
    out = warp_to_square(img, quad_keep, out_side)

    out_dir = "/public/home/zncui/datasets/wheat-guo-dataset/wheat-guo-square2"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, basename)
    cv2.imencode(".jpg", out)[1].tofile(out_path)