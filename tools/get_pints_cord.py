import os
import cv2
import numpy as np
import re

def get_red_points(image_path, coordinates):
    '''
    1) HSV 空间提取红色（包含两段 hue）
    2) 开闭运算去噪
    3) 找最大轮廓（红色框），填充为 0（移除）
    4) 重新找轮廓，按面积过滤，计算中心
    '''
    # 1. 读图并转 HSV
    img = cv2.imread(image_path)
    # mask = img[:,:,2] == 255  # 直接使用 BGR 空间的红色通道阈值
    # mask = mask.astype(np.uint8) * 255  # 转为二值图
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 2. 红色在 H 通道两侧：分别阈值后或运算
    delta_h = 10
    lower1 = np.array([0, 180, 180], dtype=np.uint8)
    upper1 = np.array([delta_h, 255, 255], dtype=np.uint8)
    lower2 = np.array([180 - delta_h, 180, 180], dtype=np.uint8)
    upper2 = np.array([180, 255, 255], dtype=np.uint8)

    mask1 = cv2.inRange(img_hsv, lower1, upper1)
    mask2 = cv2.inRange(img_hsv, lower2, upper2)
    mask = cv2.bitwise_or(mask1, mask2)

    # 3. 开闭运算去小噪声、矩形连接
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    # mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    # 4. 找出所有轮廓，按照面积降序排列，认为最大的就是相框／边框，移除之
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        frame_cnt = contours[0]
        cv2.drawContours(mask, [frame_cnt], -1, 0, thickness=cv2.FILLED)

    # 5. 重新找轮廓，只保留小区域（红色点）
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    MIN_AREA = 1
    MAX_AREA = 800
    red_points = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if MIN_AREA < area < MAX_AREA:
            M = cv2.moments(cnt)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                red_points.append((cx, cy))

    pattern = re.compile(r'(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)')
    filtered_red_points = []
    for rx, ry in red_points:
        flag = False
        for coord in coordinates:
            m = pattern.match(coord)
            if not m:
                continue
            idx, x, y, w, h = map(int, m.groups())
            x1, y1, x2, y2 = x, y, x + w, y + h

            # 如果任意红点落在框内，就跳过（去除）这个框
            if x1 <= rx <= x2 and y1 <= ry <= y2:
                flag = True
        if flag:
            filtered_red_points.append((rx, ry))
    return filtered_red_points

def get_red_border(image_path):
    img = cv2.imread(image_path)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # 红色在 H 通道两侧：分别阈值后或运算
    delta_h = 10
    lower1 = np.array([0, 100, 50], dtype=np.uint8)
    upper1 = np.array([delta_h, 255, 255], dtype=np.uint8)
    lower2 = np.array([180 - delta_h, 100, 50], dtype=np.uint8)
    upper2 = np.array([180, 255, 255], dtype=np.uint8)
    mask1 = cv2.inRange(img_hsv, lower1, upper1)
    mask2 = cv2.inRange(img_hsv, lower2, upper2)
    mask = cv2.bitwise_or(mask1, mask2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=10)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=10)

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=30)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=30)

    return mask

ori_image_path = '/public/home/zncui/datasets/wheat-guo-dataset/wheat-guo-crop-results/ori/21CA05.jpg'
red_border = get_red_border(ori_image_path)

image_path = '/public/home/zncui/datasets/wheat-guo-dataset/wheat-guo-crop-results/krita/21CA05.jpg'
img = cv2.imread(image_path)
if img is None:
    raise FileNotFoundError("无法读取图片，请检查路径。")

# 获取红色点坐标
with open('/public/home/zncui/datasets/wheat-guo-dataset/wheat-guo-crop-results/txt/21CA05.txt', 'r') as f:
    coordinates = f.readlines()
    coordinates = [coord.strip() for coord in coordinates if coord.strip()]  # 去除空
    dot_centers = get_red_points(image_path, coordinates)

# 6. 在原图上绘制蓝色实心圆，radius 可调
result = img.copy()
radius = 2   # <--- 修改这里来调整“点”的大小
for (x, y) in dot_centers:
    cv2.circle(result, (x, y), radius, (255, 0, 0), thickness=20)
# 将 red_border 区域涂成蓝色，叠加在 result 上
# for y in range(red_border.shape[0]):
#     for x in range(red_border.shape[1]):
#         if red_border[y, x] > 0:  # 如果是红色区域
#             result[y, x] = (255, 0, 0)  # 涂成蓝色
# 7. 保存结果
output_path = os.path.join('/public/home/zncui/datasets/wheat-guo-dataset/wheat-guo-crop-results/', os.path.basename(image_path))
cv2.imwrite(output_path, result)
print(f"已保存涂蓝后的图像（radius={radius}）：", output_path)
