import os

import cv2
import numpy as np
import re
from tqdm import tqdm


def get_red_points(image_path, red_border):
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

    # 过滤掉红色点在红框内的情况
    filtered_red_points = []
    for rx, ry in red_points:
        if not (red_border[ry, rx] > 0):
            filtered_red_points.append((rx, ry))
    return red_points


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
    # 消除小噪声
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=10)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=10)
    # 填充空洞
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=30)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=30)

    return mask

def remove_coordinates_by_red_points(coordinates, red_points):
    filtered = []
    pattern = re.compile(r'(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)')
    for coord in coordinates:
        m = pattern.match(coord)
        if not m:
            continue
        idx, x, y, w, h = map(int, m.groups())
        x1, y1, x2, y2 = x, y, x + w, y + h

        # 如果任意红点落在框内，就跳过（去除）这个框
        if any(x1 <= rx <= x2 and y1 <= ry <= y2 for rx, ry in red_points):
            continue

        filtered.append(coord)
    return filtered


def filter_coordinates_inside_red_border(coordinates, red_border):
    filtered = []
    pattern = re.compile(r'(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)')
    for coord in coordinates:
        m = pattern.match(coord)
        if not m:
            continue
        idx, x, y, w, h = map(int, m.groups())
        x1, y1, x2, y2 = x, y, x + w, y + h

        # 检查矩形框的四个角是否都在红色边框内
        corners = [(x1, y1), (x1, y2), (x2, y1), (x2, y2)]
        # 如果所有角都在红色边框内（值>0），则认为矩形完全在红色框内
        if all(red_border[cy, cx] > 0 for cx, cy in corners):
            continue

        filtered.append(coord)
    return filtered

if __name__ == '__main__':

    dataset_dir = '/public/home/zncui/datasets/wheat-guo-dataset/wheat-guo-crop-results2'
    grid_dir = os.path.join(dataset_dir, 'krita')
    txt_dir = os.path.join(dataset_dir, 'txt')
    ori_dir = os.path.join(dataset_dir, 'ori')
    filtered_txt_dir = os.path.join(dataset_dir, 'filtered_txt')
    os.makedirs(filtered_txt_dir, exist_ok=True)
    filtered_index_dir = os.path.join(dataset_dir, 'filtered_grid')
    os.makedirs(filtered_index_dir, exist_ok=True)

    for filename in tqdm(os.listdir(grid_dir)):
        if filename.endswith('.jpg'):
            image_path = os.path.join(grid_dir, filename)
            ori_image_path = os.path.join(ori_dir, filename)
            txt_path = os.path.join(txt_dir, filename.replace('.jpg', '.txt'))
            filtered_txt_path = os.path.join(filtered_txt_dir, filename.replace('.jpg', '_filtered.txt'))
            filtered_index_path = os.path.join(filtered_index_dir, filename.replace('.jpg', '_filtered.jpg'))

            with open(txt_path, 'r') as f:
                coordinates = f.read()
                coordinates = coordinates.strip().split('\n')
                red_border = get_red_border(ori_image_path)
                # 过滤掉矩形框完全在红色框内的坐标
                coordinates = filter_coordinates_inside_red_border(coordinates, red_border)
                red_points = get_red_points(image_path, red_border)
                filtered_coordinates = remove_coordinates_by_red_points(coordinates, red_points)
                if filtered_coordinates:
                    img = cv2.imread(ori_image_path)
                    img2  = img.copy()

                    for coord in filtered_coordinates:
                        match = re.match(r'(\d+) (\d+) (\d+) (\d+) (\d+)', coord)
                        if match:
                            index, x, y, w, h = map(int, match.groups())
                            cv2.rectangle(img2, (x, y), (x + w, y + h), (0, 255, 0), 1)
                            cv2.putText(img2, str(index), (x, max(y - 5, 0)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, lineType=cv2.LINE_AA)


                    for red_point in red_points:
                        cv2.circle(img2, red_point, 8, (0, 0, 255), -1)

                    h, w = img2.shape[:2]
                    grid_size = (3, 3)
                    dx, dy = w // grid_size[1], h // grid_size[0]
                    # 垂直线
                    for i in range(grid_size[1]):
                        x = i * dx
                        cv2.line(img2, (x, 0), (x, h), (255, 0, 0), 3)
                    # 水平线
                    for j in range(grid_size[0]):
                        y = j * dy
                        cv2.line(img2, (0, y), (w, y), (255, 0, 0), 3)
                    cv2.imwrite(filtered_index_path, img2)

            with open(filtered_txt_path, 'w') as f:
                for coord in filtered_coordinates:
                    f.write(coord + '\n')





