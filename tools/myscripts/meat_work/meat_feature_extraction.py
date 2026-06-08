import os
import re

import cv2
import glob
import pandas as pd
import numpy as np


INPUT_PATH = './data/meat.jpg'
INPUT_DIR = './data/bms_crops/'
OUTPUT_EXCEL = './data/bms_results.xlsx'

WHITE_DIR = './data/bms_white/'
RED_DIR = './data/bms_red/'
OUTIMG_DIR = './data/bms_out/'



def split_and_save_grid(
    image_path,
    out_dir,
    rows=3,
    cols=4,
    padding=0,
):
    os.makedirs(out_dir, exist_ok=True)

    img = cv2.imread(image_path)

    H, W = img.shape[:2]
    base_h = H // rows
    base_w = W // cols

    count = 0
    for r in range(rows):
        for c in range(cols):
            y1 = r * base_h
            x1 = c * base_w
            y2 = (r + 1) * base_h if r < rows - 1 else H
            x2 = (c + 1) * base_w if c < cols - 1 else W

            # padding 内缩
            y1p = max(y1 + padding, 0)
            x1p = max(x1 + padding, 0)
            y2p = max(min(y2 - padding, H), y1p + 1)
            x2p = max(min(x2 - padding, W), x1p + 1)

            crop = img[y1p:y2p, x1p:x2p].copy()

            idx = r * cols + c + 1
            out_path = os.path.join(out_dir, f"bms_{idx}.jpg")
            cv2.imwrite(out_path, crop)
            count += 1

    print(f"完成：已保存 {count} 张图片到目录：{out_dir}")


def get_red_region_mask(bgr_img,
                        hsv_h1=(0, 15), hsv_h2=(165, 180),
                        s_lo=40, v_lo=40,
                        close_ks=3, dilate_ks=3, erode_ks=3,
                        min_area=500):
    h_lo1, h_hi1 = hsv_h1
    h_lo2, h_hi2 = hsv_h2

    # --- 路径 A：HSV 双区间（更宽）---
    hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
    lower1 = np.array([h_lo1, s_lo, v_lo], dtype=np.uint8)
    upper1 = np.array([h_hi1, 255, 255], dtype=np.uint8)
    lower2 = np.array([h_lo2, s_lo, v_lo], dtype=np.uint8)
    upper2 = np.array([h_hi2, 255, 255], dtype=np.uint8)
    mask_hsv = cv2.inRange(hsv, lower1, upper1) | cv2.inRange(hsv, lower2, upper2)

    # --- 路径 B：Lab a通道（对红更敏感），Otsu 自动阈值 ---
    lab = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2LAB)
    a = lab[:, :, 1]  # OpenCV 的 Lab 已做过偏移(中心128)
    # Otsu 找阈值；阈上侧为更“红”
    _, mask_a = cv2.threshold(a, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 合并两路，得到“更广泛”的红区候选
    mask = cv2.bitwise_or(mask_hsv, mask_a)

    # 形态学：闭运算合并缝隙，再轻度膨胀抹平锯齿
    if close_ks > 0:
        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ks, close_ks))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close, iterations=1)
    if dilate_ks > 0:
        k_dil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_ks, dilate_ks))
        mask = cv2.dilate(mask, k_dil, iterations=1)

    # 填洞：取外轮廓并填充
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)

    # 只保留最大连通域（排除散落噪声）
    num, labels, stats, _ = cv2.connectedComponentsWithStats(filled, connectivity=8)
    if num > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        max_id = 1 + int(np.argmax(areas))
        big = np.zeros_like(filled)
        big[labels == max_id] = 255
        filled = big

    # 轻度腐蚀回退边缘，得到更贴合的整体轮廓
    if erode_ks > 0:
        k_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_ks, erode_ks))
        filled = cv2.erode(filled, k_erode, iterations=1)

    # 过滤过小目标（可选，通常不会触发，因为上一步已取最大连通域）
    if min_area > 0:
        if cv2.countNonZero(filled) < min_area:
            filled[:] = 0

    return filled

def get_bbox_from_img(img):
    # 提取黑色区域的掩码
    lower_black = (0, 0, 0)
    upper_black = (50, 50, 50)
    mask = cv2.inRange(img, lower_black, upper_black)

    # 开运算去噪，主要将文字部分去掉
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)

    # 找到白色区域的四个顶点，最左上角和最右下角
    coords = cv2.findNonZero(mask)
    x, y, w, h = cv2.boundingRect(coords)

    # 在原图上画出矩形框
    # img_boxed = img.copy()
    # cv2.rectangle(img_boxed, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # cv2.imwrite(os.path.join(RECT_DIR, filename), img_boxed)
    return x, y, w, h

def grey_threshold_extraction(img, full_red_mask, configs):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lower, upper = configs.get('grey_threshold', (165, 255))
    _, white_mask = cv2.threshold(gray, lower, upper, cv2.THRESH_BINARY)
    white_mask = cv2.bitwise_and(white_mask, full_red_mask)
    red_mask = cv2.bitwise_not(white_mask)
    red_mask = cv2.bitwise_and(red_mask, full_red_mask)

    return red_mask, white_mask

def red_hsv_extraction(img, full_red_mask, configs):

    alpha = configs.get('alpha', 1.8)
    beta = configs.get('beta', -75)
    enhanced = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    # 提取出 cropped 中红色图像
    hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)
    # 调整饱和度下限（第二个值）- 降低可以捕获更淡的红色
    lower_red1 = configs.get('lower_red1', (0, 100, 100)) # 从135降到100
    upper_red1 = configs.get('upper_red1', (10, 255, 255))
    lower_red2 = configs.get('lower_red2', (170, 100, 100))  # 从135降到100
    upper_red2 = configs.get('upper_red2', (180, 255, 255))
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)
    not_red_mask = cv2.bitwise_not(red_mask)
    white_mask = cv2.bitwise_and(full_red_mask, not_red_mask)
    return red_mask, white_mask

def _get_full_red_mask(cropped, filename, coordinates):
    x, y, w, h = coordinates
    full_red_mask = None
    match = re.search(r'bms_(\d+)\.jpg', filename)
    if match:
        idx = int(match.group(1))
        if idx in [1, 2, 3, 4, 5, 6, ]:
            # 提取出 cropped 中的完整红色区域
            full_red_mask = get_red_region_mask(cropped)

        elif idx in [7, 8, 9, 10, 11, 12]:
            template_img = cv2.imread(os.path.join('./data/template', f'bms_{idx}.jpg'))
            cropped_template_img = template_img[y:y + h, x:x + w]
            template_hsv = cv2.cvtColor(cropped_template_img, cv2.COLOR_BGR2HSV)
            lower_blue_strict = np.array([100, 120, 60])
            upper_blue_strict = np.array([140, 255, 255])
            lower, upper = lower_blue_strict, upper_blue_strict
            full_red_mask = blue_mask = cv2.inRange(template_hsv, lower, upper)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        full_red_mask = cv2.morphologyEx(full_red_mask, cv2.MORPH_OPEN, kernel, iterations=2)
        full_red_mask = cv2.erode(full_red_mask, kernel, iterations=3)
    return full_red_mask

def extract_black_box(img, filename, configs):
    pattern = configs.get('pattern', 'red_hsv')
    # 获取黑色区域的边界框
    coordinates = x, y, w, h = get_bbox_from_img(img)

    # 裁剪出矩形框内的图像
    cropped = img[y:y+h, x:x+w]

    full_red_mask = _get_full_red_mask(cropped, filename, coordinates)
    if pattern == 'red_hsv':
        red_mask, white_mask = red_hsv_extraction(cropped, full_red_mask, configs)
    elif pattern == 'grey_threshold':
        red_mask, white_mask = grey_threshold_extraction(cropped, full_red_mask, configs)

    # 还原
    red_board_mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    red_board_mask[y:y+h, x:x+w] = red_mask
    os.makedirs(RED_DIR, exist_ok=True)
    cv2.imwrite(os.path.join(RED_DIR, filename), red_board_mask)
    red_pixel_count = cv2.countNonZero(red_board_mask)

    white_board_mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    white_board_mask[y:y+h, x:x+w] = white_mask
    os.makedirs(WHITE_DIR, exist_ok=True)
    cv2.imwrite(os.path.join(WHITE_DIR, filename), white_board_mask)
    white_pixel_count = cv2.countNonZero(white_board_mask)

    img = img.copy()
    img[white_board_mask > 0] = (0, 255, 255)  # BGR中的黄色
    img[red_board_mask > 0] = (255, 0, 0)  # BGR中的蓝色
    os.makedirs(OUTIMG_DIR, exist_ok=True)
    cv2.imwrite(os.path.join(OUTIMG_DIR, filename), img)

    ratio = round(white_pixel_count / (white_pixel_count + red_pixel_count), 4) if white_pixel_count + red_pixel_count > 0 else 0

    print(f"Red pixel count: {red_pixel_count}, White pixel count after refinement: {white_pixel_count}, Ratio: {ratio:.4f}")

    return ratio

def load_yaml_config(config_path):
    import yaml

    def list_to_tuple(obj):
        if isinstance(obj, list):
            return tuple(list_to_tuple(i) for i in obj)
        elif isinstance(obj, dict):
            return {k: list_to_tuple(v) for k, v in obj.items()}
        else:
            return obj

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return list_to_tuple(config)


def main():
    configs = load_yaml_config('./config.yaml')
    split_and_save_grid(
        image_path=INPUT_PATH,
        out_dir=INPUT_DIR,
    )
    results = pd.DataFrame(columns=['filename', 'ratio'])
    img_list = glob.glob('./data/bms_crops/*.jpg')

    for img_path in img_list:
        print('Processing:', img_path)
        img = cv2.imread(img_path)
        filename = os.path.basename(img_path)
        Ata = ratio = extract_black_box(img, filename, configs)
        # class_value = 1.799 + 0.118*Ata + 0.005*Ata*Ata
        result_dict = {'filename': [filename], 'ratio': [ratio]}
        results = pd.concat([results, pd.DataFrame(result_dict)], ignore_index=True)
    results = results.sort_values(by='ratio', ascending=False)
    results.to_excel('./data/bms_results.xlsx', index=False)
    print(f"结果已保存到 './data/bms_results.xlsx'")

if __name__ == "__main__":
    main()