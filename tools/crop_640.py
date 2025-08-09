#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
import random

# 红色 HSV 范围
lower1, upper1 = np.array([0, 100, 50]),  np.array([10, 255, 255])
lower2, upper2 = np.array([160, 100, 50]), np.array([180, 255, 255])

def has_red_border(img, bw=5, min_pixels=20):
    """
    检测 img 四条边缘宽度为 bw 的区域内是否有至少 min_pixels 个红色像素。
    有则认为包含红框，应跳过该 patch。
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    m1 = cv2.inRange(hsv, lower1, upper1)
    m2 = cv2.inRange(hsv, lower2, upper2)
    mask = cv2.bitwise_or(m1, m2)

    # 提取四条边
    top    = mask[0:bw, :]
    bottom = mask[-bw:, :]
    left   = mask[:, 0:bw]
    right  = mask[:, -bw:]
    border_pixels = np.concatenate([
        top.ravel(),
        bottom.ravel(),
        left.ravel(),
        right.ravel()
    ])

    # 统计红色像素数
    return cv2.countNonZero(border_pixels) >= min_pixels


def main():
    source_dir = '/public/home/zncui/datasets/wheat-guo-crop'
    target_dir = '/public/home/zncui/datasets/wheat-guo-crop-640'
    os.makedirs(target_dir, exist_ok=True)

    sample_files = random.sample(
        [f for f in os.listdir(source_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))],
        100
    )

    file_list = []
    for filename in sample_files:
        source_path = os.path.join(source_dir, filename)
        img = cv2.imread(source_path)
        if img is None:
            print(f"[WARN] 读图失败：{filename}")
            continue

        h, w = img.shape[:2]
        overlap_ratio = 0.2
        window_size = 640
        stride = int(window_size * (1 - overlap_ratio))

        for y in range(0, h, stride):
            for x in range(0, w, stride):
                y_end = min(y + window_size, h)
                x_end = min(x + window_size, w)
                cropped = img[y:y_end, x:x_end]

                # 跳过包含红色边框的 patch
                if has_red_border(cropped, bw=5, min_pixels=20):
                    continue

                save_name = f"{os.path.splitext(filename)[0]}_{y}_{x}.jpg"
                target_path = os.path.join(target_dir, save_name)
                file_list.append({
                    'target_path': target_path,
                    'data': cropped
                })

    # 从候选 patches 中随机选 200 张保存
    if len(file_list) < 200:
        print(f"[WARN] 候选 patch 数量 {len(file_list)} 少于 200，全部保存。")
        final_samples = file_list
    else:
        final_samples = random.sample(file_list, 200)

    for item in final_samples:
        cv2.imwrite(item['target_path'], item['data'])
        print(f"[OK] 保存 {item['target_path']}")

if __name__ == '__main__':
    main()
