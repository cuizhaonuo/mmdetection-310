import os
import cv2
from tqdm import tqdm
# 输入／输出目录
input_dir = '/public/home/zncui/datasets/wheat-guo-dataset/imgs_and_labels/val_imgs'
output_dir = '/public/home/zncui/datasets/wheat-guo-dataset/imgs_and_labels/val_imgs'
os.makedirs(output_dir, exist_ok=True)

# 读取第一张 JPG
image_paths = sorted(
    [os.path.join(input_dir, f)
     for f in os.listdir(input_dir)
     if f.lower().endswith('.jpg')]
)

for image_path in tqdm(image_paths):
    # BGR 格式读取
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"无法读取图片：{image_path}")

    # 在副本上画网格线
    img2 = img.copy()
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

    # 直接以 BGR 格式写出，保持颜色一致
    output_path = os.path.join(output_dir, os.path.basename(image_path))
    cv2.imwrite(output_path, img2)
