import os
import random
import shutil
import re
import cv2
from tqdm import tqdm

dataset_dir = '/public/home/zncui/datasets/wheat-guo-dataset/wheat-guo-crop-results2'
ori_dir = os.path.join(dataset_dir, 'ori')
# 绝对值 + x y w h + 左上角
filtered_txt_dir = os.path.join(dataset_dir, 'filtered_txt')
# 相对值 + x y w h + 中心
labelImg_annotation_dir = os.path.join(dataset_dir, 'labelimg_annotation')

# YOLO 格式的大图数据集目录
yolo_wheat_big_image_dir = os.path.join(dataset_dir, 'yolo_wheat_big_image')
os.makedirs(yolo_wheat_big_image_dir, exist_ok=True)

images_dir = os.path.join(yolo_wheat_big_image_dir, 'images')
train_images_dir = os.path.join(images_dir, 'train')
val_images_dir = os.path.join(images_dir, 'val')
os.makedirs(train_images_dir, exist_ok=True)
os.makedirs(val_images_dir, exist_ok=True)

labels_dir = os.path.join(yolo_wheat_big_image_dir, 'labels')
train_labels_dir = os.path.join(labels_dir, 'train')
val_labels_dir = os.path.join(labels_dir, 'val')
os.makedirs(train_labels_dir, exist_ok=True)
os.makedirs(val_labels_dir, exist_ok=True)


# 40 + 10
all_images = os.listdir(ori_dir)
all_images = [img for img in all_images if img.replace('.jpg', '_filtered.txt') in os.listdir(filtered_txt_dir)]
random.seed(42)  # 固定随机种子，确保实验可重复性
train_images = random.sample(all_images, int(len(all_images) * 0.8))
val_images = [img for img in all_images if img not in train_images]


# def copy_images_and_labels(image_list, pattern='train'):
#     for img_name in image_list:
#         ori_img_path = os.path.join(ori_dir, img_name)
#         target_img_path = os.path.join(images_dir, pattern, img_name)
#         shutil.copy2(ori_img_path, target_img_path)
#
#         filtered_txt_path = os.path.join(filtered_txt_dir, img_name.replace('.jpg', '_filtered.txt'))
#         labelimg_annotation_path = os.path.join(labelImg_annotation_dir, img_name.replace('.jpg', '_filtered.txt'))
#
#         import cv2
#         img = cv2.imread(ori_img_path)
#
#         yolo_lines = []
#         import re
#         p = re.compile('(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)')
#         with open(filtered_txt_path, 'r') as f:
#             lines = f.readlines()
#             for line in lines:
#                 m = p.match(line)
#                 if not m:
#                     continue
#                 index, top_left_x, top_left_y, w, h = map(int, m.groups())
#                 # 转换为 YOLO 格式：相对值 + 中心点坐标
#                 x_center = (top_left_x + w / 2) / img.shape[1]
#                 y_center = (top_left_y + h / 2) / img.shape[0]
#                 w_rel = w / img.shape[1]
#                 h_rel = h / img.shape[0]
#
#                 category_id = 0
#                 yolo_line = f"{category_id} {x_center} {y_center} {w_rel} {h_rel}\n"
#                 yolo_lines.append(yolo_line)
#         with open(labelimg_annotation_path, 'r') as f:
#             lines = f.readlines()
#             for line in lines:
#                 m = p.match(line)
#                 if not m:
#                     continue
#                 category_id, x, y, w, h = map(int, m.groups())
#                 yolo_line = f"{category_id} {x_center} {y_center} {w_rel} {h_rel}\n"
#                 yolo_lines.append(yolo_line)
#
#         # draw yolo_lines
#         img2 = img.copy()
#         for line in yolo_lines:
#             m = p.match(line)
#             if not m:
#                 continue
#             category_id, x_center, y_center, w_rel, h_rel = map(float, m.groups())
#             x_center = int(x_center * img.shape[1])
#             y_center = int(y_center * img.shape[0])
#             w_rel = int(w_rel * img.shape[1])
#             h_rel = int(h_rel * img.shape[0])
#             top_left_x = int(x_center - w_rel / 2)
#             top_left_y = int(y_center - h_rel / 2)
#             cv2.rectangle(img2, (top_left_x, top_left_y), (top_left_x + w_rel, top_left_y + h_rel), (0, 255, 0), 1)
#         save_img_dir = os.path.join(yolo_wheat_big_image_dir, 'save')
#         os.makedirs(save_img_dir, exist_ok=True)
#         save_img_path = os.path.join(save_img_dir, img_name)
#         cv2.imwrite(save_img_path, img2)
#
#         with open(os.path.join(labels_dir, pattern, img_name.replace('.jpg', '.txt')), 'w') as f:
#             f.writelines(yolo_lines)
# …（省略前面目录、随机划分等）…

def copy_images_and_labels(image_list, pattern='train'):
    for img_name in tqdm(image_list, desc=f'{pattern.capitalize()} images'):
        ori_img_path = os.path.join(ori_dir, img_name)
        target_img_path = os.path.join(images_dir, pattern, img_name)
        shutil.copy2(ori_img_path, target_img_path)

        img = cv2.imread(ori_img_path)
        height, width = img.shape[:2]

        yolo_lines = []

        # 1) 处理 filtered_txt
        filtered_txt_path = os.path.join(filtered_txt_dir, img_name.replace('.jpg', '_filtered.txt'))
        with open(filtered_txt_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                _, tlx, tly, w, h = map(int, parts)
                # 转浮点、归一化
                x_center = (tlx + w/2) / width
                y_center = (tly + h/2) / height
                w_rel = w / width
                h_rel = h / height
                yolo_lines.append(f"0 {x_center:.6f} {y_center:.6f} {w_rel:.6f} {h_rel:.6f}\n")

        # 2) 处理 labelImg_annotations
        labelimg_annotation_path = os.path.join(labelImg_annotation_dir, img_name.replace('.jpg', '_filtered.txt'))
        with open(labelimg_annotation_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cat_id, cx, cy, w, h = map(float, parts)
                # cx, cy, w, h 本身就是归一化的中心格式，不需要再计算，只需要保留
                yolo_lines.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")


        # 3) 写出标签文件
        out_label_path = os.path.join(labels_dir, pattern, img_name.replace('.jpg', '.txt'))
        with open(out_label_path, 'w') as f:
            f.writelines(yolo_lines)


        # 4) 可视化：直接 split 浮点
        img2 = img.copy()
        for line in yolo_lines:
            parts = line.strip().split()
            cat_id, x_c, y_c, w_r, h_r = map(float, parts)
            x_c_px = int(x_c * width)
            y_c_px = int(y_c * height)
            w_px = int(w_r * width)
            h_px = int(h_r * height)
            x0 = x_c_px - w_px // 2
            y0 = y_c_px - h_px // 2
            cv2.rectangle(img2, (x0, y0), (x0 + w_px, y0 + h_px), (0, 255, 0), 1)

        # 5) 保存可视化结果
        save_img_dir = os.path.join(yolo_wheat_big_image_dir, 'save')
        os.makedirs(save_img_dir, exist_ok=True)
        cv2.imwrite(os.path.join(save_img_dir, img_name), img2)

    out_class_path = os.path.join(labels_dir, pattern, 'class.txt')
    with open(out_class_path, 'w') as f:
        f.write('wheat\n')


# copy_images_and_labels(train_images, 'train')
# copy_images_and_labels(val_images, 'val')



def merge_yolo_wheat_big_image(out_dir):
    wheat_guo_dataset_dir = '/public/home/zncui/datasets/wheat-guo-dataset'
    p = re.compile('wheat-guo-crop-results\d*/yolo_wheat_big_image$')
    yolo_wheat_big_image_dirs = [root for root, _, _ in os.walk(wheat_guo_dataset_dir) if p.search(root)]
    img_train_val_list = []
    label_train_val_list = []
    for yolo_wheat_big_image_dir in yolo_wheat_big_image_dirs:
        for root, dirs, files in os.walk(yolo_wheat_big_image_dir):
            if 'save' in root:
                continue
            for file in files:
                if file.endswith('.jpg'):
                    img_train_val_list.append(os.path.join(root, file))
                elif file.endswith('.txt') and not file.startswith('class'):
                    label_train_val_list.append(os.path.join(root, file))
    img_train_val_list = sorted(img_train_val_list)
    label_train_val_list = sorted(label_train_val_list)
    img_train_list = [img for img in img_train_val_list if 'images/train' in img]
    img_val_list = [img for img in img_train_val_list if 'images/val' in img]
    label_train_list = [label for label in label_train_val_list if 'labels/train' in label]
    label_val_list = [label for label in label_train_val_list if 'labels/val' in label]
    os.makedirs(out_dir, exist_ok=True)
    train_images_dir = os.path.join(out_dir, 'images', 'train')
    val_images_dir = os.path.join(out_dir, 'images', 'val')
    train_labels_dir = os.path.join(out_dir, 'labels', 'train')
    val_labels_dir = os.path.join(out_dir, 'labels', 'val')
    os.makedirs(train_images_dir, exist_ok=True)
    os.makedirs(val_images_dir, exist_ok=True)
    os.makedirs(train_labels_dir, exist_ok=True)
    os.makedirs(val_labels_dir, exist_ok=True)
    for img in tqdm(img_train_list, desc='Copying train images'):
        shutil.copy2(img, train_images_dir)
    for img in tqdm(img_val_list, desc='Copying val images'):
        shutil.copy2(img, val_images_dir)
    for label in tqdm(label_train_list, desc='Copying train labels'):
        shutil.copy2(label, train_labels_dir)
    for label in tqdm(label_val_list, desc='Copying val labels'):
        shutil.copy2(label, val_labels_dir)
    out_class_path = os.path.join(out_dir, 'labels', 'train', 'class.txt')
    with open(out_class_path, 'w') as f:
        f.write('wheat\n')

merge_yolo_wheat_big_image('/public/home/zncui/datasets/wheat-guo-dataset/yolo_wheat_big_image')


