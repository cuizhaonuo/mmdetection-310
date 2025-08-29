import os
import glob
import json
from pathlib import Path
from PIL import Image
import shutil
from tqdm import tqdm

dataset_dir = '/public/home/zncui/datasets/wheat-guo-dataset/yolo_wheat_big_image_v2split'

train_img_list = sorted(glob.glob(os.path.join(dataset_dir, 'images', 'train', '*.jpg')))
val_img_list = sorted(glob.glob(os.path.join(dataset_dir, 'images', 'val', '*.jpg')))

train_label_list = sorted(glob.glob(os.path.join(dataset_dir, 'labels', 'train', '*.txt')))
val_label_list = sorted(glob.glob(os.path.join(dataset_dir, 'labels', 'val', '*.txt')))

target_dir = '/public/home/zncui/datasets/wheat-guo-dataset/yolo_wheat_1024x1024_p'
target_train_img_dir = os.path.join(target_dir, 'images', 'train')
target_val_img_dir = os.path.join(target_dir, 'images', 'val')
target_train_label_dir = os.path.join(target_dir, 'labels', 'train')
target_val_label_dir = os.path.join(target_dir, 'labels', 'val')
os.makedirs(target_val_label_dir, exist_ok=True)

# ==== 配置区 ====
YOLO_ROOT = "/public/home/zncui/datasets/wheat-guo-dataset/yolo_wheat_1024x1024_p"
OUTPUT_SUBDIR = "coco2017"  # 会在 YOLO_ROOT 下创建 coco2017/
# =================

WINDOW_SIZE = 1024  # 裁剪窗口大小
OVERLAP_RATIO = 0.2  # 重叠比例

def check_box_intersection(box1, box2):
    """
    检查两个边界框是否有交集。
    box1 和 box2 格式为 [x1, y1, x2, y2]，其中 (x1, y1) 是左上角坐标，(x2, y2) 是右下角坐标。
    """
    return not (box1[0] > box2[2] or box1[2] < box2[0] or
                box1[1] > box2[3] or box1[3] < box2[1])


def crop_img_and_label(img_path, label_path, window_size=WINDOW_SIZE, overlap_ratio=OVERLAP_RATIO):
    import cv2
    import numpy as np

    # 读取图像
    img = cv2.imread(img_path)
    if img is None:
        print(f"[WARN] 读图失败：{img_path}")
        return []

    # 读取YOLO格式标签文件
    labels = []
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    # YOLO格式: class x_center y_center width height
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    labels.append([class_id, x_center, y_center, width, height])

    h, w = img.shape[:2]
    stride = int(window_size * (1 - overlap_ratio))
    crops = []

    # 对图像进行裁剪
    for y in range(0, h, stride):
        for x in range(0, w, stride):
            y_end = min(y + window_size, h)
            x_end = min(x + window_size, w)

            # 裁剪图像
            cropped_img = img[y:y_end, x:x_end]
            crop_h, crop_w = cropped_img.shape[:2]

            # 处理标签
            cropped_labels = []
            for label in labels:
                class_id, x_c, y_c, w_box, h_box = label

                # 转换为绝对坐标
                abs_x_c = x_c * w
                abs_y_c = y_c * h
                abs_w = w_box * w
                abs_h = h_box * h

                # 计算边界框的绝对坐标
                x1 = abs_x_c - abs_w / 2
                y1 = abs_y_c - abs_h / 2
                x2 = abs_x_c + abs_w / 2
                y2 = abs_y_c + abs_h / 2


                box1 = [x1, y1, x2, y2]
                box2 = [x, y, x_end, y_end]

                def intersection_area(box1, box2):
                    dx = max(0, min(box1[2], box2[2]) - max(box1[0], box2[0]))
                    dy = max(0, min(box1[3], box2[3]) - max(box1[1], box2[1]))
                    return dx * dy

                orig_area = (x2 - x1) * (y2 - y1)
                inter_area = intersection_area(box1, box2)
                if inter_area / orig_area < 0.05:
                    continue  # 剪掉太多了，丢掉

                # 检查边界框是否与裁剪区域有交集（包括边界情况）
                if check_box_intersection(box1, box2):
                    # 裁剪边界框，将超出裁剪区域的部分截断
                    # 可以理解为坐标移动
                    nx1 = max(x1, x) - x
                    ny1 = max(y1, y) - y
                    nx2 = min(x2, x_end) - x
                    ny2 = min(y2, y_end) - y

                    boxh = ny2 - ny1
                    boxw = nx2 - nx1
                    # 计算边界框被裁剪的程度（可选，用于调试）
                    # original_area = abs_w * abs_h
                    # cropped_area = (nx2 - nx1) * (ny2 - ny1)
                    # crop_ratio = cropped_area / original_area

                    # 转换回YOLO格式 (相对于裁剪后的图像)
                    nx_c = (nx1 + nx2) / 2 / crop_w
                    ny_c = (ny1 + ny2) / 2 / crop_h
                    nw = (nx2 - nx1) / crop_w
                    nh = (ny2 - ny1) / crop_h

                    # 只保存有效的边界框（面积足够大）
                    # if nw > 0.001 and nh > 0.001:
                    if boxh >=10 and boxw >= 10:
                        cropped_labels.append([class_id, nx_c, ny_c, nw, nh])

            # 只有当裁剪后的图像包含至少一个物体时才添加
            if len(cropped_labels) > 0:
                crops.append((cropped_img, (y, x), cropped_labels))

    return crops

def process_dataset(img_list, label_list, target_img_dir, target_label_dir):
    import cv2
    import os

    os.makedirs(target_img_dir, exist_ok=True)
    os.makedirs(target_label_dir, exist_ok=True)

    for img_path, label_path in tqdm(zip(img_list, label_list), desc="处理图像", total=len(img_list)):
        crops = crop_img_and_label(img_path, label_path)

        for i, (cropped_img, (y, x), cropped_labels) in enumerate(crops):
            # 保存裁剪后的图像
            img_name = f"{os.path.splitext(os.path.basename(img_path))[0]}_{y}_{x}_{i}.jpg"
            target_img_path = os.path.join(target_img_dir, img_name)
            cv2.imwrite(target_img_path, cropped_img)

            # 保存裁剪后的标签
            label_name = f"{os.path.splitext(os.path.basename(label_path))[0]}_{y}_{x}_{i}.txt"
            target_label_path = os.path.join(target_label_dir, label_name)
            with open(target_label_path, 'w') as f:
                for label in cropped_labels:
                    f.write(' '.join(map(str, label)) + '\n')

            # 如果直接在 cropped_img 上绘制标签会导致图像被修改
            img = cv2.imread(target_img_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # 将标签绘制到图像上
            for index, label in enumerate(cropped_labels):
                class_id, x_c, y_c, w_box, h_box = label
                x1 = int((x_c - w_box / 2) * img_rgb.shape[1])
                y1 = int((y_c - h_box / 2) * img_rgb.shape[0])
                x2 = int((x_c + w_box / 2) * img_rgb.shape[1])
                y2 = int((y_c + h_box / 2) * img_rgb.shape[0])
                cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img_rgb, str(index), (x1, max(y1 - 5, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            # 保存绘制了标签的图像（可选）
            save_img_path = target_img_path.replace('images', 'save').replace('.jpg', '_label.jpg')
            os.makedirs(os.path.dirname(save_img_path), exist_ok=True)
            cv2.imwrite(save_img_path, img_bgr)
    with open(os.path.join(target_train_label_dir, 'classes.txt'), 'w') as f:
        f.write('wheat\n')
    with open(os.path.join(target_val_label_dir, 'classes.txt'), 'w') as f:
        f.write('wheat\n')



def load_class_names(root):
    classes_txt = os.path.join(root, 'labels', 'train', "classes.txt")
    if os.path.isfile(classes_txt):
        with open(classes_txt, "r", encoding="utf-8") as f:
            names = [line.strip() for line in f if line.strip()]
        if len(names) == 0:
            raise RuntimeError(f"classes.txt 存在但没有类名: {classes_txt}")
        return names
    # 否则从 label 文件里搜 class id 生成 placeholder 名
    label_paths = glob.glob(os.path.join(root, "labels", "**", "*.txt"), recursive=True)
    class_ids = set()
    for lp in label_paths:
        with open(lp, "r") as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                class_ids.add(int(parts[0]))
    if not class_ids:
        raise RuntimeError("找不到任何 label 中的 class id，也没有 classes.txt")
    max_id = max(class_ids)
    # 生成 class_0, class_1 ... 直到最大
    names = [f"class_{i}" for i in range(max_id + 1)]
    return names

def yolo_line_to_coco_bbox(yolo_line, img_w, img_h):
    """
    YOLO: class x_center y_center width height (all normalized 0~1)
    COCO bbox: [x_min, y_min, width, height] in absolute pixels
    """
    parts = yolo_line.strip().split()
    if len(parts) < 5:
        return None
    class_id = int(parts[0])
    x_c = float(parts[1]) * img_w
    y_c = float(parts[2]) * img_h
    w = float(parts[3]) * img_w
    h = float(parts[4]) * img_h

    x_min = x_c - w / 2
    y_min = y_c - h / 2

    # clip to image bounds
    x_min = max(0.0, x_min)
    y_min = max(0.0, y_min)
    w = min(w, img_w - x_min)
    h = min(h, img_h - y_min)

    return class_id, [x_min, y_min, w, h]

def build_coco_annotation(split, image_dir, label_dir, categories, classid2catid, out_image_folder):
    """
    split: 'train' or 'val'
    image_dir: 原始 yolo images/{split}/
    label_dir: 原始 yolo labels/{split}/
    categories: list of dicts for coco categories
    classid2catid: mapping from yolo class index to coco category_id
    out_image_folder: 目标 images/train2017 或 val2017 路径（会复制图片到这里）
    """
    images = []
    annotations = []
    ann_id = 1
    img_id = 1

    img_patterns = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
    img_paths = []
    for pat in img_patterns:
        img_paths.extend(glob.glob(os.path.join(image_dir, pat)))
    img_paths = sorted(img_paths)
    if len(img_paths) == 0:
        print(f"[WARN] {split} 目录下没找到图片: {image_dir}")
    for img_path in tqdm(img_paths, desc=f"构造 {split} COCO 数据"):
        # 读取尺寸
        try:
            with Image.open(img_path) as im:
                img_w, img_h = im.size
        except Exception as e:
            print(f"[ERROR] 不能打开图片 {img_path}: {e}")
            continue

        # 复制图片到 COCO 目录
        os.makedirs(out_image_folder, exist_ok=True)
        dst_fname = os.path.basename(img_path)
        dst_path = os.path.join(out_image_folder, dst_fname)
        shutil.copy2(img_path, dst_path)

        # image entry
        image_info = {
            "id": img_id,
            "width": img_w,
            "height": img_h,
            # COCO 中 file_name 通常相对于 images/ 下的路径（保持原名即可）
            "file_name": dst_fname,
        }
        images.append(image_info)

        # 读取对应 label
        base = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(label_dir, f"{base}.txt")
        if os.path.isfile(label_path):
            with open(label_path, "r") as f:
                for line in f:
                    parsed = yolo_line_to_coco_bbox(line, img_w, img_h)
                    if parsed is None:
                        continue
                    class_id, bbox = parsed
                    if class_id not in classid2catid:
                        # 跳过未知类
                        continue
                    coco_cat_id = classid2catid[class_id]
                    x_min, y_min, bw, bh = bbox
                    area = bw * bh
                    annotation = {
                        "id": ann_id,
                        "image_id": img_id,
                        "category_id": coco_cat_id,
                        "bbox": [x_min, y_min, bw, bh],
                        "area": area,
                        "iscrowd": 0,
                        "segmentation": [],  # 如果没有 segmentation 则空
                    }
                    annotations.append(annotation)
                    ann_id += 1
        # else: 没有 label 就是空 annotation

        img_id += 1

    coco_dict = {
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }
    return coco_dict

def yolo2coco():
    root = YOLO_ROOT
    coco_root = os.path.join(root, OUTPUT_SUBDIR)
    images_out_root = coco_root
    ann_out_dir = os.path.join(coco_root, "annotations")
    os.makedirs(images_out_root, exist_ok=True)
    os.makedirs(ann_out_dir, exist_ok=True)

    # 读取类名，构造 categories
    class_names = load_class_names(root)
    categories = []
    classid2catid = {}
    for idx, name in enumerate(class_names):
        coco_id = idx + 1  # COCO 要求 category_id 从 1 开始
        categories.append({
            "id": coco_id,
            "name": name,
            "supercategory": "none"
        })
        classid2catid[idx] = coco_id  # YOLO 里的 class idx 映射到 COCO id

    # 处理 train 和 val
    splits = [("train", os.path.join(root, "images", "train"), os.path.join(root, "labels", "train"),
               os.path.join(images_out_root, "train2017"), os.path.join(ann_out_dir, "instances_train2017.json")),
              ("val", os.path.join(root, "images", "val"), os.path.join(root, "labels", "val"),
               os.path.join(images_out_root, "val2017"), os.path.join(ann_out_dir, "instances_val2017.json"))]

    for split_name, img_dir, lbl_dir, out_img_folder, out_json_path in splits:
        coco_dict = build_coco_annotation(split_name, img_dir, lbl_dir, categories, classid2catid, out_img_folder)
        # 写 json
        with open(out_json_path, "w", encoding="utf-8") as f:
            json.dump(coco_dict, f, ensure_ascii=False, indent=2)
        print(f"[DONE] 写入 {split_name} COCO annotations 到 {out_json_path}, 包含 {len(coco_dict['images'])} 张图, {len(coco_dict['annotations'])} 个框.")


def main():
    print("Processing training dataset...")
    process_dataset(train_img_list, train_label_list, target_train_img_dir, target_train_label_dir)

    print("Processing validation dataset...")
    process_dataset(val_img_list, val_label_list, target_val_img_dir, target_val_label_dir)

    yolo2coco()

if __name__ == '__main__':
    main()