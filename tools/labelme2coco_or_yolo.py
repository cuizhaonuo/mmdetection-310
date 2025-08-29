import os
import json
import glob
from tqdm import tqdm
import shutil

labelme_wheat_dir = '/public/home/zncui/datasets/wheat-guo-dataset/labelme_wheat_big_image'
train_img_dir = os.path.join(labelme_wheat_dir, 'train_imgs')
val_img_dir = os.path.join(labelme_wheat_dir, 'val_imgs')

train_img_list = glob.glob(os.path.join(train_img_dir, '*.jpg'))
val_img_list = glob.glob(os.path.join(val_img_dir, '*.jpg'))

coco_wheat_dir = '/public/home/zncui/datasets/wheat-guo-dataset/coco_wheat_big_image'

def labelme2coco(img_list):
    img_id = 0
    ann_id = 0
    images = []
    annotations = []
    categories = [{
        'id': 1,
        'name': 'wheat',
    }]
    for img_path in tqdm(sorted(img_list), desc='Converting LabelMe to COCO'):
        img_id += 1
        img_name = os.path.basename(img_path)
        json_file = img_path.replace('.jpg', '.json')
        with open(json_file, 'r') as f:
            data = json.load(f)
        width, height = data['imageWidth'], data['imageHeight']
        img = {
            'id': img_id,
            'file_name': img_name,
            'width': width,
            'height': height
        }
        images.append(img)
        for j, shape in data['shapes']:
            ann_id += 1
            points = shape['points']
            x1, y1 = points[0]
            x2, y2 = points[1]
            x1, y1 = int(x1), int(y1)
            x2, y2 = int(x2), int(y2)
            width = x2 - x1
            height = y2 - y1
            if width <= 0 or height <= 0:
                print(f"Invalid bounding box in {json_file}, skipping...")
                continue
            annotation = {
                'id': ann_id,
                'image_id': img_id,
                'segmentation': [],
                'bbox': [x1, y1, width, height],
                'category_id': 1,
                'area': width * height,
                'iscrowd': 0
            }
            annotations.append(annotation)
    return {
        'images': images,
        'annotations': annotations,
        'categories': categories
    }

def labelme2yolo(img_list, yolo_wheat_dataset, pattern='train'):

    for img_path in tqdm(sorted(img_list), desc='Converting LabelMe to YOLO'):
        json_file = img_path.replace('.jpg', '.json')
        with open(json_file, 'r') as f:
            data = json.load(f)
        width, height = data['imageWidth'], data['imageHeight']
        yolo_data = []
        for shape in data['shapes']:
            points = shape['points']
            x1, y1 = points[0]
            x2, y2 = points[1]
            x1, y1 = int(x1), int(y1)
            x2, y2 = int(x2), int(y2)
            if x1 >= x2 or y1 >= y2:
                print(f"Invalid bounding box in {json_file}, skipping...")
                continue
            x_center = (x1 + x2) / 2 / width
            y_center = (y1 + y2) / 2 / height
            bbox_width = (x2 - x1) / width
            bbox_height = (y2 - y1) / height
            yolo_data.append(f"0 {x_center} {y_center} {bbox_width} {bbox_height}\n")
        img_name = os.path.basename(img_path)
        label_file = os.path.join(yolo_wheat_dataset, 'labels', pattern, img_name.replace('.jpg', '.txt'))
        with open(label_file, 'w') as f:
            f.writelines(yolo_data)
    # 补充 classes.txt 文件
    with open(os.path.join(yolo_wheat_dataset, 'labels/train', 'classes.txt'), 'w') as f:
        f.write('wheat\n')
    with open(os.path.join(yolo_wheat_dataset, 'labels/val', 'classes.txt'), 'w') as f:
        f.write('wheat\n')


def create_yolo_dir(output_dir='/public/home/zncui/datasets/wheat-guo-dataset'):
    yolo_wheat_v2_dir = os.path.join(output_dir, 'yolo_wheat_big_image_v2')
    yolo_wheat_v1_dir = os.path.join(output_dir, 'yolo_wheat_big_image')

    folder_types = ['images', 'labels']
    splits = ['train', 'val']

    for folder_type in folder_types:
        for split in splits:
            os.makedirs(os.path.join(yolo_wheat_v2_dir, folder_type, split), exist_ok=True)

    if os.path.exists(yolo_wheat_v1_dir):
        for split in splits:
            src_dir = os.path.join(yolo_wheat_v1_dir, 'images', split)
            dst_dir = os.path.join(yolo_wheat_v2_dir, 'images', split)
            if os.path.exists(src_dir):
                shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)

    labelme2yolo(train_img_list, yolo_wheat_v2_dir, 'train')
    labelme2yolo(val_img_list, yolo_wheat_v2_dir, 'val')

    ratio = [0.7, 0.15, 0.15]  # 70% train, 15% val, 15% test
    if ratio is not None:
        img_list = []
        for split in splits:
            img_dir = os.path.join(yolo_wheat_v2_dir, 'images', split)
            img_list.extend([
                os.path.join(img_dir, f)
                for f in os.listdir(img_dir)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))
            ])
        split_dataset(img_list, ratio)
        plot_bbox_from_yolo(yolo_wheat_dataset='yolo_wheat_big_image_v2split')

def plot_bbox_from_yolo(yolo_wheat_dataset='yolo_wheat_big_image_v2'):
    import cv2
    wheat_dataset = '/public/home/zncui/datasets/wheat-guo-dataset'
    yolo_wheat_dir = os.path.join(wheat_dataset, yolo_wheat_dataset)
    pattern = ['train', 'val', 'test']
    for p in pattern:
        if p is None:
            continue
        img_dir = os.path.join(yolo_wheat_dir, 'images', p)
        img_list = sorted(glob.glob(os.path.join(img_dir, '*.jpg')))
        label_list = [img_path.replace('images', 'labels').replace('.jpg', '.txt') for img_path in img_list
                      if 'classes.txt' not in img_path]
        save_dir = os.path.join(yolo_wheat_dir, 'save', p)
        os.makedirs(save_dir, exist_ok=True)
        for img_path, label_path in tqdm(zip(img_list, label_list), desc=f'Plotting {p} bboxes', total=len(img_list)):
            img = cv2.imread(img_path)
            h, w = img.shape[:2]
            with open(label_path, 'r') as f:
                lines = f.readlines()
            for line in lines:
                cls_id, x_center, y_center, bbox_w, bbox_h = map(float, line.strip().split())
                x_center *= w
                y_center *= h
                bbox_w *= w
                bbox_h *= h
                x1 = int(x_center - bbox_w / 2)
                y1 = int(y_center - bbox_h / 2)
                x2 = int(x_center + bbox_w / 2)
                y2 = int(y_center + bbox_h / 2)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            save_path = os.path.join(save_dir, os.path.basename(img_path))
            cv2.imwrite(save_path, img)


def split_dataset(img_list, ratio=[0.7, 0.15, 0.15]):
    total = len(img_list)
    train_size = int(total * ratio[0])
    val_size = int(total * ratio[1])
    test_size = total - train_size - val_size
    # 随机采样
    import random
    random.seed(42)  # 设置随机种子以保证结果可复现
    random.shuffle(img_list)
    train_img_list = img_list[:train_size]
    val_img_list = img_list[train_size:train_size + val_size]
    test_img_list = img_list[train_size + val_size:]

    def create_directory_structure(yolo_wheat_dir):
        # 定义需要创建的目录结构
        folder_types = ['images', 'labels']
        splits = ['train', 'val', 'test']

        # 使用列表推导式生成所有需要创建的路径
        paths = [os.path.join(yolo_wheat_dir, folder_type, split)
                 for folder_type in folder_types
                 for split in splits]

        # 创建所有目录
        for path in paths:
            os.makedirs(path, exist_ok=True)

    yolo_wheat_dir = '/public/home/zncui/datasets/wheat-guo-dataset/yolo_wheat_big_image_v2split'
    create_directory_structure(yolo_wheat_dir)

    def copy_images_and_labels(img_list, pattern):
        for img_path in img_list:
            img_name = os.path.basename(img_path)
            label_path = img_path.replace('.jpg', '.txt').replace('images', 'labels')
            if not os.path.exists(label_path):
                continue
            # 复制图片
            dest_img_path = os.path.join(yolo_wheat_dir, 'images', pattern, img_name)
            shutil.copy(img_path, dest_img_path)
            # 复制标签
            dest_label_path = os.path.join(yolo_wheat_dir, 'labels', pattern, img_name.replace('.jpg', '.txt'))
            shutil.copy(label_path, dest_label_path)
    copy_images_and_labels(train_img_list, 'train')
    copy_images_and_labels(val_img_list, 'val')
    copy_images_and_labels(test_img_list, 'test')




def main():
    # train_coco = labelme2coco(train_img_list)
    # val_coco = labelme2coco(val_img_list)
    # train_ann_json_path = os.path.join(coco_wheat_dir, 'annotations', 'train.json')
    # val_ann_json_path = os.path.join(coco_wheat_dir, 'annotations', 'val.json')
    # with open(train_ann_json_path, 'w') as f:
    #     json.dump(train_coco, f, indent=4)
    # with open(val_ann_json_path, 'w') as f:
    #     json.dump(val_coco, f, indent=4)

    create_yolo_dir()
    plot_bbox_from_yolo(yolo_wheat_dataset='yolo_wheat_big_image_v2')


if __name__ == '__main__':
    main()