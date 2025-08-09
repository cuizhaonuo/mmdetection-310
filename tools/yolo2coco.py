import os
import cv2
import json
from pycocotools.coco import COCO

base_dir = '/public/home/zncui/datasets/wheat-guo-dataset'
label_dir = os.path.join(base_dir, 'wheat-guo-crop-640-output', 'rect_labelimg')
pred_dir = os.path.join(base_dir, 'wheat-guo-crop-640-output', 'rect_pred')
img_dir = os.path.join(base_dir, 'wheat-guo-crop-640')

project_dir = '/public/home/zncui/pycharm_project/mmdetection'
train_annotations_path = os.path.join(project_dir, 'data/coco', 'annotations', 'instances_train2017.json')

def check_list_continuity(lst):
    """Check if a list of integers is continuous."""
    return sorted(lst) == list(range(min(lst), max(lst) + 1))

def find_coco_last_id(annotations_path):
    coco = COCO(annotations_path)
    catIds = coco.getCatIds(catNms=['wheat'])
    imgIds = coco.getImgIds(catIds=catIds)
    annIds = coco.getAnnIds(catIds=catIds)
    # 检验 imgIds 是否连续
    if check_list_continuity(imgIds) and check_list_continuity(annIds):
        return imgIds[-1], annIds[-1]
    else:
        print('imgIds or annIds are not continuous.')
        return -1

coco_annotations = {
    'images': [],
    'annotations': [],
    'categories': [{'id': 1, 'name': 'wheat'}]
}

img_id, ann_id = find_coco_last_id(train_annotations_path)

current_img_id = img_id
current_ann_id = ann_id

# total = None
for root, dirs, files in os.walk(pred_dir):
    for file in files:
        if file.endswith('.txt') and not file.startswith('class'):
            label_path = os.path.join(label_dir, file)
            pred_path = os.path.join(pred_dir, file)
            img_path = os.path.join(img_dir, file.replace('.txt', '.jpg'))
            if os.path.exists(label_path):
                # 合并预测结果和标注
                with open(label_path, 'r') as f_label, open(pred_path, 'r') as f_pred:
                    label_lines = [line.split(' ', 1)[1] for line in f_label.readlines()]
                    for i, line in enumerate(label_lines):
                        x, y, w, h = [float(v) * 640 for v in line.strip().split()[:4]]
                        line = f"{x} {y} {w} {h}"
                        label_lines[i] = line
                    pred_lines = [line.replace(',', ' ') for line in f_pred.readlines()]
                    pred_lines = pred_lines + label_lines
            else:
                with open(pred_path, 'r') as f_pred:
                    pred_lines = [line.replace(',', ' ') for line in f_pred.readlines()]

            if not os.path.exists(pred_path):
                print(f"Prediction file {pred_path} does not exist for label {label_path}.")
                continue

            w, h = cv2.imread(img_path).shape[:2]
            image = {
                'file_name': img_path,
                'height': h,
                'width': w,
                'id': current_img_id,
            }
            coco_annotations['images'].append(image)

            for line in pred_lines:
                current_img_id += 1
                annotation = {
                    'id': current_ann_id,
                    'image_id': current_ann_id,
                    'category_id': 1,  # Assuming a single category
                    'segmentation': [],
                    'area': 0.0,
                    'bbox': [float(coord) for coord in line.split(' ')[:4]],
                    'iscrowd': 0,
                }
                coco_annotations['annotations'].append(annotation)

            # if len(label_lines) != len(pred_lines):
            #     print(f"Mismatch in number of lines for {file}: {len(label_lines)} labels vs {len(pred_lines)} predictions.")


with open(train_annotations_path, 'r', encoding='utf-8') as f:
    coco = json.load(f)
    coco['images'].extend(coco_annotations['images'])
    coco['annotations'].extend(coco_annotations['annotations'])

# 保存到新的文件
out_path = os.path.join(
    project_dir, 'data', 'coco', 'annotations', 'instances_train2017_merged.json'
)
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(
        coco,
        f,
        ensure_ascii=False,  # 如果有中文或非 ASCII 字符，设置为 False 才能正常写入
        indent=2  # 美化输出，缩进为 2 个空格
    )
