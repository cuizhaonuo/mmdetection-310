import os
import json
import base64
from pathlib import Path
from tqdm import tqdm

try:
    import cv2
except ImportError:
    cv2 = None
    from PIL import Image

# --------- 配置区，改成你自己的路径即可 ----------
DATA_DIR = "/public/home/zncui/datasets/wheat-guo-dataset/imgs_and_labels/val_imgs"
CLASSES_FILE = "classes.txt"
# ------------------------------------------------------

def load_classes(classes_path):
    with open(classes_path, 'r', encoding='utf-8') as f:
        names = [line.strip() for line in f.readlines() if line.strip()]
    return names

def get_image_size(img_path):
    if cv2:
        img = cv2.imread(str(img_path))
        if img is None:
            raise RuntimeError(f"Failed to read image with OpenCV: {img_path}")
        h, w = img.shape[:2]
    else:
        with Image.open(img_path) as im:
            w, h = im.size
    return w, h

def maybe_embed_image(img_path):
    # 当前默认不 embed（返回 None），如果想 embed 图像，把下面解除注释
    return None
    # with open(img_path, "rb") as f:
    #     encoded = base64.b64encode(f.read()).decode("utf-8")
    # return encoded


def yolo_to_labelme_json(image_path: Path, label_path: Path, class_names):
    w, h = get_image_size(image_path)
    shapes = []

    if label_path.is_file():
        with open(label_path, 'r', encoding='utf-8') as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 5:
                    print(f"[WARN] Skipping malformed line {line_no} in {label_path.name}: '{line}'")
                    continue
                try:
                    cls_id = int(parts[0])
                except ValueError:
                    print(f"[WARN] Invalid class id on line {line_no} in {label_path.name}: '{parts[0]}'")
                    continue
                if cls_id < 0 or cls_id >= len(class_names):
                    label = f"cls{cls_id}"
                    print(f"[WARN] class id {cls_id} out of range in {label_path.name}, using label '{label}'")
                else:
                    label = class_names[cls_id]
                try:
                    x_c, y_c, bw, bh = map(float, parts[1:5])
                except ValueError:
                    print(f"[WARN] Invalid bbox numbers on line {line_no} in {label_path.name}: '{parts[1:5]}'")
                    continue

                # 归一化 -> 绝对坐标
                box_w = bw * w
                box_h = bh * h
                center_x = x_c * w
                center_y = y_c * h

                xmin = center_x - box_w / 2
                ymin = center_y - box_h / 2
                xmax = center_x + box_w / 2
                ymax = center_y + box_h / 2

                xmin, ymin = max(0, xmin), max(0, ymin)
                xmax, ymax = min(w, xmax), min(h, ymax)

                shape = {
                    "label": label,
                    "points": [[xmin, ymin], [xmax, ymax]],
                    "group_id": None,
                    "shape_type": "rectangle",
                    "flags": {},
                    "line_color": None,
                    "fill_color": None,
                }
                shapes.append(shape)
    else:
        print(f"[INFO] No YOLO label file for image {image_path.name}, creating empty annotation.")


    json_dict = {
        "version": "5.3.3",
        "flags": {},
        "shapes": shapes,
        "imagePath": image_path.name,
        "imageData": maybe_embed_image(image_path),
        "imageHeight": h,
        "imageWidth": w,
    }
    return json_dict


def change_shapes_sequence(json_dict):
    grid_size = (3, 3)
    h, w = json_dict['imageHeight'], json_dict['imageWidth']
    grid_h, grid_w = h // grid_size[0], w // grid_size[1]
    shapes = json_dict['shapes']
    changed_shapes = []
    for y in range(0, h, grid_h):
        for x in range(0, w, grid_w):
            y_end = min(y + grid_h, h)
            x_end = min(x + grid_w, w)
            grid_shapes = []
            for shape in shapes:
                (x1, y1), (x2, y2) = shape['points']
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                # 检查中心点是否在当前网格内
                if x <= center_x < x_end and y <= center_y < y_end:
                    grid_shapes.append(shape)
            grid_shapes = sorted(grid_shapes, key=lambda s: ((s['points'][0][0] + s['points'][1][0]) / 2 + (s['points'][0][1] + s['points'][1][1]) / 2))
            changed_shapes.append(grid_shapes)
    final_shapes = [shape for sublist in changed_shapes for shape in sublist]
    json_dict['shapes'] = final_shapes
    return json_dict


def batch_convert(dir_path, classes_file="classes.txt", exts=(".jpg", ".jpeg", ".png")):
    dir_path = Path(dir_path)
    classes_path = dir_path / classes_file
    if not classes_path.exists():
        raise FileNotFoundError(f"classes.txt not found at {classes_path}")
    class_names = load_classes(classes_path)

    images = []
    for ext in exts:
        images.extend(sorted(dir_path.glob(f"*{ext}")))
        images.extend(sorted(dir_path.glob(f"*{ext.upper()}")))

    if not images:
        print(f"[WARN] No images found in {dir_path}")
        return

    for img_path in tqdm(images):
        base = img_path.stem
        label_path = dir_path / f"{base}.txt"
        json_path = dir_path / f"{base}.json"

        json_dict = yolo_to_labelme_json(img_path, label_path, class_names)
        json_dict = change_shapes_sequence(json_dict)
        with open(json_path, "w", encoding="utf-8") as jf:
            json.dump(json_dict, jf, ensure_ascii=False, indent=2)
        # print(f"[OK] Wrote {json_path.name} ({len(json_dict['shapes'])} shapes)")

def main():
    print(f"Converting YOLO labels to Labelme JSON in '{DATA_DIR}' using classes file '{CLASSES_FILE}'")
    batch_convert(DATA_DIR, classes_file=CLASSES_FILE)

if __name__ == "__main__":
    main()
