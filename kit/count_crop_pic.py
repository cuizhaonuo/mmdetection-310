import os
from PIL import Image
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# ----- Configuration ----- #
# Base directories
# base_img_dir = "/public/lab_share/nfs_field/level_1/DJI/P1/zhaoxian2022/20230526/D7H400cm"
# base_label_dir = "/public/home/zncui/pycharm_project/mmdetection/zhaoxian2022/20230526/D7H400cm"
base_img_dir = "/public/lab_share/nfs_field/level_1/DJI/P1/zhaoxian2021/20220527/D9s/"
base_label_dir = "/public/home/zncui/pycharm_project/mmdetection/zhaoxian2021/20220527/D9s"

# In each subdirectory, labels are stored under "<subdir>/results" and outputs will be saved under "<subdir>/results2".
# Scale factor (r)
r = 1.0

# Supported image file extensions
extensions = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')


# ----- Helper Function ----- #
def process_image(task):
    """
    Process a single image:
      - Crop the image using the provided crop size.
      - Read the corresponding label file (if exists) and convert box coordinates.
      - Save an Excel file with the image info and a TXT file with cropped bounding boxes.

    Parameters:
      task: tuple containing:
         sub: subdirectory name (e.g., "D6-097-R08L06")
         sub_img_path: full path of the image folder for this subdirectory.
         image_file: image file name.
         sub_label_dir: full path of the label folder (for this subdirectory).
         sub_output_dir: full path of the output folder (for this subdirectory).
         crop_width: crop width (int).
         crop_height: crop height (int).
    """
    sub, sub_img_path, image_file, sub_label_dir, sub_output_dir, crop_width, crop_height = task
    image_path = os.path.join(sub_img_path, image_file)
    label_path = os.path.join(sub_label_dir, os.path.splitext(image_file)[0] + ".txt")

    try:
        with Image.open(image_path) as orig_img:
            img_w, img_h = orig_img.size
            # Compute center crop box (assumes image is large enough since crop size is based on minimum among images)
            left = (img_w - crop_width) // 2
            top = (img_h - crop_height) // 2
            right = left + crop_width
            bottom = top + crop_height
            cropped_img = orig_img.crop((left, top, right, bottom))
    except Exception as e:
        # If image reading fails, skip this image.
        return

    # Process labels and compute boxes in cropped coordinates
    cropped_boxes = []
    if os.path.exists(label_path):
        try:
            with open(label_path, 'r') as f_label:
                lines = f_label.readlines()
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(',')
                if len(parts) != 4:
                    continue
                try:
                    x = float(parts[0])
                    y = float(parts[1])
                    box_w = float(parts[2])
                    box_h = float(parts[3])
                except:
                    continue

                # Original box coordinates
                box_left = x
                box_top = y
                box_right = x + box_w
                box_bottom = y + box_h

                # Convert to cropped image coordinates
                new_left = box_left - left
                new_top = box_top - top
                new_right = box_right - left
                new_bottom = box_bottom - top

                # Clip to crop boundaries (to include targets at the edge)
                new_left = max(new_left, 0)
                new_top = max(new_top, 0)
                new_right = min(new_right, crop_width)
                new_bottom = min(new_bottom, crop_height)

                if new_left < new_right and new_top < new_bottom:
                    cropped_boxes.append((new_left, new_top, new_right - new_left, new_bottom - new_top))
        except Exception as e:
            pass

    box_count = len(cropped_boxes)

    # Extract L1, L2, L3, L4 from the image path.
    # Expected image_path structure:
    # /public/lab_share/nfs_field/level_1/DJI/P1/zhaoxian2022/20230526/D6H400cm/<sub>/image.png
    parts = os.path.normpath(image_path).split(os.sep)
    if len(parts) >= 5:
        # Using indices relative to the end:
        # parts[-5]: "zhaoxian2022", parts[-4]: "20230526", parts[-3]: "D6H400cm", parts[-2]: sub (e.g., "D6-097-R08L06")
        L1 = parts[-5]
        L2 = parts[-4]
        L3 = parts[-3]
        L4 = parts[-2]
    else:
        L1 = L2 = L3 = L4 = ""

    # Create a DataFrame for Excel file with columns: L1, L2, L3, L4, file_name, counts, h, w, r
    data = {
        "L1": [L1],
        "L2": [L2],
        "L3": [L3],
        "L4": [L4],
        "file_name": [image_file],
        "counts": [box_count],
        "h": [crop_height],
        "w": [crop_width],
        "r": [r]
    }
    df = pd.DataFrame(data)

    # Save the Excel file (with .xlsx extension) in the output folder
    excel_filename = os.path.join(sub_output_dir, os.path.splitext(image_file)[0] + ".xlsx")
    df.to_excel(excel_filename, index=False)

    # Save the cropped bounding boxes into a TXT file (one line per box: x,y,w,h)
    txt_filename = os.path.join(sub_output_dir, os.path.splitext(image_file)[0] + ".txt")
    with open(txt_filename, 'w') as f_txt:
        for box in cropped_boxes:
            f_txt.write(f"{box[0]},{box[1]},{box[2]},{box[3]}\n")


# ----- Main Processing ----- #
# Get all subdirectories in the base image directory
subdirs = [d for d in os.listdir(base_img_dir) if os.path.isdir(os.path.join(base_img_dir, d))]
tasks = []

# Iterate over each subdirectory and build a list of tasks.
for sub in subdirs:
    sub_img_path = os.path.join(base_img_dir, sub)
    sub_label_dir = os.path.join(base_label_dir, sub, "results")
    sub_output_dir = os.path.join(base_label_dir, sub, "results2")
    os.makedirs(sub_output_dir, exist_ok=True)

    # List all image files in the subdirectory
    image_files = [f for f in os.listdir(sub_img_path) if f.lower().endswith(extensions)]
    if not image_files:
        continue

    # Compute the minimum width and height for the current subdirectory
    min_width = float('inf')
    min_height = float('inf')
    for f in image_files:
        img_file_path = os.path.join(sub_img_path, f)
        try:
            with Image.open(img_file_path) as img:
                w, h = img.size
                if w < min_width:
                    min_width = w
                if h < min_height:
                    min_height = h
        except Exception:
            continue
    crop_width = int(min_width * r)
    crop_height = int(min_height * r)

    # Append a task for each image file
    for image_file in image_files:
        tasks.append((sub, sub_img_path, image_file, sub_label_dir, sub_output_dir, crop_width, crop_height))

# Process all tasks concurrently using a progress bar
max_workers = 8  # Adjust number of threads as needed
with ThreadPoolExecutor(max_workers=max_workers) as executor:
    list(tqdm(executor.map(process_image, tasks), total=len(tasks)))

print("Processing complete.")
