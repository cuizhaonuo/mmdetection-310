import os
from PIL import Image
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# ----- Configuration ----- #
# New base directories
base_img_dir = "/public/lab_share/nfs_field/level_1/DJI/P1/zhaoxian2022/20230526"
base_label_dir = "/public/home/zncui/pycharm_project/mmdetection/zhaoxian2022/20230526"

# Scale factor (r)
r = 1.0

# Supported image file extensions
extensions = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')


# ----- Helper Function ----- #
def process_image(task):
    """
    Process a single image:
      - Crop the image using the provided crop size (center crop).
      - Read the corresponding label file (if it exists) and convert the bounding boxes.
      - Save an Excel file (with columns: L1, L2, L3, L4, file_name, counts, h, w, r)
        and a TXT file (with the cropped bounding boxes) in the output directory.

    Parameters:
      task: tuple containing:
         L4 (str): Name of the L4 directory (e.g., "D6-097-R08L06").
         sub_img_path (str): Full path to the image folder for this L4 directory.
         image_file (str): Image file name.
         sub_label_dir (str): Full path to the label folder for this L4 directory.
         sub_output_dir (str): Full path to the output folder (for Excel and TXT files).
         crop_width (int): Crop width.
         crop_height (int): Crop height.
    """
    L4, sub_img_path, image_file, sub_label_dir, sub_output_dir, crop_width, crop_height = task
    image_path = os.path.join(sub_img_path, image_file)
    label_path = os.path.join(sub_label_dir, os.path.splitext(image_file)[0] + ".txt")

    try:
        with Image.open(image_path) as orig_img:
            img_w, img_h = orig_img.size
            # Calculate center crop box
            left = (img_w - crop_width) // 2
            top = (img_h - crop_height) // 2
            right = left + crop_width
            bottom = top + crop_height
            cropped_img = orig_img.crop((left, top, right, bottom))
    except Exception:
        return

    # Process label boxes and convert them to cropped image coordinates
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

                # Clip coordinates to the crop boundaries
                new_left = max(new_left, 0)
                new_top = max(new_top, 0)
                new_right = min(new_right, crop_width)
                new_bottom = min(new_bottom, crop_height)

                if new_left < new_right and new_top < new_bottom:
                    cropped_boxes.append((new_left, new_top, new_right - new_left, new_bottom - new_top))
        except Exception:
            pass

    box_count = len(cropped_boxes)

    # Extract L1 and L2 from the base directory (hard-coded or extracted from base_img_dir)
    L1 = "zhaoxian2022"
    # L2 is the folder name in base_img_dir (here "20230526")
    L2 = "20230526"
    # L3 is the parent folder of the current L4 directory.
    L3 = os.path.basename(os.path.dirname(sub_img_path))
    L4 = L4  # current L4 directory name

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

    # Save the Excel file (using the same base file name with .xlsx extension)
    excel_filename = os.path.join(sub_output_dir, os.path.splitext(image_file)[0] + ".xlsx")
    df.to_excel(excel_filename, index=False)

    # Save the cropped bounding boxes into a TXT file (one line per box: x,y,w,h)
    txt_filename = os.path.join(sub_output_dir, os.path.splitext(image_file)[0] + ".txt")
    with open(txt_filename, 'w') as f_txt:
        for box in cropped_boxes:
            f_txt.write(f"{box[0]},{box[1]},{box[2]},{box[3]}\n")


# ----- Main Processing ----- #
# Now we iterate two levels.
# Level L3: immediate subdirectories of base_img_dir (e.g., "D6H400cm")
# Level L4: subdirectories inside each L3 directory (e.g., "D6-097-R08L06")
tasks = []
L3_dirs = [d for d in os.listdir(base_img_dir) if os.path.isdir(os.path.join(base_img_dir, d))]
for L3 in L3_dirs:
    L3_path = os.path.join(base_img_dir, L3)
    # List L4 directories inside L3
    L4_dirs = [d for d in os.listdir(L3_path) if os.path.isdir(os.path.join(L3_path, d))]
    for L4 in L4_dirs:
        sub_img_path = os.path.join(L3_path, L4)
        sub_label_dir = os.path.join(base_label_dir, L3, L4, "results")
        sub_output_dir = os.path.join(base_label_dir, L3, L4, "results2")
        os.makedirs(sub_output_dir, exist_ok=True)

        # List all image files in the L4 directory
        image_files = [f for f in os.listdir(sub_img_path) if f.lower().endswith(extensions)]
        if not image_files:
            continue

        # Calculate the minimum width and height for the current L4 directory
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

        # Create a task for each image file in this L4 directory
        for image_file in image_files:
            tasks.append((L4, sub_img_path, image_file, sub_label_dir, sub_output_dir, crop_width, crop_height))

# Process all tasks concurrently using a progress bar.
max_workers = 8  # Adjust the number of threads if necessary
with ThreadPoolExecutor(max_workers=max_workers) as executor:
    list(tqdm(executor.map(process_image, tasks), total=len(tasks)))

print("Processing complete.")
