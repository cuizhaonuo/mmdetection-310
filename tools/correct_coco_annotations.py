import re
from pycocotools.coco import COCO

coco_annotations_path = '/public/home/zncui/pycharm_project/mmdetection/data/coco/annotations/instances_train2017.json'
new_annotations_path = '/public/home/zncui/pycharm_project/mmdetection/data/coco/annotations/instances_train2017_corrected.json'
coco = COCO(coco_annotations_path)

images = coco.dataset['images']

pattern = re.compile('((?:D\d-\d{3}-\d{8}|[^/]+_\d+_\d+)\.jpg)$')

no_match_images = []
match_images = []
for i, image in enumerate(images):
    file_name = pattern.search(image['file_name'])
    if file_name is not None:
        images[i]['file_name'] = file_name.group(1)
        match_images.append(image)
        print(file_name.group(1))
    else:
        no_match_images.append(image)
        print(f"File name '{image['file_name']}' does not match the pattern.")
print(no_match_images)
print(f"Total images with no match: {len(no_match_images)}")

if len(match_images) == len(images):
    print("All images match the pattern.")
    import json
    with open(new_annotations_path, 'w') as f:
        json.dump(coco.dataset, f, indent=4)
    with open(new_annotations_path, 'r') as f:
        new_annotations = json.load(f)
        images = new_annotations['images']
        for image in  images:
            print(image['file_name'])
        print(f"Total images: {len(images)}")
