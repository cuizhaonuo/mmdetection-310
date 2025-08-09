import os

wheat_dir = '/public/home/zncui/datasets/wheat-guo'
wheat_crop_dir = '/public/home/zncui/datasets/wheat-guo-crop'

for file in os.listdir(wheat_crop_dir):
    if file not in os.listdir(wheat_dir):
        print(f"File {file} in crop directory but not in original directory.")