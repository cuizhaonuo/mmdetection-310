import os
import mmcv
import torch
import glob
import time
import numpy as np
import pandas as pd
from mmdet.apis import init_detector, inference_detector
from mmdet.visualization import DetLocalVisualizer
from mmengine.structures import InstanceData
from mmdet.structures import DetDataSample


def sliding_window_inference(model, img, window_size=512, overlap_ratio=0.2, score_thr=0.0, nms_iou=0.1):
    h, w = img.shape[:2]
    stride = int(window_size * (1 - overlap_ratio))
    all_detections = []

    for y in range(0, h, stride):
        for x in range(0, w, stride):
            y_end = min(y + window_size, h)
            x_end = min(x + window_size, w)
            crop = img[y:y_end, x:x_end, :]

            result = inference_detector(model, crop)
            instances = result.pred_instances

            valid_mask = instances.scores > score_thr
            instances = instances[valid_mask].cpu()

            if len(instances) > 0:
                instances.bboxes[:, 0::2] += x
                instances.bboxes[:, 1::2] += y
                instances.bboxes[:, 0::2] = torch.clamp(instances.bboxes[:, 0::2], 0, w)
                instances.bboxes[:, 1::2] = torch.clamp(instances.bboxes[:, 1::2], 0, h)
                all_detections.append(instances)

    if len(all_detections) == 0:
        return DetDataSample()

    merged_instances = InstanceData.cat(all_detections)
    keep = batched_nms(merged_instances.bboxes, merged_instances.scores, merged_instances.labels, nms_iou)
    final_result = DetDataSample()
    final_result.pred_instances = merged_instances[keep[:]].cpu()
    return final_result


def batched_nms(bboxes, scores, labels, iou_threshold):
    from mmcv.ops import nms
    unique_labels = labels.unique()
    keep = []

    for cls_id in unique_labels:
        cls_mask = (labels == cls_id)
        cls_bboxes = bboxes[cls_mask]
        cls_scores = scores[cls_mask]

        if cls_bboxes.shape[0] == 0:
            continue

        device = cls_bboxes.device
        _, inds = nms(cls_bboxes.to(device), cls_scores.to(device), iou_threshold=iou_threshold)
        mask_indices = cls_mask.nonzero()[:, 0]
        keep.extend(mask_indices[inds].cpu().numpy())

    scores_cpu = scores.cpu().numpy()
    return np.array(keep)[np.argsort(-scores_cpu[keep])]

if __name__ == '__main__':
    base_dir = '/public/home/zncui/pycharm_project/mmdetection'
    config_file = os.path.join(base_dir,
                               'work_dirs/co_dino_5scale_swin_l_16xb1_16e_o365tococo/co_dino_5scale_swin_l_16xb1_16e_o365tococo.py')
    checkpoint_file = os.path.join(base_dir, 'work_dirs/co_dino_5scale_swin_l_16xb1_16e_o365tococo/epoch_16.pth')
    # config_file = os.path.join(base_dir,
    #                            'work_dirs/yolox_x_8xb8-300e_coco/yolox_x_8xb8-300e_coco.py')
    # checkpoint_file = os.path.join(base_dir, 'work_dirs/yolox_x_8xb8-300e_coco/epoch_300.pth')

    input_dir = '/public/lab_share/nfs_field/level_1/DJI/P1/zhaoxian2021/'
    target_dir = os.path.join(input_dir, '20220529')
    output_dir_root = '/public/home/zncui/pycharm_project/mmdetection/zhaoxian2021/'

    model = init_detector(config_file, checkpoint_file, device='cuda:1')
    all_image_files = []
    for root, dirs, files in os.walk(target_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.JPG')):
                all_image_files.append(os.path.join(root, file))

    total_files = len(all_image_files)
    start_time = time.time()
    total_boxes = 0
    processed_count = 0
    print(f"开始处理 {total_files} 张图片...")

    for idx, input_path in enumerate(all_image_files, 1):
        filename = os.path.basename(input_path)
        try:
            rel_path = os.path.relpath(input_path, input_dir)
            rel_dir, filename = os.path.split(rel_path)
            parts = rel_dir.split(os.sep) if rel_dir else []
            L1 = 'zhaoxian2021'
            L2 = parts[0] if len(parts) >= 1 else ''
            L3 = parts[1] if len(parts) >= 2 else ''
            L4 = parts[2] if len(parts) >= 3 else ''
            file_name = filename

            output_image_dir = os.path.join(output_dir_root, rel_dir)
            output_image_path = os.path.join(output_image_dir, filename)
            os.makedirs(output_image_dir, exist_ok=True)

            img = mmcv.imread(input_path, channel_order='rgb')
            result = sliding_window_inference(model, img, window_size=640, overlap_ratio=0.2, score_thr=0.3,
                                              nms_iou=0.3)

            visualizer = DetLocalVisualizer()
            visualizer.dataset_meta = model.dataset_meta
            visualizer.add_datasample('detection', img, result, draw_pred=True, draw_gt=False, show=False,
                                      out_file=output_image_path)

            results_dir = os.path.join(output_image_dir, 'results')
            os.makedirs(results_dir, exist_ok=True)
            base_name = os.path.splitext(filename)[0]
            excel_path = os.path.join(results_dir, f"{base_name}.xlsx")
            txt_path = os.path.join(results_dir, f"{base_name}.txt")

            pred_instances = result.pred_instances
            bboxes = pred_instances.bboxes.cpu().numpy()
            scores = pred_instances.scores.cpu().numpy()
            labels = pred_instances.labels.cpu().numpy()

            score_threshold = 0.3
            # 创建掩码，筛选分数大于阈值的索引
            mask = scores > score_threshold
            # 应用掩码获取筛选后的结果
            bboxes = bboxes[mask]
            scores = scores[mask]
            labels = labels[mask]

            data = []
            data.append({
                'L1': L1, 'L2': L2, 'L3': L3, 'L4': L4,
                'file_name': file_name, 'counts': len(bboxes)
            })

            df = pd.DataFrame(data)
            df.to_excel(excel_path, index=False)

            with open(txt_path, 'w') as f_txt:
                for bbox in bboxes:
                    x_min, y_min, x_max, y_max = bbox
                    x, y, w, h = map(int, (x_min, y_min, x_max - x_min, y_max - y_min))
                    f_txt.write(f"{x},{y},{w},{h}\n")

            num_boxes = len(bboxes)
            total_boxes += num_boxes
            processed_count += 1

            elapsed = time.time() - start_time
            avg_time = elapsed / idx
            remaining = avg_time * (total_files - idx)
            progress_msg = (f"进度: {idx}/{total_files} 已用: {elapsed:.1f}s | 剩余: {remaining:.1f}s | "
                            f"当前框: {num_boxes} | 总框: {total_boxes} | {input_path}")
            print(f"\r{progress_msg}", end="", flush=True)

        except Exception as e:
            print(f"\n处理 {filename} 时出错: {str(e)}")
            continue

    total_time = time.time() - start_time
    print(f"\n\n处理完成！共处理 {processed_count}/{total_files} 张图片")
    print(f"总预测框数量: {total_boxes}")
    print(f"总耗时: {total_time:.2f} 秒 (平均 {total_time / processed_count:.2f} 秒/张)")