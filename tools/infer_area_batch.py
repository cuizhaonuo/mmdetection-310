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
from mmcv.ops import nms

# 加快推理速度的设置
torch.backends.cudnn.benchmark = True

def sliding_window_inference(model, img, window_size=1024, overlap_ratio=0.1,
                             score_thr=0.0, nms_iou=0.1):
    """对图像进行滑窗推理。"""
    h, w = img.shape[:2]
    stride = int(window_size * (1 - overlap_ratio))
    all_detections = []

    for y in range(0, h, stride):
        for x in range(0, w, stride):
            y_end = min(y + window_size, h)
            x_end = min(x + window_size, w)
            crop = img[y:y_end, x:x_end, :]

            # 推理
            result = inference_detector(model, crop)
            instances = result.pred_instances

            # 筛选分数大于 score_thr 的结果
            valid_mask = instances.scores > score_thr
            instances = instances[valid_mask].cpu()

            if len(instances) > 0:
                # 调整回原图坐标
                instances.bboxes[:, 0::2] += x
                instances.bboxes[:, 1::2] += y
                instances.bboxes[:, 0::2] = torch.clamp(instances.bboxes[:, 0::2], 0, w)
                instances.bboxes[:, 1::2] = torch.clamp(instances.bboxes[:, 1::2], 0, h)
                all_detections.append(instances)

    if len(all_detections) == 0:
        return DetDataSample()

    # 合并所有窗口的结果并做 NMS
    merged_instances = InstanceData.cat(all_detections)
    keep = batched_nms(merged_instances.bboxes, merged_instances.scores,
                       merged_instances.labels, nms_iou)
    final_result = DetDataSample()
    final_result.pred_instances = merged_instances[keep[:]].cpu()
    return final_result

def batched_nms(bboxes, scores, labels, iou_threshold):
    """对不同类别的检测框分别做 NMS，然后根据分数排序合并结果。"""
    unique_labels = labels.unique()
    keep = []

    for cls_id in unique_labels:
        cls_mask = (labels == cls_id)
        cls_bboxes = bboxes[cls_mask]
        cls_scores = scores[cls_mask]

        if cls_bboxes.shape[0] == 0:
            continue

        device = cls_bboxes.device
        # 这里的 nms 基于 GPU 运算，可显著加速
        _, inds = nms(cls_bboxes.to(device),
                      cls_scores.to(device),
                      iou_threshold=iou_threshold)
        mask_indices = cls_mask.nonzero()[:, 0]
        keep.extend(mask_indices[inds].cpu().numpy())

    scores_cpu = scores.cpu().numpy()
    # 根据分数降序排序
    return np.array(keep)[np.argsort(-scores_cpu[keep])]

if __name__ == '__main__':
    base_dir = '/public/home/zncui/pycharm_project/mmdetection'
    config_file = os.path.join(base_dir,
                               'work_dirs/co_dino_5scale_swin_l_16xb1_16e_o365tococo/co_dino_5scale_swin_l_16xb1_16e_o365tococo.py')
    checkpoint_file = os.path.join(base_dir, 'work_dirs/co_dino_5scale_swin_l_16xb1_16e_o365tococo/epoch_16.pth')
    input_dir = '/public/lab_share/nfs_field/level_1/DJI/P1/zhaoxian2021'
    output_dir_root = '/public/home/zncui/pycharm_project/mmdetection/zhaoxian2021'

    # 使用 GPU:1
    model = init_detector(config_file, checkpoint_file, device='cuda:1')

    # 一次性创建可视化对象，避免在循环中重复创建
    visualizer = DetLocalVisualizer()
    visualizer.dataset_meta = model.dataset_meta

    all_image_files = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                all_image_files.append(os.path.join(root, file))

    total_files = len(all_image_files)
    start_time = time.time()
    total_boxes = 0
    processed_count = 0

    print(f"开始处理 {total_files} 张图片...")

    # 在推理前后使用 no_grad, 避免不必要的梯度计算
    with torch.no_grad():
        for idx, input_path in enumerate(all_image_files, 1):
            filename = os.path.basename(input_path)
            try:
                # 解析路径，用来生成输出信息
                rel_path = os.path.relpath(input_path, input_dir)
                rel_dir, filename = os.path.split(rel_path)
                parts = rel_dir.split(os.sep) if rel_dir else []
                L1 = 'zhaoxian2021'
                L2 = parts[0] if len(parts) >= 1 else ''
                L3 = parts[1] if len(parts) >= 2 else ''
                L4 = parts[2] if len(parts) >= 3 else ''
                file_name = filename

                # 创建输出目录
                output_image_dir = os.path.join(output_dir_root, rel_dir)
                output_image_path = os.path.join(output_image_dir, filename)
                os.makedirs(output_image_dir, exist_ok=True)

                # 读取图像
                img = mmcv.imread(input_path, channel_order='rgb')

                # 滑窗推理
                result = sliding_window_inference(
                    model, img,
                    window_size=640,    # 调大滑窗尺寸
                    overlap_ratio=0.1,   # 减少重叠
                    score_thr=0.3,
                    nms_iou=0.3
                )

                # 可视化并保存结果
                visualizer.add_datasample(
                    'detection', img, result,
                    draw_pred=True, draw_gt=False,
                    show=False, out_file=output_image_path
                )

                # 保存结果到 Excel 和 TXT
                results_dir = os.path.join(output_image_dir, 'results')
                os.makedirs(results_dir, exist_ok=True)
                base_name = os.path.splitext(filename)[0]
                excel_path = os.path.join(results_dir, f"{base_name}.xlsx")
                txt_path = os.path.join(results_dir, f"{base_name}.txt")

                pred_instances = result.pred_instances
                bboxes = pred_instances.bboxes.cpu().numpy()
                scores = pred_instances.scores.cpu().numpy()
                labels = pred_instances.labels.cpu().numpy()

                # 筛选分数大于阈值的框
                score_threshold = 0.3
                mask = scores > score_threshold
                bboxes = bboxes[mask]
                scores = scores[mask]
                labels = labels[mask]

                # 写入 Excel (只记录框数和路径信息)
                data = [{
                    'L1': L1,
                    'L2': L2,
                    'L3': L3,
                    'L4': L4,
                    'file_name': file_name,
                    'counts': len(bboxes)
                }]
                df = pd.DataFrame(data)
                df.to_excel(excel_path, index=False)

                # 写入 TXT (详细的坐标信息)
                with open(txt_path, 'w') as f_txt:
                    for bbox in bboxes:
                        x_min, y_min, x_max, y_max = bbox
                        x, y, w, h = map(int, (x_min, y_min, x_max - x_min, y_max - y_min))
                        f_txt.write(f"{x},{y},{w},{h}\n")

                num_boxes = len(bboxes)
                total_boxes += num_boxes
                processed_count += 1

                # 打印进度信息
                elapsed = time.time() - start_time
                avg_time = elapsed / idx
                remaining = avg_time * (total_files - idx)
                progress_msg = (
                    f"进度: {idx}/{total_files} 已用: {elapsed:.1f}s | "
                    f"剩余: {remaining:.1f}s | 当前框: {num_boxes} | "
                    f"总框: {total_boxes} | {input_path}"
                )
                print(f"\r{progress_msg}", end="", flush=True)

            except Exception as e:
                print(f"\n处理 {filename} 时出错: {str(e)}")
                continue

    total_time = time.time() - start_time
    print(f"\n\n处理完成！共处理 {processed_count}/{total_files} 张图片")
    print(f"总预测框数量: {total_boxes}")
    if processed_count > 0:
        print(f"总耗时: {total_time:.2f} 秒 (平均 {total_time / processed_count:.2f} 秒/张)")
    else:
        print("无成功处理的图片。")
