import os
import mmcv
import torch
import glob
import time
import numpy as np
from mmdet.apis import init_detector, inference_detector
from mmdet.visualization import DetLocalVisualizer
from mmengine.structures import InstanceData
from mmdet.structures import DetDataSample
from concurrent.futures import ThreadPoolExecutor


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

            # 过滤并转换到CPU
            valid_mask = instances.scores > score_thr
            instances = instances[valid_mask].cpu()  # 关键修改：提前转移到CPU

            if len(instances) > 0:
                instances.bboxes[:, 0::2] += x
                instances.bboxes[:, 1::2] += y
                instances.bboxes[:, 0::2] = torch.clamp(instances.bboxes[:, 0::2], 0, w)
                instances.bboxes[:, 1::2] = torch.clamp(instances.bboxes[:, 1::2], 0, h)

                all_detections.append(instances)

    if len(all_detections) == 0:
        return DetDataSample()  # 返回空DetDataSample

    merged_instances = InstanceData.cat(all_detections)
    keep = batched_nms(merged_instances.bboxes, merged_instances.scores, merged_instances.labels, nms_iou)

    # 转换为DetDataSample格式
    final_result = DetDataSample()
    final_result.pred_instances = merged_instances[keep[:]].cpu()  # 确保在CPU
    return final_result


def batched_nms(bboxes, scores, labels, iou_threshold):
    """跨设备安全的NMS处理"""
    from mmcv.ops import nms
    unique_labels = labels.unique()
    keep = []

    for cls_id in unique_labels:
        cls_mask = (labels == cls_id)
        cls_bboxes = bboxes[cls_mask]
        cls_scores = scores[cls_mask]

        if cls_bboxes.shape[0] == 0:
            continue

        # 确保输入在相同设备
        device = cls_bboxes.device
        _, inds = nms(cls_bboxes.to(device), cls_scores.to(device), iou_threshold=iou_threshold)

        # 跨设备索引转换
        mask_indices = cls_mask.nonzero()[:, 0]
        keep.extend(mask_indices[inds].cpu().numpy())

    # 按分数排序（在CPU处理）
    scores_cpu = scores.cpu().numpy()
    return np.array(keep)[np.argsort(-scores_cpu[keep])]


def process_image(model, input_path, output_dir, window_size, overlap_ratio, score_thr, nms_iou):
    filename = os.path.basename(input_path)
    output_path = os.path.join(output_dir, filename)

    try:
        # 处理图片
        img = mmcv.imread(input_path, channel_order='rgb')
        result = sliding_window_inference(model, img, window_size=window_size, overlap_ratio=overlap_ratio,
                                          score_thr=score_thr, nms_iou=nms_iou)

        # 保存结果
        visualizer = DetLocalVisualizer()
        visualizer.dataset_meta = model.dataset_meta
        visualizer.add_datasample('detection', img, result,
                                  draw_pred=True, draw_gt=False,
                                  show=False, out_file=output_path)
        return True, len(result.pred_instances.bboxes)

    except Exception as e:
        print(f"\n处理 {filename} 时出错: {str(e)}")
        return False, 0


if __name__ == '__main__':
    # 配置路径（保持原有配置不变）
    base_dir = '/public/home/zncui/pycharm_project/mmdetection'
    config_file = os.path.join(base_dir,
                               'work_dirs/co_dino_5scale_swin_l_16xb1_16e_o365tococo/co_dino_5scale_swin_l_16xb1_16e_o365tococo.py')
    checkpoint_file = os.path.join(base_dir, 'work_dirs/co_dino_5scale_swin_l_16xb1_16e_o365tococo/epoch_16.pth')
    input_dir = '/public/lab_share/nfs_field/raw_data/DJI/P1/zhaoxian2022/20230526DJI/DCIM/DJI_202305260737_008_D9H400cm/'
    output_dir = '/public/home/zncui/pycharm_project/mmdetection/infer_dirs/DJI_202305260737_008_D9H400cm/'

    # 初始化目录和模型
    os.makedirs(output_dir, exist_ok=True)
    model = init_detector(config_file, checkpoint_file, device='cuda:0')

    # 获取文件列表
    jpg_files = glob.glob(os.path.join(input_dir, '*.JPG'))
    total_files = len(jpg_files)
    start_time = time.time()
    total_boxes = 0
    processed_count = 0

    print(f"开始处理 {total_files} 张图片...")

    # 使用线程池来并行处理图片
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = []
        for idx, input_path in enumerate(jpg_files, 1):
            futures.append(executor.submit(process_image, model, input_path, output_dir, window_size=640, overlap_ratio=0.2, score_thr=0.0, nms_iou=0.1))

        # 等待任务完成并计算统计
        for idx, future in enumerate(futures, 1):
            success, num_boxes = future.result()

            if success:
                total_boxes += num_boxes
                processed_count += 1

            # 计算时间
            elapsed = time.time() - start_time
            avg_time = elapsed / idx
            remaining = avg_time * (total_files - idx)

            # 打印进度信息
            progress_msg = (
                f"进度: {idx}/{total_files} "
                f"已用: {elapsed:.1f}s | 剩余: {remaining:.1f}s | "
                f"当前框: {num_boxes} | 总框: {total_boxes} | "
                f"{jpg_files[idx-1]}"
            )
            print(f"\r{progress_msg}", end="", flush=True)

    # 最终统计
    total_time = time.time() - start_time
    print(f"\n\n处理完成！共处理 {processed_count}/{total_files} 张图片")
    print(f"总预测框数量: {total_boxes}")
    print(f"总耗时: {total_time:.2f} 秒 (平均 {total_time / processed_count:.2f} 秒/张)")
