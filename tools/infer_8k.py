import os
import mmcv
import torch
import numpy as np
from mmdet.apis import init_detector, inference_detector
from mmdet.visualization import DetLocalVisualizer
from mmengine.structures import InstanceData
from mmdet.structures import DetDataSample  # 新增导入DetDataSample
import time

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
    # final_result.pred_instances = merged_instances[keep[:1000]].cpu()  # 确保在CPU
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


# 配置路径和主函数保持不变
base_dir = '/public/home/zncui/pycharm_project/mmdetection'
in_img = '/public/lab_share/nfs_field/level_1/DJI/P1/zhaoxian2022/20230208/706D6H400cm/D6-001-R01L13/DJI_20230208160419_0106_0.png'
out_img = '/public/home/zncui/pycharm_project/mmdetection/infer_dirs/DJI_20230208160419_0106_0-out.png'

if __name__ == '__main__':
    config_file = os.path.join(base_dir,
                               'work_dirs/co_dino_5scale_swin_l_16xb1_16e_o365tococo/co_dino_5scale_swin_l_16xb1_16e_o365tococo.py')
    checkpoint_file = r'/public/home/zncui/pycharm_project/mmdetection/work_dirs/co_dino_5scale_swin_l_16xb1_16e_o365tococo/epoch_16.pth'

    model = init_detector(config_file, checkpoint_file, device='cuda:0')
    img = mmcv.imread(in_img, channel_order='rgb')

    # 获取DetDataSample格式的结果
    start_time = time.time()
    result = sliding_window_inference(model, img, window_size=640, overlap_ratio=0.2, score_thr=0.0, nms_iou=0.1)
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"执行时间: {execution_time:.4f} 秒")

    print('预测框数量', len(result.pred_instances.bboxes))

    # 可视化（不再需要手动包装字典）
    visualizer = DetLocalVisualizer()
    visualizer.dataset_meta = model.dataset_meta
    visualizer.add_datasample(
        'detection',
        img,
        result,  # 直接传入DetDataSample对象
        draw_pred=True,
        draw_gt=False,
        show=False,
        out_file=out_img,
        pred_score_thr=0.1
    )