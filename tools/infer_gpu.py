import os
import mmcv
import torch
import numpy as np
from mmdet.apis import init_detector, inference_detector
from mmdet.visualization import DetLocalVisualizer
from mmengine.structures import InstanceData
from mmdet.structures import DetDataSample


def sliding_window_inference(model, img_path, device, window_size=640, overlap_ratio=0.5,
                             score_thr=0.0, nms_iou=0.3, batch_size=4):
    """符合MMDetection数据流的滑动窗口推理"""
    # 读取原始图像信息
    img = mmcv.imread(img_path)
    h, w = img.shape[:2]
    stride = int(window_size * (1 - overlap_ratio))
    all_detections = []
    batch_crops = []

    # 生成窗口坐标
    windows = [(y, x, min(y + window_size, h), min(x + window_size, w))
               for y in range(0, h, stride)
               for x in range(0, w, stride)]

    with torch.no_grad():
        for i, (y, x, y_end, x_end) in enumerate(windows):
            # 构建符合test pipeline的数据格式
            data = {
                'img_path': img_path,
                'img': img[y:y_end, x:x_end, :],
                'ori_shape': (y_end - y, x_end - x, 3),
                'img_shape': (y_end - y, x_end - x, 3),
                'scale_factor': 1.0
            }
            batch_crops.append((y, x, data))

            if len(batch_crops) == batch_size or i == len(windows) - 1:
                # 批量推理
                batch_results = []
                for _, _, crop_data in batch_crops:
                    result = inference_detector(model, crop_data)
                    batch_results.append(result)

                # 处理结果
                for idx, (y_b, x_b, _) in enumerate(batch_crops):
                    instances = batch_results[idx].pred_instances
                    valid_mask = instances.scores > score_thr
                    instances = instances[valid_mask]

                    if len(instances) > 0:
                        instances.bboxes[:, 0::2] += x_b
                        instances.bboxes[:, 1::2] += y_b
                        instances.bboxes[:, 0::2] = torch.clamp(instances.bboxes[:, 0::2], 0, w)
                        instances.bboxes[:, 1::2] = torch.clamp(instances.bboxes[:, 1::2], 0, h)
                        all_detections.append(instances.cpu())

                batch_crops.clear()

    if not all_detections:
        return DetDataSample()

    merged_instances = InstanceData.cat(all_detections).to(device)
    keep = batched_nms(merged_instances.bboxes, merged_instances.scores,
                       merged_instances.labels, nms_iou)

    final_result = DetDataSample()
    final_result.pred_instances = merged_instances[keep[:1000]].cpu()
    return final_result


def batched_nms(bboxes, scores, labels, iou_threshold):
    from mmcv.ops import nms
    unique_labels = torch.unique(labels)
    keep = []

    for cls_id in unique_labels:
        cls_mask = labels == cls_id
        if not cls_mask.any():
            continue

        cls_bboxes = bboxes[cls_mask]
        cls_scores = scores[cls_mask]
        _, inds = nms(cls_bboxes, cls_scores, iou_threshold=iou_threshold)

        mask_indices = torch.where(cls_mask)[0]
        keep.append(mask_indices[inds])

    return torch.cat(keep).cpu().numpy() if keep else np.array([], dtype=int)


# 配置路径
base_dir = '/public/home/zncui/pycharm_project/mmdetection'
in_img = '/public/lab_share/nfs_field/level_1/DJI/P1/zhaoxian2021/20220528/706D6point4m/D6-002-R01L12/DJI_20220528113109_0004_0.png'
out_img = '/public/home/zncui/pycharm_project/mmdetection/infer_dirs/DJI_20220528113109_0004_0-out.png'

if __name__ == '__main__':
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    # 初始化模型（保持原始精度）
    config_file = os.path.join(base_dir,
                               'work_dirs/co_dino_5scale_swin_l_16xb1_16e_o365tococo/co_dino_5scale_swin_l_16xb1_16e_o365tococo.py')
    checkpoint_file = r'/public/home/zncui/pycharm_project/mmdetection/work_dirs/co_dino_5scale_swin_l_16xb1_16e_o365tococo/epoch_16.pth'
    model = init_detector(config_file, checkpoint_file, device=device)

    # 执行推理（输入图像路径而非张量）
    result = sliding_window_inference(model, in_img, device=device, batch_size=8)

    # 可视化
    visualizer = DetLocalVisualizer()
    visualizer.dataset_meta = model.dataset_meta
    visualizer.add_datasample(
        'detection',
        mmcv.imread(in_img),
        result,
        draw_pred=True,
        show=True,
        out_file=out_img
    )