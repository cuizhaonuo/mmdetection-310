import os
from os import makedirs
import shutil
import mmcv
import torch
import glob
import time
import numpy as np
import pandas as pd
from mpmath import linspace
import math
from tqdm import tqdm
from mmdet.apis import init_detector, inference_detector
from mmdet.visualization import DetLocalVisualizer
from mmengine.structures import InstanceData
from mmdet.structures import DetDataSample
from mmcv.ops import soft_nms, nms
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# from tools.yolo2coco import coco_annotations

# 加快推理速度的设置
torch.backends.cudnn.benchmark = True

def sliding_window_inference(model, img, window_size=1024, overlap_ratio=0.1,
                             score_thr=0.0, nms_iou=0.1, iou_type='IoU'):
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
            # CoDETR, (649, 640, 3) --> DetDataSample
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
        # 1) 创建一个空的 InstanceData
        empty_inst = InstanceData()
        # 2) 给它加上 bboxes / scores / labels
        empty_inst.bboxes = torch.zeros((0, 4), dtype=torch.float32)
        empty_inst.scores = torch.zeros((0,), dtype=torch.float32)
        empty_inst.labels = torch.zeros((0,), dtype=torch.int64)
        # 3) 放进 DetDataSample 里
        empty_result = DetDataSample()
        empty_result.pred_instances = empty_inst
        return empty_result

    # 合并所有窗口的结果并做 NMS
    merged_instances = InstanceData.cat(all_detections)
    keep = batched_nms(merged_instances.bboxes, merged_instances.scores,
                       merged_instances.labels, nms_iou, iou_type=iou_type)
    final_result = DetDataSample()
    final_result.pred_instances = merged_instances[keep[:]].cpu()
    return final_result

def batched_nms(bboxes, scores, labels, iou_threshold, iou_type='IoU'):
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
        inds = custom_nms(cls_bboxes.to(device),
                      cls_scores.to(device),
                      iou_threshold=iou_threshold,
                      iou_type=iou_type)
        mask_indices = cls_mask.nonzero()[:, 0]
        keep.extend(mask_indices[inds].cpu().numpy())

    scores_cpu = scores.cpu().numpy()
    # 根据分数降序排序
    return np.array(keep)[np.argsort(-scores_cpu[keep])]




def custom_soft_nms(bboxes, scores, iou_threshold=0.3, sigma=0.5, score_threshold=0.001):
    """
    Soft-NMS 算法实现。

    参数:
        bboxes (Tensor): [N, 4] 的张量，表示 N 个边界框，每个边界框格式为 (x1, y1, x2, y2)。
        scores (Tensor): [N] 的张量，表示每个边界框的置信分数。
        iou_threshold (float): IOU 阈值，用于抑制的最小交并比。
        sigma (float): 高斯加权的标准差。
        score_threshold (float): 最低分数阈值，低于该值的框会被移除。

    返回:
        keep (Tensor): 保留的边界框索引。
    """
    if bboxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64)

    # 获取边界框的坐标
    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    x2 = bboxes[:, 2]
    y2 = bboxes[:, 3]

    # 计算每个边界框的面积
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    # 根据分数降序排序
    _, order = scores.sort(0, descending=True)

    keep = []
    while order.numel() > 0:
        i = order[0].item()  # 当前最高分数的索引
        keep.append(i)

        if order.numel() == 1:
            break

        # 计算当前框与剩余框的交集
        xx1 = torch.max(x1[i], x1[order[1:]])
        yy1 = torch.max(y1[i], y1[order[1:]])
        xx2 = torch.min(x2[i], x2[order[1:]])
        yy2 = torch.min(y2[i], y2[order[1:]])

        # 计算交集的宽高
        w = torch.clamp(xx2 - xx1 + 1, min=0)
        h = torch.clamp(yy2 - yy1 + 1, min=0)

        # 计算交集面积
        inter = w * h

        # 计算 IOU
        iou = inter / (areas[i] + areas[order[1:]] - inter)

        # 根据 IOU 更新分数
        weights = torch.exp(-(iou ** 2) / sigma)
        scores[order[1:]] *= weights

        # 移除分数低于阈值的框
        inds = (scores[order[1:]] > score_threshold).nonzero(as_tuple=False).squeeze()
        order = order[inds + 1]

    return torch.tensor(keep, dtype=torch.int64)



def custom_nms(bboxes, scores, iou_threshold, iou_type='IoU'):
    if bboxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=bboxes.device)

    x1, y1, x2, y2 = bboxes.unbind(dim=1)
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    _, order = scores.sort(descending=True)

    keep = []
    eps = 1e-7

    while order.numel() > 0:
        i = order[0].item()
        keep.append(i)
        if order.numel() == 1:
            break

        idxs = order[1:]
        # intersection
        xx1 = torch.max(x1[i], x1[idxs])
        yy1 = torch.max(y1[i], y1[idxs])
        xx2 = torch.min(x2[i], x2[idxs])
        yy2 = torch.min(y2[i], y2[idxs])
        w = (xx2 - xx1 + 1).clamp(min=0)
        h = (yy2 - yy1 + 1).clamp(min=0)
        inter = w * h

        # basic union and IoU
        union = areas[i] + areas[idxs] - inter
        iou = inter / union.clamp(min=eps)

        if iou_type == 'IoU':
            ious = iou
        elif iou_type == 'IoU_min':
            ious = inter / torch.min(areas[i], areas[idxs]).clamp(min=eps)

        elif iou_type == 'GIoU':
            # smallest enclosing box
            cx1 = torch.min(x1[i], x1[idxs])
            cy1 = torch.min(y1[i], y1[idxs])
            cx2 = torch.max(x2[i], x2[idxs])
            cy2 = torch.max(y2[i], y2[idxs])
            cw = (cx2 - cx1 + 1).clamp(min=0)
            ch = (cy2 - cy1 + 1).clamp(min=0)
            c_area = cw * ch
            ious = iou - (c_area - union) / c_area.clamp(min=eps)

        elif iou_type in ('DIoU', 'CIoU', 'EIoU'):
            # center distance squared
            bxi = (x1[i] + x2[i]) / 2
            byi = (y1[i] + y2[i]) / 2
            bxj = (x1[idxs] + x2[idxs]) / 2
            byj = (y1[idxs] + y2[idxs]) / 2
            dist_sq = (bxi - bxj)**2 + (byi - byj)**2

            # diagonal squared of enclosing box
            ex1 = torch.min(x1[i], x1[idxs])
            ey1 = torch.min(y1[i], y1[idxs])
            ex2 = torch.max(x2[i], x2[idxs])
            ey2 = torch.max(y2[i], y2[idxs])
            diag_sq = ((ex2 - ex1)**2 + (ey2 - ey1)**2).clamp(min=eps)

            if iou_type == 'DIoU':
                ious = iou - dist_sq / diag_sq

            else:
                # widths & heights
                wi = x2[i] - x1[i] + 1
                hi = y2[i] - y1[i] + 1
                wj = x2[idxs] - x1[idxs] + 1
                hj = y2[idxs] - y1[idxs] + 1

                if iou_type == 'CIoU':
                    v = (4 / (math.pi**2)) * (torch.atan(wj/(hj+eps)) - torch.atan(wi/(hi+eps)))**2
                    alpha = v / ((1 - iou) + v + eps)
                    ious = iou - (dist_sq / diag_sq + alpha * v)

                else:  # 'EIoU'
                    cw = (torch.max(x2[i], x2[idxs]) - torch.min(x1[i], x1[idxs]) + 1).clamp(min=eps)
                    ch = (torch.max(y2[i], y2[idxs]) - torch.min(y1[i], y1[idxs]) + 1).clamp(min=eps)
                    w_diff = (wj - wi)**2
                    h_diff = (hj - hi)**2
                    ious = iou - (dist_sq/diag_sq + w_diff/(cw**2) + h_diff/(ch**2))

        else:
            raise ValueError(f"Unknown iou_type: {iou_type}")

        # keep boxes with metric ≤ threshold
        mask = ious <= iou_threshold
        order = idxs[mask]

    return torch.as_tensor(keep, dtype=torch.int64, device=bboxes.device)




if __name__ == '__main__':
    # 配置文件和权重
    base_dir = '/public/home/zncui/pycharm_project/mmdetection'
    # config_file = os.path.join(base_dir,
    #                            'work_dirs/co_dino_5scale_swin_l_16xb1_16e_o365tococo/co_dino_5scale_swin_l_16xb1_16e_o365tococo_pred.py')
    # checkpoint_file = os.path.join(base_dir, 'work_dirs/co_dino_5scale_swin_l_16xb1_16e_o365tococo/epoch_16.pth')
    config_file = os.path.join(base_dir,
                               'work_dirs/co_dino_guo_yolo2coco_1/co_dino_5scale_swin_l_16xb1_16e_o365tococo.py')
    checkpoint_file = os.path.join(base_dir, 'work_dirs/co_dino_guo_yolo2coco_1/epoch_16.pth')

    # 输入和输出目录
    # 本次输入是 20 张验证集
    dataset_dir = '/public/home/zncui/datasets/wheat-guo-dataset'
    input_dir = os.path.join(dataset_dir, 'yolo_wheat_big_image', 'images', 'val')
    os.makedirs(input_dir, exist_ok=True)
    output_dir = os.path.join(dataset_dir, 'yolo_wheat_big_image_results')
    os.makedirs(output_dir, exist_ok=True)

    # 使用 GPU:1
    model = init_detector(config_file, checkpoint_file, device='cuda:2')

    # 一次性创建可视化对象，避免在循环中重复创建
    visualizer = DetLocalVisualizer()
    visualizer.dataset_meta = model.dataset_meta

    all_image_files = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                all_image_files.append(os.path.join(root, file))

    # Test
    all_image_files = sorted(all_image_files)[:2]

    # all_image_files = [f for f in all_image_files if os.path.basename(f) == '23B345.jpg']

    total_files = len(all_image_files)
    start_time = time.time()

    print(f"开始处理 {total_files} 张图片...")

    # iou_type = ['IoU', 'GIoU', 'DIoU', 'CIoU', 'EIoU']
    iou_type_list = ['IoU_min']

    # real_counts = 10819 - 200

    # 在推理前后使用 no_grad, 避免不必要的梯度计算
    with torch.no_grad():
        # all_data = []
        min_diff = float('inf')
        cur_nms_threshold = -1
        cur_score_threshold = -1
        for iou_type in iou_type_list:
            for nms_threshold in np.linspace(0.2, 0.6, 2):
                for score_threshold in np.linspace(0.2, 0.6, 2):
                    # pred_counts = 0
                    print(f"IOU: {iou_type}, NMS阈值: {float(nms_threshold):.2f}, 分数阈值: {float(score_threshold):.2f}")

                    # coco_annotations = {
                    #     'images': [],
                    #     'annotations': [],
                    #     'categories': [{'id': 1, 'name': 'wheat'}]
                    # }
                    for idx, input_path in enumerate(tqdm(all_image_files, desc="Processing images"), 1):
                        filename = os.path.basename(input_path)
                        basename = os.path.splitext(filename)[0]
                        try:
                        # for i in range(1):
                            output_ori_image_path = os.path.join(output_dir, 'ori', filename)
                            makedirs(os.path.join(output_dir, 'ori'), exist_ok=True)
                            # output_rect_image_path = os.path.join(output_dir, 'rect', filename)
                            # makedirs(os.path.join(output_dir, 'rect'), exist_ok=True)
                            # output_circle_image_path = os.path.join(output_dir, 'circle', filename)
                            # makedirs(os.path.join(output_dir, 'circle'), exist_ok=True)
                            output_txt_image_path = os.path.join(output_dir, 'txt', f'{basename}.txt')
                            makedirs(os.path.join(output_dir, 'txt'), exist_ok=True)
                            # output_index_path = os.path.join(output_dir, 'index', filename)
                            # makedirs(os.path.join(output_dir, 'index'), exist_ok=True)

                            # 将原图复制到输出目录
                            shutil.copy2(input_path, output_ori_image_path)

                            # 保存结果到 Excel 和 TXT
                            # results_dir = os.path.join(output_dir, 'results')
                            # os.makedirs(results_dir, exist_ok=True)

                            # 读取图像
                            img = mmcv.imread(input_path, channel_order='rgb')

                            # 滑窗推理
                            # 已经证实预测的框是 xyxy 格式
                            result = sliding_window_inference(
                                model, img,
                                window_size=1024,    # 调大滑窗尺寸
                                overlap_ratio=0.15,   # 减少重叠
                                score_thr=score_threshold,
                                nms_iou=nms_threshold,
                                iou_type=iou_type
                            )

                            pred_instances = result.pred_instances
                            bboxes = pred_instances.bboxes.cpu().numpy()
                            scores = pred_instances.scores.cpu().numpy()
                            labels = pred_instances.labels.cpu().numpy()

                            # pred_counts += len(bboxes)

                            # 可视化并保存结果
                            # visualizer.add_datasample(
                            #     'detection', img, result,
                            #     draw_pred=True, draw_gt=False,
                            #     show=False, out_file=output_rect_image_path
                            # )

                            # 绘制圆点
                            # import cv2
                            #
                            # img2 = img.copy()  # assume img is a NumPy array H×W×3, dtype=uint8
                            # for bbox in bboxes:
                            #     x0, y0, x1, y1 = bbox
                            #     cx, cy = int((x0 + x1) / 2), int((y0 + y1) / 2)
                            #     cv2.circle(img2, (cx, cy), radius=5, color=(255, 0, 0), thickness=-1, lineType=cv2.LINE_AA)
                            # img2_bgr = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)
                            # cv2.imwrite(output_circle_image_path, img2_bgr)
                            #
                            # # 带框+编号的版本
                            # img3 = img.copy()
                            # for i, bbox in enumerate(bboxes):
                            #     x0, y0, x1, y1 = [int(v) for v in bbox]
                            #     cv2.rectangle(img3, (x0, y0), (x1, y1), (0, 255, 0), 1, lineType=cv2.LINE_AA)
                            #     cv2.putText(img3, str(i), (x0, max(y0 - 5, 0)),
                            #                 cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, lineType=cv2.LINE_AA)
                            # img3_bgr = cv2.cvtColor(img3, cv2.COLOR_RGB2BGR)
                            # cv2.imwrite(output_index_path, img3_bgr)



                            # 写入 Excel (只记录框数和路径信息，并根据 iou_type 保存文件)
                            # all_data.append({
                            #     'file_name': filename,
                            #     'iou_type': iou,
                            #     'counts': len(bboxes),
                            #     'nms_threshold': float(nms_threshold),
                            #     'score_threshold': float(score_threshold),
                            # })

                            # 写入 TXT (详细的坐标信息)
                            # txt_path = output_txt_image_path
                            # with open(txt_path, 'w') as f_txt:
                            #     for i, bbox in enumerate(bboxes):
                            #         x_min, y_min, x_max, y_max = bbox
                            #         x, y, w, h = map(int, (x_min, y_min, x_max - x_min, y_max - y_min))
                            #         # 左上角坐标和宽高格式
                            #         f_txt.write(f"{i} {x} {y} {w} {h}\n")

                            # 测试
                            output_txt_image_path = os.path.join(output_dir, 'txt', f'nms_{nms_threshold}_score_{score_threshold}', f'{basename}.txt')
                            os.makedirs(os.path.join(output_dir, 'txt', f'nms_{nms_threshold}_score_{score_threshold}'), exist_ok=True)
                            txt_path = output_txt_image_path
                            with open(txt_path, 'w') as f_txt:
                                for i, bbox in enumerate(bboxes):
                                    x_min, y_min, x_max, y_max = bbox
                                    # yolo 格式
                                    x_center = (x_min + x_max) / 2
                                    y_center = (y_min + y_max) / 2
                                    width = x_max - x_min
                                    height = y_max - y_min
                                    # 注意：yolo 格式的坐标需要归一化到 [0, 1] 范围
                                    x_center /= img.shape[1]
                                    y_center /= img.shape[0]
                                    width /= img.shape[1]
                                    height /= img.shape[0]
                                    f_txt.write(f"0 {x_center} {y_center} {width} {height}\n")

                            # image = {
                            #     'file_name': filename,
                            #     'height': img.shape[0],
                            #     'width': img.shape[1],
                            #     'id': 556556 + idx,  # 假设从556556开始编号
                            # }
                            # coco_annotations['images'].append(image)
                            #
                            # for bi, bbox in enumerate(bboxes):
                            #     x_min, y_min, x_max, y_max = bbox
                            #     x, y, w, h = map(float, (x_min, y_min, x_max - x_min, y_max - y_min))
                            #     annotation = {
                            #         'id': 556556 + bi,  # 假设从556556开始编号
                            #         'image_id': 556556 + idx,
                            #         'category_id': 1,  # 假设只有一个类别
                            #         'segmentation': [],
                            #         'area': 0.0,
                            #         'bbox': [x, y, w, h],
                            #         'iscrowd': 0,
                            #     }
                            #     coco_annotations['annotations'].append(annotation)


                        except Exception as e:
                            print(f"\n处理 {filename} 时出错: {str(e)}")


                    print(f"已处理 {idx}/{total_files} 张图片，耗时: {time.time() - start_time:.2f} 秒")
                    # 计算召回率和准确率
                    # label_instances_dir = os.path.join(dataset_dir, 'yolo_wheat_big_image', 'labels', 'val')
                    # label_instances_files = [f for f in glob.glob(os.path.join(label_instances_dir, '*.txt')) if
                    #                          os.path.basename(f) != 'classes.txt']
                    pred_instances_dir = os.path.join(output_dir, 'txt', f'nms_{nms_threshold}_score_{score_threshold}')
                    pred_instances_files = glob.glob(os.path.join(pred_instances_dir, '*.txt'))
                    total_diff_counts = 0
                    total_tp = 0
                    total_fp = 0
                    total_fn = 0
                    total_gt_boxes = 0
                    total_pred_boxes = 0

                    for pred_file in pred_instances_files:
                        filename = os.path.basename(pred_file)
                        label_file = os.path.join(dataset_dir, 'yolo_wheat_big_image', 'labels', 'val', filename)
                        if not os.path.exists(label_file):
                            break
                        img_file = label_file.replace('labels', 'images').replace('.txt', '.jpg')
                        h, w = mmcv.imread(img_file, channel_order='rgb').shape[:2]
                        with open(label_file, 'r') as f_label, open(pred_file, 'r') as f_pred:
                            label_lines = f_label.readlines()
                            pred_lines = f_pred.readlines()

                            # 解析 YOLO 格式文件
                            gt_boxes = []
                            for line in label_lines:
                                parts = line.strip().split()
                                if len(parts) >= 5:  # 确保有足够的数据
                                    class_id, x_center, y_center, width, height = map(float, parts[:5])
                                    # 转换为绝对坐标
                                    x_center_abs = x_center * w
                                    y_center_abs = y_center * h
                                    width_abs = width * w
                                    height_abs = height * h
                                    gt_boxes.append([x_center_abs, y_center_abs, width_abs, height_abs])

                            pred_boxes = []
                            for line in pred_lines:
                                parts = line.strip().split()
                                if len(parts) >= 5:  # 确保有足够的数据
                                    class_id, x_center, y_center, width, height = map(float, parts[:5])
                                    # 转换为绝对坐标
                                    x_center_abs = x_center * w
                                    y_center_abs = y_center * h
                                    width_abs = width * w
                                    height_abs = height * h
                                    pred_boxes.append([x_center_abs, y_center_abs, width_abs, height_abs])

                            total_gt_boxes += len(gt_boxes)
                            total_pred_boxes += len(pred_boxes)


                            # 计算 IoU
                            def calculate_iou(box1, box2):
                                # YOLO格式 [x_center, y_center, width, height]
                                # 转换为 [x1, y1, x2, y2]
                                b1_x1, b1_y1 = box1[0] - box1[2] / 2, box1[1] - box1[3] / 2
                                b1_x2, b1_y2 = box1[0] + box1[2] / 2, box1[1] + box1[3] / 2
                                b2_x1, b2_y1 = box2[0] - box2[2] / 2, box2[1] - box2[3] / 2
                                b2_x2, b2_y2 = box2[0] + box2[2] / 2, box2[1] + box2[3] / 2

                                # 计算交集
                                inter_x1 = max(b1_x1, b2_x1)
                                inter_y1 = max(b1_y1, b2_y1)
                                inter_x2 = min(b1_x2, b2_x2)
                                inter_y2 = min(b1_y2, b2_y2)

                                # 交集面积
                                inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

                                # 各自面积
                                b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
                                b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)

                                # IoU
                                iou = inter_area / (b1_area + b2_area - inter_area + 1e-7)
                                return iou


                            # 使用 IoU 阈值匹配边界框
                            iou_threshold = nms_threshold  # 可调整的 IoU 阈值
                            matched_gt = set()
                            tp = 0  # 真正例
                            fp = 0  # 假正例

                            for pred_box in pred_boxes:
                                best_iou = 0
                                best_gt_idx = -1

                                for gt_idx, gt_box in enumerate(gt_boxes):
                                    if gt_idx in matched_gt:
                                        continue  # 跳过已匹配的真实框

                                    iou = calculate_iou(pred_box, gt_box)
                                    if iou > best_iou:
                                        best_iou = iou
                                        best_gt_idx = gt_idx

                                if best_iou >= iou_threshold:
                                    tp += 1
                                    matched_gt.add(best_gt_idx)
                                else:
                                    fp += 1

                            # 计算当前图片的指标
                            fn = len(gt_boxes) - len(matched_gt)  # 假负例

                            # 累加到总计数
                            total_tp += tp
                            total_fp += fp
                            total_fn += fn

                            # 计算单张图片的指标
                            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                            # print(
                            #     f"文件 {basename}: 精确率={precision:.4f}, 召回率={recall:.4f}, F1={f1:.4f}, 框数差距={abs(len(gt_boxes) - len(pred_boxes))}")

                    # 计算所有图片的汇总指标
                    total_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
                    total_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
                    total_f1 = 2 * total_precision * total_recall / (total_precision + total_recall) if (total_precision + total_recall) > 0 else 0

                    print("\n汇总指标:")
                    print(f"NMS阈值: {float(nms_threshold):.2f}, 分数阈值: {float(score_threshold):.2f}\n")
                    print(f"总精确率: {total_precision:.4f}")
                    print(f"总召回率: {total_recall:.4f}")
                    print(f"总F1分数: {total_f1:.4f}")
                    print(f"总预测框数: {total_pred_boxes}, 总真实框数: {total_gt_boxes}, 差距: {abs(total_gt_boxes - total_pred_boxes)}")
                    print(f"真正例(TP): {total_tp}, 假正例(FP): {total_fp}, 假负例(FN): {total_fn}")

                    if abs(total_gt_boxes - total_pred_boxes) < min_diff:
                        min_diff = abs(total_gt_boxes - total_pred_boxes)
                        cur_nms_threshold = nms_threshold
                        cur_score_threshold = score_threshold
                        print(f"当前最小差异: {min_diff}，NMS阈值: {float(cur_nms_threshold):.2f}, 分数阈值: {float(cur_score_threshold):.2f}")

                    # 将结果保存为 log 文件
                    log_path = os.path.join(output_dir, f'results.log')
                    with open(log_path, 'a') as log_file:
                        # log_file.write(f"IOU: {iou_type}, NMS阈值: {float(nms_threshold):.2f}, 分数阈值: {float(score_threshold):.2f}\n")
                        # log_file.write("汇总指标:\n")
                        # log_file.write(f"总精确率: {total_precision:.4f}\n")
                        # log_file.write(f"总召回率: {total_recall:.4f}\n")
                        # log_file.write(f"总F1分数: {total_f1:.4f}\n")
                        # log_file.write(f"总预测框数: {total_pred_boxes}, 总真实框数: {total_gt_boxes}, 差距: {abs(total_gt_boxes - total_pred_boxes)}\n")
                        # log_file.write(f"真正例(TP): {total_tp}, 假正例(FP): {total_fp}, 假负例(FN): {total_fn}\n")
                        # log_file.write(f"当前最小差异: {min_diff}，NMS阈值: {float(cur_nms_threshold):.2f}, 分数阈值: {float(cur_score_threshold):.2f}\n")
                        log_file.write("-" * 50 + "\n")


                    # 保存 COCO 格式的结果
                    # import json
                    # coco_output_path = os.path.join(output_dir, 'instances_ped.json')
                    # with open(coco_output_path, 'w', encoding='utf-8') as f:
                    #     json.dump(
                    #         coco_annotations,
                    #         f,
                    #         ensure_ascii=False,  # 保持中文可读
                    #         indent=2  # 缩进格式化，便于阅读
                    #     )

                    # print(f"处理完毕，IOU: {iou}, NMS阈值: {float(nms_threshold):.2f}, 分数阈值: {float(score_threshold):.2f}, 总框数: {pred_counts}, 实际框数: {real_counts}, 差异: {abs(pred_counts - real_counts)}")
                    #
                    # diff = abs(real_counts - pred_counts)
                    # if diff < min_diff:
                    #     min_diff = diff
                    #     print(f"当前最小差异: {min_diff}，IOU: {iou}, NMS阈值: {float(nms_threshold):.2f}, 分数阈值: {float(score_threshold):.2f}")






        # df = pd.DataFrame(all_data)
        # excel_path = os.path.join(results_dir, f"iou.xlsx")
        # df.to_excel(excel_path, index=False)

        # from mpl_toolkits.mplot3d import Axes3D

        # 绘制三维散点图
        # fig = plt.figure(figsize=(10, 6))
        # ax = fig.add_subplot(111, projection='3d')
        # sc = ax.scatter(df['score_threshold'], df['nms_threshold'], df['counts'], c=df['counts'], cmap='viridis', s=50, alpha=0.7)
        # fig.colorbar(sc, label='Counts')
        # ax.set_xlabel('Score Threshold')
        # ax.set_ylabel('NMS Threshold')
        # ax.set_zlabel('Counts')
        # ax.set_title('3D Scatter Plot of Score Threshold, NMS Threshold, and Counts')
        # scatter_plot_path = os.path.join(results_dir, 'scatter_plot_3d.png')
        # plt.savefig(scatter_plot_path)
        # plt.close()