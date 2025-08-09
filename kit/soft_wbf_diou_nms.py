import torch
import numpy as np


def soft_nms(dets, scores, iou_thr=0.5, sigma=0.5, min_score=0.01, method='gaussian'):
    """Gaussian Soft-NMS (Non-Maximum Suppression)"""
    N = dets.size(0)
    # dets: [N, 4], [x1, y1, x2, y2], scores: [N]

    for i in range(N):
        for j in range(i + 1, N):
            # Ensure dets[i] and dets[j] are correctly sized tensors
            box1 = dets[i]  # [4]
            box2 = dets[j]  # [4]
            iou = bbox_iou(box1, box2)  # Calculate IoU

            if iou > iou_thr:
                if method == 'gaussian':
                    # Apply Gaussian penalty
                    weight = np.exp(-(iou ** 2) / sigma)
                elif method == 'linear':
                    # Apply linear penalty
                    weight = 1 - iou
                else:
                    weight = 0

                # Adjust scores based on IoU
                if scores[j] > scores[i]:
                    scores[j] *= weight
                else:
                    scores[i] *= weight

    # Remove low-score boxes
    keep = scores > min_score
    return keep.nonzero().squeeze(), scores[keep]


def bbox_iou(box1, box2):
    """Calculate IoU between two boxes"""
    # Ensure that box1 and box2 are 1D tensors of shape [4]
    assert box1.shape[0] == 4, f"Expected shape [4] but got {box1.shape}"
    assert box2.shape[0] == 4, f"Expected shape [4] but got {box2.shape}"

    # box: [x1, y1, x2, y2]
    inter_x1 = torch.max(box1[0], box2[0])
    inter_y1 = torch.max(box1[1], box2[1])
    inter_x2 = torch.min(box1[2], box2[2])
    inter_y2 = torch.min(box1[3], box2[3])

    inter_area = torch.max(inter_x2 - inter_x1, torch.tensor(0.0)) * torch.max(inter_y2 - inter_y1, torch.tensor(0.0))
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area


def batched_soft_nms(bboxes, scores, labels, iou_threshold=0.5, sigma=0.5, min_score=0.01):
    """对不同类别的检测框分别做 Soft-NMS，然后根据分数排序合并结果。"""
    unique_labels = labels.unique()
    keep = []

    for cls_id in unique_labels:
        cls_mask = (labels == cls_id)
        cls_bboxes = bboxes[cls_mask]
        cls_scores = scores[cls_mask]

        if cls_bboxes.shape[0] == 0:
            continue

        device = cls_bboxes.device
        # Soft-NMS 基于 GPU 运算，可显著加速
        keep_indices, updated_scores = soft_nms(cls_bboxes.to(device),
                                                cls_scores.to(device),
                                                iou_thr=iou_threshold,
                                                sigma=sigma,
                                                min_score=min_score,
                                                method='gaussian')
        mask_indices = cls_mask.nonzero()[:, 0]

        # Ensure keep_indices is iterable and not a 0-dimensional tensor
        if isinstance(keep_indices, torch.Tensor):
            keep_indices = keep_indices.cpu().numpy()
        if keep_indices.ndim == 0:
            keep_indices = [keep_indices.item()]  # Convert to list if it's scalar

        # Now, we can safely iterate over keep_indices
        keep.extend(mask_indices[keep_indices].cpu().numpy())

    scores_cpu = scores.cpu().numpy()
    # 根据分数降序排序
    return np.array(keep)[np.argsort(-scores_cpu[keep])]


# Example Usage
bboxes = torch.tensor([[50, 50, 150, 150], [60, 60, 160, 160], [200, 200, 250, 250]], dtype=torch.float32)
scores = torch.tensor([0.9, 0.8, 0.7])
labels = torch.tensor([1, 1, 2])

keep = batched_soft_nms(bboxes, scores, labels, iou_threshold=0.5, sigma=0.5, min_score=0.01)
print("Kept boxes:", keep)
