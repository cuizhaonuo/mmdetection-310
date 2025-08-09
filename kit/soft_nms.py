import numpy as np
import torch
from torchvision.ops import nms


def soft_nms(boxes, scores, sigma=0.5, threshold=0.3):
    """
    Implement Soft-NMS.
    boxes: Tensor of shape (N, 4), where N is the number of boxes, each box represented by [x1, y1, x2, y2]
    scores: Tensor of shape (N,), confidence scores corresponding to each box
    sigma: The parameter that controls the decay rate
    threshold: IoU threshold to filter boxes

    Returns:
        The indices of boxes to keep after applying Soft-NMS.
    """
    # Number of boxes
    N = boxes.size(0)

    # Iterate through all boxes
    for i in range(N):
        # Compare box[i] with other boxes
        iou = box_iou(boxes[i], boxes)  # compute IoU between box[i] and all other boxes

        # Reduce the score of overlapping boxes
        for j in range(i + 1, N):
            if iou[j] > threshold:
                scores[j] = scores[j] * np.exp(- (iou[j] ** 2) / sigma)

    # Select boxes based on their final scores
    keep = scores > 0.5  # Threshold for keeping boxes
    return keep


def box_iou(box1, boxes):
    """
    Compute IoU between a single box and an array of boxes
    """
    x1, y1, x2, y2 = box1
    xx1, yy1, xx2, yy2 = boxes.T

    # Compute the area of intersection
    inter_x1 = torch.max(x1, xx1)
    inter_y1 = torch.max(y1, yy1)
    inter_x2 = torch.min(x2, xx2)
    inter_y2 = torch.min(y2, yy2)

    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)

    # Compute the area of the boxes
    area_box1 = (x2 - x1) * (y2 - y1)
    area_boxes = (xx2 - xx1) * (yy2 - yy1)

    # Compute IoU
    union_area = area_box1 + area_boxes - inter_area
    return inter_area / union_area


# Example usage
boxes = torch.tensor([[10, 20, 50, 50], [12, 22, 48, 48], [60, 60, 100, 100]], dtype=torch.float)
scores = torch.tensor([0.9, 0.8, 0.7])

# Apply Soft-NMS
keep = soft_nms(boxes, scores)

# Keep only selected boxes
selected_boxes = boxes[keep]
selected_scores = scores[keep]

print("Selected boxes:", selected_boxes)
print("Selected scores:", selected_scores)
