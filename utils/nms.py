import numpy as np


def intersection_over_union(boxA, boxB):
    # 计算两个边界框的交并比(IOU)
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


def non_max_suppression(boxes, scores, iou_threshold=0.5):
    """
    实现非极大值抑制(NMS)，输入是边界框和对应的分数，
    返回经过NMS处理后的边界框列表。
    """
    # 根据分数排序
    sorted_indices = np.argsort(scores)[::-1]

    keep_boxes = []
    while sorted_indices.size > 0:
        # 选择当前最高分的框
        idx = sorted_indices[0]
        keep_boxes.append(idx)

        # 计算当前框与其他所有框的IOU
        ious = np.array([intersection_over_union(boxes[idx], boxes[i]) for i in sorted_indices[1:]])

        # 删除与当前框IOU大于阈值的框
        remove_indices = np.where(ious > iou_threshold)[0] + 1  # +1是因为我们忽略了第一个元素（当前最高分的框）
        sorted_indices = np.delete(sorted_indices, remove_indices)
        sorted_indices = np.delete(sorted_indices, 0)  # 移除已经处理过的最高分框的索引

    return keep_boxes


# 示例用法
if __name__ == "__main__":
    # 单类别应用NMS
    # np.array()  创建numpy数组
    boxes = np.array([[10, 10, 40, 40], [11, 12, 43, 43], [9, 9, 39, 38]])  # [xmin, ymin, xmax, ymax]
    scores = np.array([0.9, 0.8, 0.7])  # 每个框的置信度
    iou_thresh = 0.1  # iou阈值

    # 应用NMS
    indices_to_keep = non_max_suppression(boxes, scores, iou_threshold=iou_thresh)
    print("保留的边界框索引:", indices_to_keep)