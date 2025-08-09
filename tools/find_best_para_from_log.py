import re

log_path = '/public/home/zncui/pycharm_project/mmdetection/tools/infer_area_batch.log'

# 示例
# '处理完毕，IOU: IoU_min, NMS阈值: 0.46, 分数阈值: 0.54, 总框数: 9007, 实际框数: 10619, 差异: 1612'
pattern = re.compile('处理完毕，IOU: (.*), NMS阈值: (.*), 分数阈值: (.*), 总框数: (.*), 实际框数: (.*), 差异: (.*)')

with open(log_path, 'r') as f:
    lines = f.readlines()
    lines = [line.strip() for line in lines if line.strip()]
    for line in lines:
        match = pattern.search(line)
        if match:
            iou, nms_threshold, score_threshold, total_boxes, actual_boxes, diff = match.groups()
            print(f'IoU: {iou}, NMS阈值: {nms_threshold}, 分数阈值: {score_threshold}, 总框数: {total_boxes}, 实际框数: {actual_boxes}, 差异: {diff}')

# 最终结果:
# score_threshold: 0.19, nms_threshold: 0.37, diff: 4
# score_threshold: 0.10, nms_threshold: 0.10, diff: 6

