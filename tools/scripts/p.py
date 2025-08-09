import torch
from mmengine.structures import BaseDataElement
gt_instances = BaseDataElement()
bboxes = torch.rand((5, 4))
scores = torch.rand((5,))
img_id = 0
img_shape = (800, 1333)
gt_instances = BaseDataElement(
    metainfo=dict(img_id=img_id, img_shape=img_shape),
    bboxes=bboxes, scores=scores)
# gt_instances = BaseDataElement(
#     metainfo=dict(img_id=img_id, img_shape=(640, 640)))
print(gt_instances)