import torch
import mmcv
from mmengine.visualization import Visualizer
from mmdet.models.backbones import DINOv3_UNet

img_path = 'demo/demo.jpg'

image = mmcv.imread(img_path, channel_order='rgb')
visualizer = Visualizer(image=image)

visualizer.show()




