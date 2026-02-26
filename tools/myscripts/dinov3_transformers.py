import torch
import mmcv

from transformers import AutoImageProcessor, ConvNextModel
from transformers.image_utils import load_image

from mmengine.visualization import Visualizer


model_id = "facebook/dinov3-convnext-tiny-pretrain-lvd1689m"

# 1) 读图
url = "/public/home/zncui/pycharm_project/mmdetection/demo/demo.jpg"
url = "/public/home/zncui/datasets/wheats_rle_coco/coco/train2017/D6-059-20230524.jpg"
image = load_image(url)  # -> PIL.Image

# 2) 预处理器
processor = AutoImageProcessor.from_pretrained(model_id)

# 3) 模型（开启输出中间层）
model = ConvNextModel.from_pretrained(model_id)
model.config.output_hidden_states = True  # 关键！
model.eval()

# 4) 前向 & 拿到每个 stage 的特征
inputs = processor(images=image, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)  # BaseModelOutputWithNoAttention
    # outputs.hidden_states: tuple[Tensor], 依次为 stem 与每个 stage 的输出
    # 约定：idx 0=stem，1~4=stage1..stage4
    hiddens = outputs.hidden_states

# 5) 组织为 res2~res5（与 ResNet 风格对应的下采样倍率）
# ConvNeXt 的约定：stage1~4 的特征图下采样倍率依次为 4, 8, 16, 32
stage_names = ["stem", "res2", "res3", "res4", "res5"]
features_per_stage = {name: feat for name, feat in zip(stage_names, hiddens)}

# 现在你可以按需取用：
res2 = features_per_stage["res2"]  # 形状: [B, C, H/4,  W/4 ]
res3 = features_per_stage["res3"]  # 形状: [B, C, H/8,  W/8 ]
res4 = features_per_stage["res4"]  # 形状: [B, C, H/16, W/16]
res5 = features_per_stage["res5"]  # 形状: [B, C, H/32, W/32]



print({k: tuple(v.shape) for k, v in features_per_stage.items()})

image = mmcv.imread(url, channel_order='rgb')
visualizer = Visualizer()
drawn_img = visualizer.draw_featmap(featmap=res4[0],
                                    overlaid_image=image,
                                    channel_reduction='select_max',
                                    # topk=12,
                                    # arrangement=(4, 3)
                                    )
visualizer.show(drawn_img)
