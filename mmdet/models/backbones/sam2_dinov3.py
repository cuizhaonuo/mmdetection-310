import torch
import torch.nn as nn
import torch.nn.functional as F

from mmengine.model import BaseModule
from mmdet.registry import MODELS  # mmdet>=3.x

from transformers import ConvNextModel


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)

    def forward(self, x1, x2=None):
        if x2 is not None:
            diffY = x1.size()[2] - x2.size()[2]
            diffX = x1.size()[3] - x2.size()[3]
            x2 = F.pad(x2, [diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2])
            x = torch.cat([x1, x2], dim=1)
        else:
            x = x1
        x = self.up(x)
        return self.conv(x)


class DINOv3_UNet(nn.Module):
    def __init__(self, dinov3_weight_path=None, dinov3_local_path="./"):
        super().__init__()

        self.dino = torch.hub.load(
            repo_or_dir=dinov3_local_path,
            model="dinov3_vit7b16",   # 7B: embed_dim=4096
            source="local",
            pretrained=False,
            trust_repo=True
        )
        if dinov3_weight_path:
            checkpoint = torch.load(dinov3_weight_path, map_location='cpu', weights_only=True)
            self.dino.load_state_dict(checkpoint, strict=True)
            print("✓ Local weights successfully loaded")

        # 冻结 backbone
        for p in self.dino.parameters():
            p.requires_grad = False

        # 关键修改：动态获取通道数
        in_ch = getattr(self.dino, 'embed_dim', None)
        if in_ch is None:
            raise AttributeError("Backbone has no attribute 'embed_dim'. Please check the DINOv3 hub model.")

        self.reduce1 = nn.Conv2d(in_ch, 128, 1)
        self.reduce2 = nn.Conv2d(in_ch, 128, 1)
        self.reduce3 = nn.Conv2d(in_ch, 128, 1)
        self.reduce4 = nn.Conv2d(in_ch, 128, 1)

        self.up1 = Up(256, 128)
        self.up2 = Up(256, 128)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 128)
        self.head = nn.Conv2d(128, 1, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        feats = self.dino.forward_features(x)['x_norm_patchtokens']  # [B, N, C(embed_dim)]
        # patch_size=16 时，N=(H/16)*(W/16)
        feats = feats.view(B, H // 16, W // 16, -1).permute(0, 3, 1, 2)  # [B, C, H/16, W/16]

        x1 = F.interpolate(self.reduce1(feats), size=(H // 4,  W // 4),  mode='bilinear')
        x2 = F.interpolate(self.reduce2(feats), size=(H // 8,  W // 8),  mode='bilinear')
        x3 = F.interpolate(self.reduce3(feats), size=(H // 16, W // 16), mode='bilinear')
        x4 = F.interpolate(self.reduce4(feats), size=(H // 32, W // 32), mode='bilinear')

        x = self.up4(x4)
        x = self.up3(x, x3)
        x = self.up2(x, x2)
        x = self.up1(x, x1)
        out = self.head(x)
        # 你这里 head 调用了两次，第二次才插值；第一行是多余的，删掉
        out = F.interpolate(out, scale_factor=2, mode='bilinear')
        return out



@MODELS.register_module()
class DINOv3FPNBackbone(BaseModule):
    """DINOv3 单尺度 backbone -> 多尺度金字塔输出（可接 FPN）
    约定：DINOv3 使用 patch_size=16（*16 系列），输入 H,W 可被 32 整除。
    生成 4 个尺度：stride 4/8/16/32（命名为 C2/C3/C4/C5）。
    """

    def __init__(
        self,
        model_name: str = "dinov3_vit7b16",     # 例如: dinov3_vitl16 / dinov3_vit7b16
        dinov3_local_path: str = "./",
        dinov3_weight_path: str = None,        # 可选，本地权重路径
        out_indices=(0, 1, 2, 3),              # 选择输出哪些尺度
        out_channels: int = 256,               # 每个尺度输出通道
        norm_eval: bool = True,                # 训练时 BN 是否冻结
        init_cfg=None,
        freeze_backbone: bool = True,          # 是否冻结 DINOv3
    ):
        super().__init__(init_cfg=init_cfg)
        self.out_indices = tuple(out_indices)
        self.out_channels = out_channels
        self.norm_eval = norm_eval

        # 1) 加载 DINOv3 backbone（从本地 hub）
        self.dino = torch.hub.load(
            repo_or_dir=dinov3_local_path,
            model=model_name,
            source="local",
            pretrained=False,
            trust_repo=True
        )
        if dinov3_weight_path:
            # 如果权重是纯 state_dict，用 weights_only=True 更安全；不行就去掉该参数
            state = torch.load(dinov3_weight_path, map_location="cpu", weights_only=True)
            self.dino.load_state_dict(state, strict=True)

        # 2) 冻结 DINOv3（默认 True）
        if freeze_backbone:
            for p in self.dino.parameters():
                p.requires_grad = False

        # 3) 读出 embed_dim，作为 patch token 的通道数
        embed_dim = getattr(self.dino, "embed_dim", None)
        if embed_dim is None:
            raise AttributeError("Backbone has no attribute 'embed_dim'. Please check the DINOv3 hub model.")
        self.embed_dim = embed_dim

        # 4) 四个尺度的 1x1 投影（把 embed_dim -> out_channels）
        self.proj_c2 = nn.Conv2d(embed_dim, out_channels * 2**0, 1)
        self.proj_c3 = nn.Conv2d(embed_dim, out_channels * 2**1, 1)
        self.proj_c4 = nn.Conv2d(embed_dim, out_channels * 2**2, 1)
        self.proj_c5 = nn.Conv2d(embed_dim, out_channels * 2**3, 1)

        # 记录特征 stride & channels（部分检测器会读取）
        self._feat_strides = (4, 8, 16, 32)
        self._feat_channels = (256, 512, 1024, 2048)

    @property
    def out_shape(self):
        # 兼容部分框架调用（可选）
        return self._feat_channels

    def train(self, mode=True):
        super().train(mode)
        # 如需 BN 冻结
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()

    def forward(self, x):
        """返回 tuple(C2,C3,C4,C5)，顺序固定，最终由 out_indices 选取"""
        B, C, H, W = x.shape

        # DINOv3 提供 tokens：{'x_norm_patchtokens': [B, N, C]}
        tokens = self.dino.forward_features(x)["x_norm_patchtokens"]

        # N = (H/16)*(W/16)，把 patch tokens reshape 成 feature map
        fmap = tokens.view(B, H // 16, W // 16, self.embed_dim).permute(0, 3, 1, 2)
        # fmap: [B, embed_dim, H/16, W/16]  —— 基础尺度（stride=16）

        # 生成金字塔（注意，这里 C2/C3 是由 fmap 上采样得到）
        # C2: stride=4  -> H/4,  W/4
        c2 = F.interpolate(fmap, size=(H // 4, W // 4), mode="bilinear", align_corners=False)
        c2 = self.proj_c2(c2)

        # C3: stride=8  -> H/8,  W/8
        c3 = F.interpolate(fmap, size=(H // 8, W // 8), mode="bilinear", align_corners=False)
        c3 = self.proj_c3(c3)

        # C4: stride=16 -> H/16, W/16
        c4 = self.proj_c4(fmap)

        # C5: stride=32 -> H/32, W/32（用下采样）
        c5 = F.interpolate(fmap, size=(H // 32, W // 32), mode="bilinear", align_corners=False)
        c5 = self.proj_c5(c5)

        all_feats = (c2, c3, c4, c5)
        outs = tuple(all_feats[i] for i in self.out_indices)
        return outs

    # 供 Neck 等组件查询
    def get_feature_channels(self):
        return [self._feat_channels[i] for i in self.out_indices]

    def get_feature_strides(self):
        return [self._feat_strides[i] for i in self.out_indices]



@MODELS.register_module()
class ConvNeXtBackbone(nn.Module):
    """
    ConvNeXt backbone wrapper returning stage1-4 features,
    with channel dimensions aligned to ResNet-50 style:
        [256, 512, 1024, 2048]
    """
    def __init__(
        self,
        model_id: str = "facebook/dinov3-convnext-tiny-pretrain-lvd1689m",
        num_stages: int = 4,
        trainable: bool = False,
        local_files_only: bool = False,
        cache_dir: str | None = None,
    ):
        super().__init__()
        assert 1 <= num_stages <= 4
        self.num_stages = num_stages

        # 加载 ConvNeXt
        self.backbone = ConvNextModel.from_pretrained(
            model_id,
            local_files_only=local_files_only,
            cache_dir=cache_dir,
        )
        self.backbone.config.output_hidden_states = True

        if not trainable:
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()

        # ConvNeXt Tiny 默认通道数
        convnext_channels = list(self.backbone.config.hidden_sizes)  # [96, 192, 384, 768]
        # ResNet-50 对应通道数
        resnet_channels = [256, 512, 1024, 2048]

        # 构建 1×1 Conv 用于通道对齐
        self.channel_align = nn.ModuleList()
        for in_c, out_c in zip(convnext_channels[:num_stages], resnet_channels[:num_stages]):
            self.channel_align.append(
                nn.Conv2d(in_c, out_c, kernel_size=1, bias=False)
            )

        # 记录输出通道数
        self.out_channels = tuple(resnet_channels[:num_stages])

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        # ConvNeXtModel 的输入参数名是 pixel_values
        outputs = self.backbone(pixel_values=x)
        hiddens = outputs.hidden_states  # (stem, stage1, stage2, stage3, stage4)

        feats = []
        for i in range(self.num_stages):
            stage_feat = hiddens[i + 1]  # 跳过 stem
            stage_feat = self.channel_align[i](stage_feat)
            feats.append(stage_feat)

        return tuple(feats)
