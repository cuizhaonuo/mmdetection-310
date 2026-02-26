from __future__ import annotations
from typing import Dict, Any, Tuple, Optional
import torch
import torch.nn as nn
from mmdet.registry import MODELS


# =========================
# 统一输出的 Adapter
# =========================
class OutputAdapter(nn.Module):
    """统一 forward 输出，方便测试/下游复用。

    kind:
      - "single":  单尺度输出 -> {"feat": Tensor}
      - "pyramid": 多尺度输出 -> {"pyramid": Tuple[Tensor, ...]}
    """
    def __init__(self, model: nn.Module, kind: str = "single"):
        super().__init__()
        assert kind in ("single", "pyramid"), f"Unsupported kind: {kind}"
        self.model = model
        self.kind = kind

    def forward(self, x: torch.Tensor) -> Dict[str, Any]:
        out = self.model(x)

        if self.kind == "single":
            # 保险起见：若底层误返回了 tuple/list，则取第一个
            if isinstance(out, (tuple, list)):
                out = out[0]
            return {"feat": out}

        # self.kind == "pyramid"
        if not isinstance(out, (tuple, list)):
            out = (out,)
        return {"pyramid": tuple(out)}


# =========================
# 具体模型的 Adapter（注册到 MODELS）
# =========================
@MODELS.register_module()
class DINOv3UNetAdapter(OutputAdapter):
    """把 DINOv3_UNet 包装成统一输出 -> 单尺度 {"feat": Tensor}"""
    def __init__(self, model_cfg: Dict[str, Any]):
        # model_cfg 例如: {'type': 'DINOv3_UNet', ...}
        inner = MODELS.build(model_cfg)
        super().__init__(inner, kind="single")


@MODELS.register_module()
class DINOv3FPNAdapter(OutputAdapter):
    """把 DINOv3FPNBackbone 包装成统一输出 -> 金字塔 {"pyramid": (P2, P3, P4, P5, ...)}"""
    def __init__(self, model_cfg: Dict[str, Any]):
        inner = MODELS.build(model_cfg)
        super().__init__(inner, kind="pyramid")


@MODELS.register_module()
class ConvNeXtBackboneAdapter(OutputAdapter):
    """把 ConvNeXtBackbone 包装成统一输出（默认按单尺度统一为 'feat'）

    说明：如果你希望把 ConvNeXt 的多 stage 也当作金字塔用，
         可将 super().__init__(inner, kind="single") 改为 "pyramid"，
         前提是底层 forward 返回的是 tuple/list。
    """
    def __init__(self, model_cfg: Dict[str, Any]):
        inner = MODELS.build(model_cfg)
        super().__init__(inner, kind="single")


# =========================
# 预设（可按需修改路径/参数）
# =========================
PRESETS: Dict[str, Dict[str, Any]] = {
    "dinov3_unet": dict(
        type="DINOv3UNetAdapter",
        model_cfg=dict(
            type="DINOv3_UNet",
            dinov3_local_path="/public/home/zncui/pycharm_project/dinov3",
            dinov3_weight_path=None,
        ),
    ),
    "dinov3_fpn": dict(
        type="DINOv3FPNAdapter",
        model_cfg=dict(
            type="DINOv3FPNBackbone",
            model_name="dinov3_vit7b16",
            dinov3_local_path="/public/home/zncui/pycharm_project/dinov3",
            dinov3_weight_path=None,
            freeze_backbone=True,
            out_indices=(0, 1, 2, 3),
            out_channels=256,
        ),
    ),
    "convnext": dict(
        type="ConvNeXtBackboneAdapter",
        model_cfg=dict(
            type="ConvNeXtBackbone",
            model_id="facebook/dinov3-convnext-tiny-pretrain-lvd1689m",
            num_stages=4,
            trainable=False,
            local_files_only=False,
            cache_dir=None,
        ),
    ),
}


# =========================
# 构建与测试
# =========================
def build_model(choice: str) -> nn.Module:
    """根据 PRESETS 里给定的 choice 构建对应 *Adapter*（内部再构建真实模型）"""
    assert choice in PRESETS, f"Unknown choice: {choice}. Valid: {list(PRESETS)}"
    cfg = PRESETS[choice]
    model = MODELS.build(cfg)
    return model.eval()


def _print_out_shape(out: Dict[str, Any]) -> None:
    """统一打印输出形状，便于快速核对"""
    if "feat" in out and isinstance(out["feat"], torch.Tensor):
        t = out["feat"]
        print("feat:", tuple(t.shape))
    if "pyramid" in out and isinstance(out["pyramid"], (tuple, list)):
        shapes = [tuple(t.shape) for t in out["pyramid"]]
        print("pyramid:", shapes)


if __name__ == "__main__":
    # ===== 仅需改这里来切换不同模型 =====
    CHOICE = "dinov3_fpn"     # 可选: "dinov3_unet" | "dinov3_fpn" | "convnext"
    INPUT_HW = (448, 448)     # H, W
    # =================================

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(CHOICE).to(device)

    x = torch.randn(1, 3, *INPUT_HW, device=device)
    with torch.no_grad():
        out = model(x)

    _print_out_shape(out)
