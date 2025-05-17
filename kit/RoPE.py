import torch

def apply_rope(x, theta=10000):
    # x 的形状: (num_tokens, dim)
    num_tokens, d = x.shape
    # 生成位置索引 t: [0, 1, 2, ...]
    t = torch.arange(num_tokens, device=x.device).float()
    # 计算频率向量，长度为 d // 2
    freqs = 1.0 / (theta ** (torch.arange(0, d, 2, device=x.device) / d))
    # 计算旋转角度，形状: (num_tokens, d // 2)
    angles = t[:, None] * freqs[None, :]
    cos = torch.cos(angles)  # 形状: (num_tokens, d // 2)
    sin = torch.sin(angles)  # 形状: (num_tokens, d // 2)
    # 将 x 按两两一组分割，形状: (num_tokens, d // 2, 2)
    x_pairs = x.view(num_tokens, -1, 2)
    # 应用旋转变换
    x0 = x_pairs[..., 0] * cos - x_pairs[..., 1] * sin
    x1 = x_pairs[..., 0] * sin + x_pairs[..., 1] * cos
    # 合并并恢复原始形状
    return torch.cat([x0, x1], dim=-1).view_as(x)

# 测试输入
x = torch.tensor([[1.0, 2.0, 3.0, 4.0],
                  [0.5, 1.5, 2.5, 3.5],
                  [0.0, 1.0, 2.0, 3.0]])
output = apply_rope(x)
print(output)