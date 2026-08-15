"""视觉→神经元: MNIST 手写数字训练, 验证神经元分化与可解码性。

大脑对应: 视觉皮层 (V1 边缘) → 字形 → 数字语义 (IPS)
数据: 28×28 像素 → 视觉编码器 → SparseUnit → 数字分类

验证:
  1. 神经元是否按数字分化 (0-9 各有神经元子集)
  2. 神经元激活可否解码数字 (线性探针)
  3. 视觉→语义通路: 数字识别正确率
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))

from units.sparse_unit import SparseUnit
from core.plasticity import PlasticityMechanism, PlasticityPipeline
import units.plasticity_impl  # noqa: 注册机制
import units.neurogenesis  # noqa: 注册神经生长


class VisionSparseModel(nn.Module):
    """视觉编码 + SparseUnit 神经元池 + 分类头。"""

    def __init__(self, d_model: int = 128, d_pool: int = 1024, top_k: int = 64,
                 n_classes: int = 10, mechanisms: str = ""):
        super().__init__()
        # 视觉编码: 任意尺寸 → CNN → 固定特征
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((4, 4)),  # 固定输出 4×4
            nn.Flatten(),
            nn.Linear(32 * 4 * 4, d_model),
            nn.LayerNorm(d_model),
        )
        # 神经元池 (稀疏单元, 无 block 结构 — 直接一层)
        self.unit = SparseUnit(d_model=d_model, d_pool=d_pool, top_k=top_k)
        # 分类头: 神经元输出 → 数字
        self.head = nn.Linear(d_model, n_classes)

        # 挂载可塑性机制
        if mechanisms:
            mechs = []
            for n in mechanisms.split(","):
                cls = None
                for reg_name, reg_cls in PlasticityMechanism.registry.items():
                    if reg_name == n:
                        cls = reg_cls
                        break
                if cls:
                    mechs.append(cls())
            if mechs:
                self.unit.set_plasticity(PlasticityPipeline(mechs))

    def set_mechanism_params(self, name: str, **kwargs) -> None:
        """训练前调机制参数 (如 neurogenesis 的 grow_after)。"""
        if getattr(self.unit, "plasticity", None) is None:
            return
        for mech in self.unit.plasticity.mechanisms:
            if mech.name == name:
                for k, v in kwargs.items():
                    if hasattr(mech, k):
                        setattr(mech, k, v)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """x: (B, 1, 28, 28) → (logits, stats)"""
        feat = self.encoder(x)          # (B, d_model)
        # SparseUnit 期望 (B, S, d) — 无序列, S=1
        feat = feat.unsqueeze(1)        # (B, 1, d)
        out, stats = self.unit(feat)    # (B, 1, d)
        out = out.squeeze(1)            # (B, d)
        logits = self.head(out)
        return logits, stats


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--d_pool", type=int, default=1024)
    p.add_argument("--top_k", type=int, default=64)
    p.add_argument("--mechanisms", type=str, default="")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    device = torch.device(args.device)
    # 直接加载 npy (绕过 torchvision 下载)
    import numpy as np
    img = np.load("data/mnist/train_img.npy")
    lab = np.load("data/mnist/train_lab.npy")
    t_img = np.load("data/mnist/test_img.npy")
    t_lab = np.load("data/mnist/test_lab.npy")
    train_x = torch.from_numpy(img).float().unsqueeze(1) / 255.0
    train_y = torch.from_numpy(lab).long()
    test_x = torch.from_numpy(t_img).float().unsqueeze(1) / 255.0
    test_y = torch.from_numpy(t_lab).long()
    train_ds = torch.utils.data.TensorDataset(train_x, train_y)
    test_ds = torch.utils.data.TensorDataset(test_x, test_y)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=args.batch_size)

    model = VisionSparseModel(d_pool=args.d_pool, top_k=args.top_k,
                              mechanisms=args.mechanisms).to(device)
    # 调低生长触发 (小任务, 步数少)
    if "neurogenesis" in (args.mechanisms or ""):
        model.set_mechanism_params("neurogenesis", grow_after=150,
                                   grow_interval=50, load_mult=1.2)
    print(f"模型: 视觉CNN + SparseUnit({args.d_pool}池, top-{args.top_k}) "
          f"机制:{args.mechanisms or 'B0'}")
    print(f"参数量: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    for epoch in range(args.epochs):
        model.train()
        total, correct = 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits, _ = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            opt.step()
            correct += (logits.argmax(-1) == y).sum().item()
            total += y.numel()
        # 测试
        model.eval()
        t_correct, t_total = 0, 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                logits, _ = model(x)
                t_correct += (logits.argmax(-1) == y).sum().item()
                t_total += y.numel()
        print(f"  epoch {epoch+1}: train acc={correct/total:.3f} "
              f"test acc={t_correct/t_total:.3f}")

    # 保存
    out = Path(f"runs/mnist_visual_{args.mechanisms or 'B0'}_seed42.pt")
    torch.save({"model": model.state_dict(), "config": vars(args)}, out)
    print(f"保存: {out}")


if __name__ == "__main__":
    main()
