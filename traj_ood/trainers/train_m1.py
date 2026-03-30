"""
训练分类模型（M1）
"""

import torch
import torch.nn.functional as F

def train_m1(model, train_loader, device, epochs=10, lr=1e-3):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()

    for epoch in range(epochs):
        total_loss = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            logits = model(x)

            # 标准分类损失
            loss = F.cross_entropy(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"[Epoch {epoch+1}] Loss: {total_loss:.4f}")

    return model