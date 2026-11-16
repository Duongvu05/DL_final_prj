import copy
from collections.abc import Callable

import torch
from loguru import logger
from torch import nn


def fit(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    step_loss: Callable[[nn.Module, tuple, int], torch.Tensor],
    val_metric: Callable[[nn.Module], float],
    *,
    epochs: int = 50,
    lr: float = 1e-3,
    clip: float = 1.0,
    patience: int = 8,
    device: str = "cpu",
) -> nn.Module:
    """Generic loop. val_metric returns a number where lower is better (for example RMSE)."""
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    best, best_state, bad = float("inf"), copy.deepcopy(model.state_dict()), 0
    for epoch in range(1, epochs + 1):
        model.train()
        total, seen = 0.0, 0
        for batch in train_loader:
            batch = tuple(b.to(device) for b in batch)
            opt.zero_grad()
            loss = step_loss(model, batch, epoch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()
            total += loss.item() * len(batch[0])
            seen += len(batch[0])
        model.eval()
        with torch.no_grad():
            val = val_metric(model)
        logger.info(f"epoch {epoch} train_loss {total / max(1, seen):.5f} val {val:.5f}")
        if val < best:
            best, best_state, bad = val, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                logger.info("early stopping")
                break
    model.load_state_dict(best_state)
    return model
