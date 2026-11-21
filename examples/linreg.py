"""Toy task: each worker fits the same linear model on its own synthetic shard.

ferry coordinator --postgres-dsn memory://
ferry aggregator --coordinator localhost:7070 --advertise localhost:7071 --model examples.linreg:make_model
ferry worker --coordinator localhost:7070 --task examples.linreg:make_task --task-arg seed=1 --h 50
ferry worker --coordinator localhost:7070 --task examples.linreg:make_task --task-arg seed=2 --h 120
"""

from __future__ import annotations

import torch

from ferry.worker import TrainTask

IN_DIM = 16
TRUE_W = torch.linspace(-1.0, 1.0, IN_DIM).unsqueeze(0)


def make_model() -> torch.nn.Module:
    torch.manual_seed(0)
    return torch.nn.Linear(IN_DIM, 1)


def make_task(seed: int | str = 0, batch_size: int | str = 32, lr: float | str = 0.05) -> TrainTask:
    seed, batch_size, lr = int(seed), int(batch_size), float(lr)
    gen = torch.Generator().manual_seed(seed)
    model = make_model()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    def train_step() -> tuple[float, int]:
        x = torch.randn(batch_size, IN_DIM, generator=gen)
        y = x @ TRUE_W.T + 0.5 + 0.01 * torch.randn(batch_size, 1, generator=gen)
        loss = torch.nn.functional.mse_loss(model(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        return loss.item(), batch_size

    return TrainTask(model=model, train_step=train_step, optimizer=opt, describe={"task": "linreg"})


def eval_loss(state_dict: dict) -> float:
    model = make_model()
    model.load_state_dict(state_dict)
    gen = torch.Generator().manual_seed(1234)
    x = torch.randn(512, IN_DIM, generator=gen)
    y = x @ TRUE_W.T + 0.5
    with torch.no_grad():
        return torch.nn.functional.mse_loss(model(x), y).item()
