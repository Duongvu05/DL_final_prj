import torch
from torch import nn

from dl_final_prj.engine import FitConfig, fit


def _setup():
    x = torch.randn(64, 4, generator=torch.Generator().manual_seed(0))
    y = x.sum(1, keepdim=True)

    def batches(epoch, g):
        perm = torch.randperm(64, generator=g)
        for i in range(0, 64, 16):
            yield x[perm[i : i + 16]], y[perm[i : i + 16]]

    def step(m, b, epoch):
        return nn.functional.mse_loss(m(b[0]), b[1])

    def val(m):
        return float(nn.functional.mse_loss(m(x), y))

    return batches, step, val


def _model():
    torch.manual_seed(123)
    return nn.Sequential(nn.Linear(4, 8), nn.Dropout(0.3), nn.Linear(8, 1))


def test_resume_matches_uninterrupted_run(tmp_path):
    batches, step, val = _setup()
    cfg = FitConfig(epochs=4, patience=10, lr=1e-2)
    m1 = _model()
    full, h1 = fit(
        tmp_path / "full", m1, [{"params": m1.parameters()}], batches, step, val, cfg, seed=0
    )

    m2 = _model()
    fit(
        tmp_path / "part",
        m2,
        [{"params": m2.parameters()}],
        batches,
        step,
        val,
        FitConfig(epochs=2, patience=10, lr=1e-2),
        seed=0,
    )
    # simulate preemption after epoch 2: the checkpoint says "not stopped" and training continues
    ck = torch.load(tmp_path / "part" / "ckpt.pt", weights_only=False)
    ck["stopped"] = False
    torch.save(ck, tmp_path / "part" / "ckpt.pt")
    m3 = _model()
    resumed, h3 = fit(
        tmp_path / "part", m3, [{"params": m3.parameters()}], batches, step, val, cfg, seed=0
    )

    assert [round(h["val"], 10) for h in h1] == [round(h["val"], 10) for h in h3]
    for a, b in zip(full.state_dict().values(), resumed.state_dict().values()):
        torch.testing.assert_close(a, b)


def test_step_schedule_drops_lr_and_resumes(tmp_path):
    batches, step, val = _setup()
    cfg = FitConfig(epochs=4, patience=10, lr=1e-2, schedule="step", step_epoch=2, step_gamma=0.1)
    m = _model()
    _, h = fit(tmp_path / "s", m, [{"params": m.parameters()}], batches, step, val, cfg, seed=0)
    lrs = [round(r["lr"], 6) for r in h]
    assert lrs == [0.01, 0.001, 0.001, 0.0001]


def test_prepare_run_dir_blocks_stale_checkpoints(tmp_path):
    import pytest

    from dl_final_prj.engine import prepare_run_dir

    run = tmp_path / "run"
    prepare_run_dir(run, {"lr": 1e-3, "data": {"size": 1}}, force=False)
    (run / "ckpt.pt").write_text("x")
    prepare_run_dir(run, {"lr": 1e-3, "data": {"size": 1}}, force=False)  # same settings resume
    with pytest.raises(ValueError):
        prepare_run_dir(run, {"lr": 5e-4, "data": {"size": 1}}, force=False)
    prepare_run_dir(run, {"lr": 5e-4, "data": {"size": 1}}, force=True)  # force wipes the run
    assert not (run / "ckpt.pt").exists()
    other = tmp_path / "legacy"
    other.mkdir()
    (other / "ckpt.pt").write_text("x")
    with pytest.raises(ValueError):
        prepare_run_dir(other, {"lr": 1e-3}, force=False)
