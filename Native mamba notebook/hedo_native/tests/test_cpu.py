"""CPU unit tests (no GPU, no mamba_ssm). Run: python -m pytest tests -q  (or python tests/test_cpu.py)

A small *causal* stand-in replaces Mamba2 here only; train.py and gate.py refuse
anything but the official mamba_ssm class.
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import models  # noqa: E402
from metrics import retrieval_metrics  # noqa: E402


class CausalStandIn(nn.Module):
    def __init__(self, d_model, d_state, d_conv, expand, headdim):
        super().__init__()
        self.conv = nn.Conv1d(d_model, d_model, d_conv, padding=d_conv - 1)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x):
        L = x.shape[1]
        return self.out(torch.tanh(self.conv(x.transpose(1, 2))[..., :L].transpose(1, 2))) + x


models.MAMBA2_FACTORY = CausalStandIn
torch.manual_seed(0)


def batch(U=8, k=5):
    img = torch.randn(U, 196, 768)
    txt = torch.randn(U * k, 64, 768)
    lengths = torch.randint(5, 64, (U * k,))
    mask = torch.arange(64)[None] < lengths[:, None]
    pair = torch.arange(U).repeat_interleave(k)
    return img, txt, mask, pair


def test_all_variants_forward_backward():
    img, txt, mask, pair = batch()
    for v in models.VARIANTS:
        m = models.RetrievalModel(models.build_config(v))
        out = m(img, txt, mask, pair)
        loss, parts = models.total_loss(m, out, pair, 1e-4)
        loss.backward()
        missing = [n for n, p in m.named_parameters() if p.grad is None]
        assert torch.isfinite(loss) and not missing, (v, missing)
        assert out["z_img"].shape == (8, 128) and out["z_txt"].shape == (40, 128)
        if m.cfg.exchanges:
            assert out["pair_sim"].shape == (8, 40) and "infonce_exchange" in parts


def test_param_matching():
    counts = {v: models.count_trainable(models.RetrievalModel(models.build_config(v))) for v in models.VARIANTS}
    assert counts["full_linear_operator"] == counts["full"]
    assert abs(counts["baseline_param_matched"] - counts["full"]) / counts["full"] < 0.01, counts
    assert abs(counts["transformer_param_matched"] - counts["baseline"]) / counts["baseline"] < 0.01, counts
    assert abs(counts["baseline_param_matched_x"] - counts["full_x"]) / counts["full_x"] < 0.01, counts
    hedo = models.HEDO(128)
    assert sum(p.numel() for p in hedo.parameters()) == sum(p.numel() for p in models.LinearResidualStack(128).parameters())


def test_hedo_affine():
    h = models.HEDO(128).double()
    a, b = torch.randn(2, 3, 128, dtype=torch.float64), torch.randn(2, 3, 128, dtype=torch.float64)
    assert (h(a + b) - h(a) - h(b) + h(torch.zeros_like(a))).abs().max() < 1e-10


def test_boundaries():
    for L, want in {1: [0], 16: [15], 17: [15, 16], 33: [15, 31, 32]}.items():
        y = torch.arange(L, dtype=torch.float32).view(1, L, 1)
        s, c = models.chunk_boundaries(y, torch.ones(1, L, dtype=torch.bool), 16)
        assert s.view(-1)[c.view(-1) > 0].long().tolist() == want


def test_padding_invariance_and_eval_determinism():
    m = models.RetrievalModel(models.build_config("full")).eval()
    _, txt, mask, _ = batch()
    L = int(mask[0].sum())
    a = m.encode_text(txt[:1], mask[:1])
    b = m.encode_text(txt[:1, :L], mask[:1, :L])
    assert torch.allclose(a, b, atol=1e-5)
    assert torch.equal(m.encode_text(txt, mask), m.encode_text(txt, mask))


def test_smooth_bounds():
    h = models.ChunkWiseHVSC(128, 64)
    mu, lv = h.posterior("img", torch.randn(4, 3, 128) * 1e4)
    assert mu.abs().max() <= 10 and lv.min() >= -1.5 and lv.max() <= 1.5
    s = torch.randn(2, 3, 128)
    mu, lv = h.posterior("img", s)
    assert torch.allclose(lv, torch.full_like(lv, -1.0), atol=1e-5)
    kl = h.symmetric_kl(mu, lv, torch.ones(2, 3), mu, lv, torch.ones(2, 3))
    assert abs(float(kl)) < 1e-6


def test_symmetric_kl_matches_textbook_formula():
    torch.manual_seed(3)
    mu_i, mu_t = torch.randn(3, 4, 64, dtype=torch.float64) * 3, torch.randn(3, 4, 64, dtype=torch.float64) * 3
    lv_i, lv_t = torch.rand(3, 4, 64, dtype=torch.float64) * 3 - 1.5, torch.rand(3, 4, 64, dtype=torch.float64) * 3 - 1.5
    m = torch.ones(3, 4, dtype=torch.float64)
    vi, vt, d2 = lv_i.exp(), lv_t.exp(), (mu_i - mu_t) ** 2
    ref = 0.5 * (0.5 * (lv_t - lv_i + (vi + d2) / vt - 1) + 0.5 * (lv_i - lv_t + (vt + d2) / vi - 1)).mean(-1).mean()
    got = models.ChunkWiseHVSC.symmetric_kl(mu_i, lv_i, m, mu_t, lv_t, m)
    assert torch.allclose(got.double(), ref, rtol=1e-5)


def test_infonce_chance_and_alignment():
    _, _, _, pair = batch()
    zi = torch.nn.functional.normalize(torch.randn(8, 128), dim=-1)
    zt = torch.nn.functional.normalize(torch.randn(40, 128), dim=-1)
    chance = 0.5 * (np.log(40) + np.log(8))
    assert abs(float(models.multipositive_infonce(zi, zt, pair, torch.tensor(1e-6))) - chance) < 1e-3
    # 5 equal positives per image: i2t floor is log 5, t2i floor 0 -> 0.5 * log 5
    assert abs(float(models.multipositive_infonce(zi, zi[pair], pair, torch.tensor(100.0))) - 0.5 * np.log(5)) < 1e-2


def test_metrics_and_independent_evaluator():
    rng = np.random.default_rng(0)
    img = rng.normal(size=(1000, 128)).astype(np.float32)
    txt = (np.repeat(img, 5, axis=0) + rng.normal(scale=3.0, size=(5000, 128))).astype(np.float32)
    img /= np.linalg.norm(img, axis=1, keepdims=True)
    txt /= np.linalg.norm(txt, axis=1, keepdims=True)
    m = retrieval_metrics(img, txt)
    assert 0 < m["mean_recall"] < 100
    rand = retrieval_metrics(img, rng.permutation(txt))
    assert rand["mean_recall"] < 2.0

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        data = tmp / "data" / "flickr8k"
        data.mkdir(parents=True)
        ids = [f"img{i:04d}.jpg" for i in range(1000)]
        with open(data / "captions_test.csv", "w", encoding="utf-8") as f:
            f.write("image_id,cap_idx,caption\n")
            for i in ids:
                for k in range(5):
                    f.write(f"{i},{k},a caption\n")
        run = tmp / "run"
        run.mkdir()
        np.save(run / "test_image_embeddings.npy", img)
        np.save(run / "test_text_embeddings.npy", txt)
        json.dump(ids, open(run / "test_image_ids.json", "w"))
        json.dump(m, open(run / "test_results.json", "w"))
        script = Path(__file__).resolve().parents[1] / "evaluate_independent.py"
        ok = subprocess.run([sys.executable, str(script), "--run", str(run), "--data-dir", str(data)],
                            capture_output=True, text=True)
        assert ok.returncode == 0, ok.stdout + ok.stderr
        bad = dict(m, mean_recall=m["mean_recall"] + 0.01)
        json.dump(bad, open(run / "test_results.json", "w"))
        fail = subprocess.run([sys.executable, str(script), "--run", str(run), "--data-dir", str(data)],
                              capture_output=True, text=True)
        assert fail.returncode == 1


def test_energy_theorem_adversarial():
    torch.manual_seed(0)
    x = torch.randn(4, 50, 128, dtype=torch.float64) * 3
    for scale_u, omega, theta, damp in ((1, None, None, None), (30, 5.0, 5.0, -5.0), (3, 2.0, 5.0, -20.0),
                                        (10, 1.0, 0.0, 5.0), (100, 8.0, 10.0, -30.0)):
        op = models.EnergyHEDO(128, 64, steps=10).double()
        with torch.no_grad():
            op.U.mul_(scale_u)
            op.b.normal_()
            op.p_proj.weight.normal_(std=1.0)
            if omega is not None:
                op.omega.fill_(omega)
                op.theta.fill_(theta)
                op.damping.bias.fill_(damp)
        t = op.trajectory(x)
        e = t["energies"]
        assert ((e[..., 1:] - e[..., :-1]) / e[..., :-1].abs().clamp(min=1.0)).max() <= 1e-10
        assert t["attenuation"].max() <= 1.0 and t["attenuation"].min() > 0
        # the bound on the Hessian really holds: compare with the exact Hessian at a random point
        q = torch.randn(128, dtype=torch.float64)
        H = torch.autograd.functional.hessian(lambda v: op.potential(v), q)
        assert torch.linalg.eigvalsh(H).max() <= op.smoothness_bound() * (1 + 1e-9)
        assert torch.allclose(torch.autograd.functional.jacobian(op.potential, q), op.grad_potential(q))


def test_exchange_noop_at_init_and_effect_when_gated():
    m = models.RetrievalModel(models.build_config("full_x")).eval()
    img, txt, mask, _ = batch(4, 1)
    with torch.no_grad():
        zi, ai, _ = m.pass1("img", img, None, sample=False)
        zt, at, _ = m.pass1("txt", txt, mask, sample=False)
        assert torch.allclose(m.pair_similarity(ai, at, mask), (zi * zt).sum(-1), atol=1e-6)
        for k in ("img", "txt"):
            m.xhvsc.gate[k].fill_(1.0)
        s_on = m.pair_similarity(ai, at, mask)
        s_off = m.pair_similarity(ai, at, mask, message_scale=(0.0, 0.0))
        assert torch.allclose(s_off, (zi * zt).sum(-1), atol=1e-6)
        assert (s_on - s_off).abs().max() > 1e-4


def test_rerank_metrics_and_independent_evaluator():
    import metrics
    rng = np.random.default_rng(1)
    img = rng.normal(size=(200, 32)); img /= np.linalg.norm(img, axis=1, keepdims=True)
    txt = np.repeat(img, 5, 0) + rng.normal(scale=2.0, size=(1000, 32)); txt /= np.linalg.norm(txt, axis=1, keepdims=True)
    i2t_idx, t2i_idx = metrics.topk_candidates(img, txt, 8)
    owner = metrics.caption_owner(200)
    # oracle exchange score: positives score highest -> any positive inside the top-K moves to rank 0
    rr = {"i2t_idx": i2t_idx, "i2t_score": (owner[i2t_idx] == np.arange(200)[:, None]) + rng.random(i2t_idx.shape) * 0.1,
          "t2i_idx": t2i_idx, "t2i_score": (t2i_idx == owner[:, None]) + rng.random(t2i_idx.shape) * 0.1}
    m = metrics.retrieval_metrics(img, txt, rr)
    dual_i2t, _ = metrics.retrieval_ranks(img, txt)
    assert m["i2t_r1"] == np.mean(dual_i2t < 8) * 100
    assert m["mean_recall"] >= m["dual_mean_recall"]
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        data = tmp / "data"
        data.mkdir()
        ids = [f"i{i}.jpg" for i in range(200)]
        with open(data / "captions_test.csv", "w", encoding="utf-8") as f:
            f.write("image_id,cap_idx,caption\n" + "".join(f"{i},{k},c\n" for i in ids for k in range(5)))
        run = tmp / "run"
        run.mkdir()
        np.save(run / "test_image_embeddings.npy", img.astype(np.float32))
        np.save(run / "test_text_embeddings.npy", txt.astype(np.float32))
        np.savez(run / "test_rerank.npz", **rr)
        json.dump(ids, open(run / "test_image_ids.json", "w"))
        stored = metrics.retrieval_metrics(img.astype(np.float32), txt.astype(np.float32), rr)
        json.dump(stored, open(run / "test_results.json", "w"))
        script = Path(__file__).resolve().parents[1] / "evaluate_independent.py"
        ok = subprocess.run([sys.executable, str(script), "--run", str(run), "--data-dir", str(data)],
                            capture_output=True, text=True)
        assert ok.returncode == 0, ok.stdout + ok.stderr


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
