"""Pre-training gate on real cached Flickr8k features. Must pass before any run.

 1. official mamba_ssm Mamba2 is the class used by every Mamba-2 variant
 2. chunk-boundary indices for lengths 1..33 and padded batches
 3. U x N multi-positive InfoNCE geometry (8 images x 40 captions)
 4. every variant: real-batch forward/backward finite, every trainable parameter receives a gradient
 5. Mamba-2 fused (mem-efficient) path vs reference path on real inputs, forward and gradient
 6. padding invariance of the text branch (full 64-token padding vs truncated to caption length)
 7. legacy HEDO is affine (why it was replaced)
 8. Theorem 1: EnergyHEDO energy never increases (float64, trained-scale and adversarial parameters, real tokens)
 9. exchange HVSC: exact no-op at initialisation (score == pass-1 cosine); depends on the partner once gated
10. parameter counts for all variants; matched controls within 1%

Writes REPORT_DIR/gate_report.json and param_counts.json. Exit 1 on any failure.
"""

import sys

import numpy as np
import torch

from common import FEATURE_DIR, REPORT_DIR, banner, environment_manifest, provenance, set_seed, write_json
from models import (HEDO, PROPOSED, VARIANTS, EnergyHEDO, RetrievalModel, assert_native_mamba2, build_config,
                    chunk_boundaries, count_trainable, multipositive_infonce, total_loss)

report = {}


def check(name, ok, detail=None):
    report[name] = {"pass": bool(ok), "detail": detail}
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f": {detail}" if detail is not None else ""), flush=True)


def main():
    device = torch.device("cuda")
    set_seed(0)
    banner("PRE-TRAINING GATE")

    from mamba_ssm import Mamba2
    check("mamba2_module", Mamba2.__module__ == "mamba_ssm.modules.mamba2", Mamba2.__module__)

    expected = {1: [0], 7: [6], 8: [7], 9: [8], 15: [14], 16: [15], 17: [15, 16], 31: [15, 30], 32: [15, 31],
                33: [15, 31, 32]}
    ok, detail = True, {}
    for L, want in expected.items():
        y = torch.arange(L, dtype=torch.float32).view(1, L, 1)
        states, cmask = chunk_boundaries(y, torch.ones(1, L, dtype=torch.bool), 16)
        got = states.view(-1)[cmask.view(-1) > 0].long().tolist()
        detail[L] = got
        ok &= got == want
    y = torch.arange(40, dtype=torch.float32).view(1, 40, 1).repeat(2, 1, 1)
    m = torch.zeros(2, 40, dtype=torch.bool)
    m[0, :40], m[1, :17] = True, True
    states, cmask = chunk_boundaries(y, m, 16)
    ok &= cmask.tolist() == [[1, 1, 1], [1, 1, 0]] and states[1, :2].view(-1).tolist() == [15, 16]
    check("chunk_boundaries", ok, detail)

    img = torch.from_numpy(np.load(FEATURE_DIR / "img_train.npy", mmap_mode="r")[:8].astype(np.float32)).to(device)
    txt = torch.from_numpy(np.load(FEATURE_DIR / "txt_train.npy", mmap_mode="r")[:40].astype(np.float32)).to(device)
    mask = torch.from_numpy(np.load(FEATURE_DIR / "mask_train.npy")[:40].astype(bool)).to(device)
    pair = torch.arange(8, device=device).repeat_interleave(5)
    check("real_features_finite", bool(torch.isfinite(img).all() and torch.isfinite(txt).all()),
          {"img": list(img.shape), "txt": list(txt.shape), "mean_caption_len": float(mask.sum(1).float().mean())})

    zi = torch.nn.functional.normalize(torch.randn(8, 128, device=device), dim=-1)
    zt = torch.nn.functional.normalize(torch.randn(40, 128, device=device), dim=-1)
    loss_rand = float(multipositive_infonce(zi, zt, pair, torch.tensor(1.0, device=device)))
    zt_perfect = zi[pair]
    loss_perfect = float(multipositive_infonce(zi, zt_perfect, pair, torch.tensor(100.0, device=device)))
    floor = 0.5 * float(np.log(5))  # 5 equal positives per image: i2t floor log 5, t2i floor 0
    check("infonce_geometry", abs(loss_perfect - floor) < 1e-2 and loss_rand > floor + 0.5,
          {"random_scale1": loss_rand, "aligned_scale100": loss_perfect, "expected_floor": floor})

    params = {}
    for variant in VARIANTS:
        model = RetrievalModel(build_config(variant)).to(device)
        n_native = assert_native_mamba2(model)
        params[variant] = count_trainable(model)
        out = model(img, txt, mask, pair, sample=True)
        loss, parts = total_loss(model, out, pair, 1e-4)
        loss.backward()
        no_grad = [n for n, p in model.named_parameters() if p.requires_grad and (p.grad is None)]
        bad = [n for n, p in model.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()]
        check(f"variant_{variant}", torch.isfinite(loss) and not no_grad and not bad,
              {"loss": float(loss), **{k: float(v) for k, v in parts.items()}, "native_mamba2_modules": n_native,
               "trainable_params": params[variant], "params_without_grad": no_grad, "nonfinite_grads": bad})

    torch.manual_seed(0)
    fused = Mamba2(d_model=128, d_state=64, d_conv=4, expand=2, headdim=64, use_mem_eff_path=True).to(device)
    ref = Mamba2(d_model=128, d_state=64, d_conv=4, expand=2, headdim=64, use_mem_eff_path=False).to(device)
    ref.load_state_dict(fused.state_dict())
    proj = torch.nn.Linear(768, 128).to(device)
    x = proj(img).detach()
    x1, x2 = x.clone().requires_grad_(True), x.clone().requires_grad_(True)
    y1, y2 = fused(x1), ref(x2)
    (y1.square().mean()).backward()
    (y2.square().mean()).backward()
    fwd = float((y1 - y2).abs().max() / y2.abs().max().clamp(min=1e-12))
    bwd = float((x1.grad - x2.grad).abs().max() / x2.grad.abs().max().clamp(min=1e-12))
    check("mamba2_fused_vs_reference", fwd < 1e-3 and bwd < 1e-3 and torch.isfinite(y1).all(),
          {"rel_forward_diff": fwd, "rel_input_grad_diff": bwd})

    model = RetrievalModel(build_config("full")).to(device).eval()
    lengths = mask.sum(1)
    worst = 1.0
    for i in range(0, 40, 5):
        L = int(lengths[i])
        a = model.encode_text(txt[i:i + 1], mask[i:i + 1])
        b = model.encode_text(txt[i:i + 1, :L], mask[i:i + 1, :L])
        worst = min(worst, float((a * b).sum()))
    check("padding_invariance_text", worst > 0.9999, {"min_cosine": worst})

    hedo = HEDO(128).to(device).double()
    a, b = torch.randn(3, 5, 128, device=device, dtype=torch.float64), torch.randn(3, 5, 128, device=device,
                                                                                   dtype=torch.float64)
    zero = torch.zeros_like(a)
    residual = float((hedo(a + b) - hedo(a) - hedo(b) + hedo(zero)).abs().max())
    check("legacy_hedo_is_affine", residual < 1e-9, {"additivity_residual": residual})

    tokens = RetrievalModel(build_config(PROPOSED)).to(device).proj["img"](img).detach().double()
    worst, detail = -float("inf"), {}
    for name, scale_u, omega, theta, damp in (("init", 1.0, None, None, None), ("stiff", 30.0, 5.0, 5.0, -5.0),
                                              ("undamped", 3.0, 2.0, 5.0, -20.0), ("heavy", 10.0, 1.0, 0.0, 5.0)):
        torch.manual_seed(1)
        op = EnergyHEDO(128, 64, steps=8).to(device).double()
        with torch.no_grad():
            op.U.mul_(scale_u)
            op.b.normal_()
            op.p_proj.weight.normal_(std=0.5)
            if omega is not None:
                op.omega.fill_(omega)
                op.theta.fill_(theta)
                op.damping.bias.fill_(damp)
        e = op.trajectory(tokens)["energies"]
        inc = float(((e[..., 1:] - e[..., :-1]) / e[..., :-1].abs().clamp(min=1.0)).max())
        detail[name] = {"max_relative_step_increase": inc, "mean_dissipated": float((e[..., 0] - e[..., -1]).mean())}
        worst = max(worst, inc)
    check("energy_theorem_nonincreasing", worst <= 1e-10, detail)

    xm = RetrievalModel(build_config("hvsc_x")).to(device).eval()
    with torch.no_grad():
        zi, ai, _ = xm.pass1("img", img[:4], None, sample=False)
        zt, at, _ = xm.pass1("txt", txt[:4], mask[:4], sample=False)
        s_init = xm.pair_similarity(ai, at, mask[:4])
        noop = float((s_init - (zi * zt).sum(-1)).abs().max())
        for m in ("img", "txt"):
            xm.xhvsc.gate[m].fill_(1.0)
        s_a = xm.pair_similarity(ai, at, mask[:4])
        perm = torch.tensor([1, 2, 3, 0], device=device)
        s_b = xm.pair_similarity(ai, {k: v[perm] for k, v in at.items()}, mask[:4][perm])
        zt_b = zt[perm]
        partner_effect = float(((s_b - s_a) - ((zi * zt_b).sum(-1) - (zi * zt).sum(-1))).abs().max())
    check("exchange_noop_at_init_and_active_when_gated", noop < 1e-5 and partner_effect > 1e-4,
          {"init_abs_diff": noop, "gated_partner_effect": partner_effect})

    check("linear_operator_param_match", params["full_linear_operator"] == params["full"],
          {k: params[k] for k in ("full", "full_linear_operator")})
    rel = {"baseline_param_matched_vs_full": abs(params["baseline_param_matched"] - params["full"]) / params["full"],
           "baseline_param_matched_x_vs_full_x":
               abs(params["baseline_param_matched_x"] - params["full_x"]) / params["full_x"],
           "transformer_vs_baseline":
               abs(params["transformer_param_matched"] - params["baseline"]) / params["baseline"]}
    check("param_matched_controls", max(rel.values()) < 0.01, rel)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    write_json(REPORT_DIR / "param_counts.json", params)
    passed = all(v["pass"] for v in report.values())
    write_json(REPORT_DIR / "gate_report.json", {"pass": passed, "checks": report,
                                                 "environment": environment_manifest(), "provenance": provenance()})
    print("GATE:", "PASS" if passed else "FAIL")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
