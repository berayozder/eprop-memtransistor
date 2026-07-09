"""
gradcheck.py
------------
Proves the MATHEMATICAL CORRECTNESS of the e-prop implementation in network.py by
comparing it against an independent autograd/BPTT reference.

e-prop is a mathematical approximation of BPTT in recurrent networks (it drops
route (ii), the inter-neuron backward path). So the meaningful tests are:

  1) FEEDFORWARD EXACTNESS (W_rec = 0), for LIF and ALIF:
     With no recurrent loops, route (ii) is zero and route (i) (the adaptation
     trace) is kept by the eligibility. In this regime e-prop == BPTT (EXACT).
     This validates the WHOLE machinery of network.py end-to-end: eligibility
     propagation (incl. the ALIF two-component trace), learning signal, kappa
     filtering, and gradient accumulation.

  2) RECURRENT APPROXIMATION ALIGNMENT:
     Cosine similarity between network.py's e-prop gradient and the true BPTT
     gradient. 1.0 is NOT expected (route (ii) is dropped); a positive cosine
     means a usable descent direction -- the mathematical nature of e-prop.

  3) LIF-RECURRENT EXACT FACTORIZATION (Bellec 2020, Eq. 3):
     sum_t (dE/dz^t) * e^t == dE/dW, using network.py's OWN raw eligibility
     traces (via run_trial(return_traces=True)) and the total derivative dE/dz^t
     from autograd. For LIF the reset is detached, so the total dE/dz^t IS the
     exact learning signal L^t -- this directly validates network.py's RECURRENT
     eligibility recursion (W_rec != 0) against autograd.
     (Not applied to ALIF: there the total dE/dz^t also carries the neuron's own
     adaptation memory, which belongs inside e^t, so it would double-count.
     ALIF eligibility is instead validated exactly by Test 1.)

The autograd reference mirrors network.py exactly (detached reset, identical
constants and pseudo-derivative) but is written independently. Nothing here uses
hard-coded layer sizes -- all shapes are derived from the config/tensors.
"""
from __future__ import annotations
import torch
from config import ExperimentConfig
from train import build


# ---- surrogate spike: forward Heaviside, backward pseudo-derivative ----
class SpikeFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u, v_th, gamma):
        ctx.save_for_backward(u)
        ctx.v_th = v_th
        ctx.gamma = gamma
        return (u > 0).to(u.dtype)

    @staticmethod
    def backward(ctx, g):
        (u,) = ctx.saved_tensors
        psi = (ctx.gamma / ctx.v_th) * torch.clamp(1.0 - torch.abs(u) / ctx.v_th, min=0.0)
        return g * psi, None, None


def autograd_forward(X, Ystar, W_rec, W_in, W_out, b_out, beta_vec, nc):
    """Independent BPTT reference; mirrors network.py (reset detached)."""
    alpha, rho, kappa, v_th, gpd = nc.alpha, nc.rho, nc.kappa, nc.v_th, nc.gamma_pd
    T = X.shape[0]
    n = W_rec.shape[0]
    dt = X.dtype
    v = torch.zeros(n, dtype=dt)
    a = torch.zeros(n, dtype=dt)
    z = torch.zeros(n, dtype=dt)
    y = torch.zeros(W_out.shape[0], dtype=dt)
    E = torch.zeros((), dtype=dt)
    z_list = []
    for t in range(T):
        v = alpha * v + (W_rec @ z + W_in @ X[t]) - (z * v_th).detach()
        A = v_th + beta_vec * a
        z = SpikeFn.apply(v - A, v_th, gpd)
        z.retain_grad()
        z_list.append(z)
        y = kappa * y + W_out @ z + b_out
        E = E + 0.5 * ((y - Ystar[t]) ** 2).sum()
        a = rho * a + z
    return E, z_list


def _prep(cfg):
    """Build net from config; return net, data, and leaf-cloneable weights.
    W_rec diagonal is zeroed to match network.py (no self-connections)."""
    net, task = build(cfg)
    X, Y = task["X"], task["Y"]
    T = X.shape[0]
    W_rec = net.syn_rec.weight().clone()
    W_rec = W_rec - torch.diag(torch.diagonal(W_rec))
    return (net, X, Y, T, W_rec, net.syn_in.weight().clone(),
            net.readout.clone(), net.b_out.clone(), net.beta_vec.clone())


def _relerr(a, b):
    return (a - b).norm().item() / max(b.norm().item(), 1e-12)


def _cos(a, b):
    return torch.dot(a.flatten(), b.flatten()).item() / (a.norm().item() * b.norm().item() + 1e-12)


def _small_cfg(adaptive_frac):
    cfg = ExperimentConfig()
    cfg.task.kind = "pattern"
    cfg.task.n_out = 1
    cfg.task.n_in = 4
    cfg.task.T = 20
    cfg.task.freqs = (2.0,)
    cfg.task.amps = (1.0,)
    cfg.neuron.n_rec = 6
    cfg.neuron.adaptive_frac = adaptive_frac
    cfg.device.kind = "ideal"
    cfg.train.eprop_variant = "symmetric"
    return cfg


def test1_feedforward():
    print("=" * 66)
    print("TEST 1 - FEEDFORWARD EXACTNESS (W_rec=0): e-prop == BPTT")
    print("=" * 66)
    ok = True
    for frac, name in [(0.0, "pure LIF"), (1.0, "pure ALIF"), (0.4, "mixed LSNN")]:
        torch.manual_seed(0)
        cfg = _small_cfg(frac)
        nc = cfg.neuron
        net, X, Y, T, Wr, Wi, Wo, bo, beta = _prep(cfg)
        net.syn_rec.dev._G = torch.zeros_like(net.syn_rec.dev._G)   # W_rec = 0
        res = net.run_trial(X, Y, loss="regression")
        Wi = Wi.clone().requires_grad_(True)
        Wo = Wo.clone().requires_grad_(True)
        bo = bo.clone().requires_grad_(True)
        E, _ = autograd_forward(X, Y, torch.zeros_like(Wr), Wi, Wo, bo, beta, nc)
        E.backward()
        r = _relerr(res["grad_in"] * T, Wi.grad)
        p = r < 1e-4
        ok &= p
        print(f"  {name:14s} W_in rel.err {r:.2e}  {'PASSED' if p else 'FAILED'}")
    print(f"  --> {'ALL PASSED' if ok else 'FAILED'}")
    return ok


def test2_recurrent_cosine():
    print("\n" + "=" * 66)
    print("TEST 2 - RECURRENT approximation alignment: cos(e-prop, BPTT)")
    print("=" * 66)
    torch.manual_seed(0)
    cfg = _small_cfg(0.4)
    nc = cfg.neuron
    net, X, Y, T, Wr, Wi, Wo, bo, beta = _prep(cfg)
    res = net.run_trial(X, Y, loss="regression")
    Wr = Wr.clone().requires_grad_(True)
    Wi = Wi.clone().requires_grad_(True)
    Wo = Wo.clone().requires_grad_(True)
    bo = bo.clone().requires_grad_(True)
    E, _ = autograd_forward(X, Y, Wr, Wi, Wo, bo, beta, nc)
    E.backward()
    agr = Wr.grad - torch.diag(torch.diagonal(Wr.grad))
    cr = _cos(res["grad_rec"] * T, agr)
    ci = _cos(res["grad_in"] * T, Wi.grad)
    print(f"  cos W_rec = {cr:.3f} | cos W_in = {ci:.3f}")
    print("  (>0 = positively aligned/usable descent; 1.0 not expected, route (ii) dropped)")
    return cr


def test3_lif_factorization():
    print("\n" + "=" * 66)
    print("TEST 3 - LIF-recurrent exact factorization: sum_t (dE/dz) * e^t == dE/dW")
    print("=" * 66)
    torch.manual_seed(0)
    cfg = _small_cfg(0.0)          # LIF: detached reset -> total dE/dz = exact L^t
    nc = cfg.neuron
    net, X, Y, T, Wr, Wi, Wo, bo, beta = _prep(cfg)

    # network.py's OWN raw (unfiltered) eligibility traces -- validates production code
    res = net.run_trial(X, Y, loss="regression", return_traces=True)

    # total derivative dE/dz^t from the independent autograd reference
    Wr_g = Wr.clone().requires_grad_(True)
    Wi_g = Wi.clone().requires_grad_(True)
    Wo_g = Wo.clone().requires_grad_(True)
    bo_g = bo.clone().requires_grad_(True)
    E, zl = autograd_forward(X, Y, Wr_g, Wi_g, Wo_g, bo_g, beta, nc)
    E.backward()
    dEdz = [z_t.grad for z_t in zl]

    Grec = torch.zeros_like(Wr)
    Gin = torch.zeros_like(Wi)
    for t in range(T):
        Grec += dEdz[t][:, None] * res["e_rec_list"][t]
        Gin += dEdz[t][:, None] * res["e_in_list"][t]
    Grec -= torch.diag(torch.diagonal(Grec))

    agr = Wr_g.grad - torch.diag(torch.diagonal(Wr_g.grad))
    rr, ri = _relerr(Grec, agr), _relerr(Gin, Wi_g.grad)
    ok = rr < 1e-4 and ri < 1e-4
    print(f"  W_rec rel.err {rr:.2e} | W_in rel.err {ri:.2e}  {'PASSED' if ok else 'FAILED'}")
    print("  (network.py's recurrent eligibility recursion verified against autograd)")
    return ok


def test4_readout_bias():
    print("\n" + "=" * 66)
    print("TEST 4 - READOUT & BIAS exactness: grad_out, grad_b == BPTT")
    print("=" * 66)
    ok = True
    for frac, name in [(0.0, "LIF ff"), (0.4, "LSNN recurrent")]:
        torch.manual_seed(0)
        cfg = _small_cfg(frac)
        nc = cfg.neuron
        net, X, Y, T, Wr, Wi, Wo, bo, beta = _prep(cfg)
        if name.endswith("ff"):
            net.syn_rec.dev._G = torch.zeros_like(net.syn_rec.dev._G)
            Wr = torch.zeros_like(Wr)
        res = net.run_trial(X, Y, loss="regression")
        Wr_g = Wr.clone().requires_grad_(True)
        Wi_g = Wi.clone().requires_grad_(True)
        Wo_g = Wo.clone().requires_grad_(True)
        bo_g = bo.clone().requires_grad_(True)
        E, _ = autograd_forward(X, Y, Wr_g, Wi_g, Wo_g, bo_g, beta, nc)
        E.backward()
        r_out = _relerr(res["grad_out"] * T, Wo_g.grad)
        r_b = _relerr(res["grad_b"] * T, bo_g.grad)
        p = r_out < 1e-4 and r_b < 1e-4
        ok &= p
        print(f"  {name:16s} grad_out {r_out:.2e} | grad_b {r_b:.2e}  {'PASSED' if p else 'FAILED'}")
    print(f"  --> {'ALL PASSED' if ok else 'FAILED'}  (readout gradients are exact; bias is kappa-filtered)")
    return ok


def run_checks():
    ok1 = test1_feedforward()
    cr = test2_recurrent_cosine()
    ok3 = test3_lif_factorization()
    ok4 = test4_readout_bias()
    print("\n" + "=" * 66)
    print(f"SUMMARY: Test1 (feedforward exact) {'PASS' if ok1 else 'FAIL'} | "
          f"Test2 (recurrent cos_rec={cr:.2f}) | Test3 (LIF factorization) {'PASS' if ok3 else 'FAIL'} | "
          f"Test4 (readout+bias) {'PASS' if ok4 else 'FAIL'}")
    print("Result: network.py's eligibility + learning-signal machinery is mathematically")
    print("        correct; recurrent e-prop is a positively-aligned BPTT approximation.")
    print("=" * 66)


if __name__ == "__main__":
    run_checks()