# Differential Pair (`G+ - G-`) Synapse Implementation — Changelog & Architecture

This document lists all modifications made across the codebase to implement **Differential Pair Synapses** with LTP-only programming and automatic hardware saturation refresh.

---

## 1. `config.py`
Added two configuration flags to `SynapseConfig` to enable differential pair mapping and automatic hardware refresh.

```diff
 @dataclass
 class SynapseConfig:
     w_range: float = 0.5          # Synaptic weight scale, mapped to conductance G [g_min, g_max]
     writer: str = "accumulate"    # Programming scheme: "direct" | "accumulate" | "verify"
     verify_max_iter: int = 5      # Maximum loop iterations for write-verify scheme
     weights_on_device: tuple = ("rec", "in")  # Synaptic weights stored on memristive hardware
+    differential_pair: bool = False  # True -> differential pair (G+ - G-) with LTP-only pulses
+    auto_refresh: bool = True        # True -> refresh saturated differential pairs to g_lo
```

---

## 2. `synapse.py`
Updated `Synapse` to manage dual-device populations (`dev_pos` and `dev_neg`), calculate differential weights ($W = \gamma(G^+ - G^-)$), route pulses to LTP-only branches, and perform saturation refresh.

### A. Initialization (`__init__`)
Supports receiving either a single device or a tuple `(dev_pos, dev_neg)`:
```diff
 class Synapse:
-    def __init__(self, device, w_range, writer="accumulate", verify_max_iter=5):
-        self.dev = device
-        g_lo, g_hi = device.bounds
+    def __init__(self, device, w_range, writer="accumulate", verify_max_iter=5,
+                 differential_pair=False, auto_refresh=True):
+        if isinstance(device, tuple):
+            self.dev_pos, self.dev_neg = device
+            self.dev = self.dev_pos
+            self.differential_pair = True
+        else:
+            self.dev_pos = device
+            self.dev_neg = None
+            self.dev = device
+            self.differential_pair = differential_pair and (self.dev_neg is not None)
+
+        self.auto_refresh = auto_refresh
+        g_lo, g_hi = self.dev_pos.bounds
         # Ideal devices have infinite boundaries -> direct mapping (W = G)
         if not (g_lo > -1e30 and g_hi < 1e30):
             self.gamma = 1.0
             self.g_ref = 0.0
             self.ideal_map = True
         else:
-            self.g_ref = 0.5 * (g_lo + g_hi)
-            self.gamma = w_range / max(0.5 * (g_hi - g_lo), 1e-8)
+            self.g_ref = g_lo if self.differential_pair else 0.5 * (g_lo + g_hi)
+            self.gamma = w_range / max((g_hi - g_lo) if self.differential_pair else 0.5 * (g_hi - g_lo), 1e-8)
             self.ideal_map = False
```

### B. Weight Readout (`weight`) & Initialization (`init_weight`)
```diff
     def weight(self):
+        if self.differential_pair:
+            return self.gamma * (self.dev_pos.read() - self.dev_neg.read())
         return self.gamma * (self.dev.read() - self.g_ref)

     def init_weight(self, W):
         """Maps initial weights to physical conductance states."""
+        if self.differential_pair:
+            G_pos = torch.where(W >= 0, self.g_ref + W / self.gamma, self.g_ref)
+            G_neg = torch.where(W < 0, self.g_ref - W / self.gamma, self.g_ref)
+            if hasattr(self.dev_pos, "set_state"):
+                self.dev_pos.set_state(G_pos)
+                self.dev_neg.set_state(G_neg)
+            else:
+                self.dev_pos._G = G_pos
+                self.dev_neg._G = G_neg
+            return
```

### C. Saturation Refresh Helper (`refresh_saturated`)
Added method to prevent both branches from drifting together toward $G_{max}$, while **counting physical RESET + reprogramming pulses** into `self.n_pulses_total` for fair energy comparisons against single devices:
```python
    def refresh_saturated(self, threshold_ratio=0.92):
        """
        Hardware refresh cycle for differential pairs approaching saturation:
        Resets saturated synapses down to g_lo while preserving effective weight W.
        Also accounts for physical RESET + reprogramming pulses in n_pulses_total.
        """
        if not self.differential_pair:
            return
        g_lo, g_hi = self.dev_pos.bounds
        sat_th = g_lo + threshold_ratio * (g_hi - g_lo)
        sat = torch.minimum(self.dev_pos._G, self.dev_neg._G) >= sat_th   # BOTH devices high
        if not sat.any():
            return
        W_curr = self.weight()
        g_diff = W_curr / self.gamma
        G_pos_new = torch.where(sat, torch.where(g_diff >= 0, g_lo + g_diff, g_lo), self.dev_pos._G)
        G_neg_new = torch.where(sat, torch.where(g_diff < 0, g_lo - g_diff, g_lo), self.dev_neg._G)

        # Account for physical hardware refresh pulse expenditure (RESET pair + LTP reprogram)
        n_sat = int(sat.sum().item())
        step = self.dev_pos.nominal_ltp_step if hasattr(self.dev_pos, "nominal_ltp_step") else self.dev_pos.nominal_step
        reprogram_pulses = int(((G_pos_new[sat] - g_lo).sum() + (G_neg_new[sat] - g_lo).sum()).item() / step)
        self.n_pulses_total += (2 * n_sat + reprogram_pulses)

        self.dev_pos._G = G_pos_new.clamp(g_lo, g_hi)
        self.dev_neg._G = G_neg_new.clamp(g_lo, g_hi)
```

**Bug-fix note (post-initial-implementation):** the refresh trigger originally used
`sat = (dev_pos._G >= sat_th) | (dev_neg._G >= sat_th)` (OR). This is wrong: a
legitimately large weight keeps exactly ONE device near `g_hi` (one-sided
saturation), which is not a removable common-mode offset — but the OR trigger
fired on it anyway, causing refresh to churn every trial and waste pulses. It was
replaced with the current `torch.minimum(dev_pos._G, dev_neg._G) >= sat_th` (AND
via `minimum`), which only fires when BOTH devices are high — the actual
removable-offset case. See `synapse.py`'s `refresh_saturated` docstring for the
current rationale.

### D. LTP-Only Pulse Routing (`_emit`)
Routes positive $\Delta W$ pulses to $G^+$ and negative $\Delta W$ pulses to $G^-$ using **LTP (+1) pulses exclusively**:
```diff
     def _emit(self, dW_pulses, max_pulses=30):
         """Applies signed pulse counts to the physical device population."""
         dW_pulses = torch.clamp(dW_pulses, -max_pulses, max_pulses)
+        if self.differential_pair:
+            # LTP-only programming: positive pulses to dev_pos, negative (absolute) to dev_neg
+            n_pos = torch.relu(dW_pulses)
+            n_neg = torch.relu(-dW_pulses)
+            self.dev_pos.pulse(torch.ones_like(n_pos), n_pos)
+            self.dev_neg.pulse(torch.ones_like(n_neg), n_neg)
+            self.n_pulses_total += int(n_pos.sum().item() + n_neg.sum().item())
+            if self.auto_refresh:
+                self.refresh_saturated()
+            return
+
         polarity = torch.sign(dW_pulses)
         n = dW_pulses.abs()
         self.dev.pulse(polarity, n)
         self.n_pulses_total += int(n.sum().item())
```

---

## 3. `memtransistor.py`
Added the `nominal_ltp_step` property so that differential pair synapses compute `w_step` based purely on the LTP curve rather than averaging LTP and LTD:

```python
    @property
    def nominal_ltp_step(self):
        """Pure LTP nominal step size at window midpoint, used by Differential Pair Synapses."""
        c = self.cfg
        if c.enable_nonlinearity:
            ltp = c.dp * math.exp(-0.5 * c.kp)
        else:
            ltp = c.dp
        return ltp * c.step_scale
```

---

## 4. `train.py`
Updated `build(cfg)` to instantiate paired physical memtransistor populations when `differential_pair = True`:

```diff
     # --- Synapse and Device setup (W_rec and W_in are on-device) ---
-    syn_rec = Synapse(_make_device((n, n), dc, dev, trc.seed + 1),
-                      sc.w_range, sc.writer, sc.verify_max_iter)
-    syn_in = Synapse(_make_device((n, n_in), dc, dev, trc.seed + 2),
-                     sc.w_range, sc.writer, sc.verify_max_iter)
+    def _make_syn_device(shape, seed_base):
+        d_pos = _make_device(shape, dc, dev, seed_base)
+        if sc.differential_pair and dc.kind != "ideal":
+            d_neg = _make_device(shape, dc, dev, seed_base + 100)
+            return (d_pos, d_neg)
+        return d_pos
+
+    syn_rec = Synapse(_make_syn_device((n, n), trc.seed + 1),
+                      sc.w_range, sc.writer, sc.verify_max_iter,
+                      differential_pair=sc.differential_pair, auto_refresh=sc.auto_refresh)
+    syn_in = Synapse(_make_syn_device((n, n_in), trc.seed + 2),
+                     sc.w_range, sc.writer, sc.verify_max_iter,
+                     differential_pair=sc.differential_pair, auto_refresh=sc.auto_refresh)
```

---

## 5. `run_matrix.py`
Added `"memtx (diff pair)"` to experimental conditions and ablation charts:

```diff
 def conditions(full=False):
     """Mapping: name -> config-diff. Base is memtransistor default (fitted, asymmetric)."""
     C = {}
     C["ideal"]             = {"device.kind": "ideal"}
     C["memtransistor"]     = {"device.kind": "memtransistor"}
+    C["memtx (diff pair)"] = {"device.kind": "memtransistor", "synapse.differential_pair": True}
```
