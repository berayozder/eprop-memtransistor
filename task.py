"""
task.py
=======
GENERATES THE HEADLINE PROBLEM the network must solve: Evidence Accumulation (Bellec 2020 Fig. 3).

Each trial gets a fresh RANDOM sequence of left/right cues. The network must COUNT
the cues, HOLD the running majority across a delay (working memory), and report
left/right in a final decision window. Because every trial is random, the network
cannot memorize -- it must genuinely use its recurrent weights (the ones stored on
the physical memtransistor array). This forces the device weights to actively learn
working memory attractors, so the device's physical flaws show up cleanly.

CUSTOM DESIGN, NOW CHECKED AGAINST THE OFFICIAL REFERENCE CODE (github.com/
IGITUGraz/eligibility_propagation, Figure_3_and_S7_e_prop_tutorials/tools.py,
generate_click_task_data(), as invoked by tutorial_evidence_accumulation_with_alif.py):

  MATCHES the reference exactly:
    - four-channel-group architecture (left/right/recall/noise) -- same design.
    - n_cues=7 (config.py's default) -- exact match.

  CONFIRMED DIFFERENT from the reference (not just "unverified" -- actually
  compared, line by line):
    - Cue generation process: the reference does NOT flip a fair coin per cue.
      It picks a per-TRIAL biased probability pair (p_group=0.3 vs 1-p_group=0.7,
      with which side gets 0.7 randomized per trial) and draws all n_cues from
      that biased distribution -- i.e. each trial has a consistent lean toward
      one side. This module instead draws each cue independently at p_left=0.5
      (a genuinely different generative process, not just a different number --
      see make_evidence_trial() below).
    - Timing scale: the reference uses t_cue=100ms, an implied gap of 50ms
      (t_interval=150 minus t_cue=100), a delay of ~1050ms, and a 150ms
      recall/decision window (total sequence ~2250ms). config.py's "full" scale
      defaults (cue_dur=15, gap=5, delay=50, decision=30; total ~220ms) are
      roughly 10x SHORTER overall -- a real, substantial compression versus the
      published task, likely for faster CPU-only training, not a byte-for-byte
      reproduction as "full scale" might suggest.
    - Spike rates: the reference uses a cue rate of 0.04 and a noise rate of 0.01
      (f0=40Hz via input_f0=40/1000, noise=f0/4); config.py's cue_rate=0.4 and
      noise_rate=0.05 are 10x and 5x higher respectively.
  See config.py's NeuronConfig for the corresponding tau_a/beta comparison.
"""
from __future__ import annotations
import math
import torch


# ========================== EVIDENCE ACCUMULATION (headline) ==========================
def evidence_geometry(tc):
    """Derive the input size and sequence length of the evidence task from its config.

    Returns:
        n_in: number of input channels = 4 groups x n_group (left/right/recall/noise).
        T:    total steps = cue phase + delay + decision window.
    """
    n_in = 4 * tc.n_group
    T = tc.n_cues * (tc.cue_dur + tc.gap) + tc.delay + tc.decision
    return n_in, T


def make_evidence_trial(tc, gen, torch_device="cpu", dtype=torch.float32):
    """Build ONE random evidence-accumulation trial.

    Input channels are split into four groups, each of size n_group:
        [0:ng]      = "left" cue channels
        [ng:2ng]    = "right" cue channels
        [2ng:3ng]   = "recall" channel (the "decide now" signal in the decision window)
        [3ng:4ng]   = pure noise channel

    Args:
        tc:  TaskConfig (evidence parameters).
        gen: a torch RNG generator (varied per trial so each trial is different).
    Returns:
        X:     [T, n_in] input spikes.
        Ystar: [T, 2] one-hot target, non-zero only in the decision window.
        mask:  [T] 1 inside the decision window, else 0 (which steps count).
        label: 0 = left majority, 1 = right majority.
    """
    ng = tc.n_group
    n_in, T = evidence_geometry(tc)

    # Background noise on every channel (low rate) -- makes the task realistically hard.
    X = (torch.rand(T, n_in, generator=gen) < tc.noise_rate).to(dtype)

    # Cues: each cue is randomly a "left" or "right" cue.
    sides_left = (torch.rand(tc.n_cues, generator=gen) < tc.p_left)   # True = this cue is "left"
    for c in range(tc.n_cues):
        start = c * (tc.cue_dur + tc.gap)                            # when this cue begins
        grp = slice(0, ng) if bool(sides_left[c]) else slice(ng, 2 * ng)   # left or right channels
        spikes = (torch.rand(tc.cue_dur, ng, generator=gen) < tc.cue_rate).to(dtype)
        X[start:start + tc.cue_dur, grp] = spikes                   # fire the chosen group during the cue

    # Recall cue: fires the recall channels during the decision window ("report now").
    dec_start = tc.n_cues * (tc.cue_dur + tc.gap) + tc.delay
    recall = (torch.rand(tc.decision, ng, generator=gen) < tc.cue_rate).to(dtype)
    X[dec_start:dec_start + tc.decision, 2 * ng:3 * ng] = recall

    # Correct answer = majority side. n_cues is odd, so there is never a tie.
    n_left = int(sides_left.sum().item())
    label = 0 if n_left > (tc.n_cues - n_left) else 1               # 0 = left, 1 = right

    # Target and mask are non-zero only in the decision window.
    Ystar = torch.zeros(T, 2, dtype=dtype)
    mask = torch.zeros(T, dtype=dtype)
    Ystar[dec_start:dec_start + tc.decision, label] = 1.0
    mask[dec_start:dec_start + tc.decision] = 1.0

    return (X.to(torch_device), Ystar.to(torch_device),
            mask.to(torch_device), label)