from __future__ import annotations
import math
import numpy as np

EPS = 1e-12
def power_score_probabilities(x, inverse_temperature: float) -> np.ndarray:
    """Normalize score**beta weights, where beta is the inverse temperature."""
    scores = np.asarray(x, dtype=float)
    base_w = np.power(scores, inverse_temperature) + 1e-10
    sum_w = float(np.sum(base_w))
    p = (base_w / sum_w) if (np.isfinite(sum_w) and sum_w > 0.0) else np.ones(len(scores), dtype=float) / len(scores)

    return p
def softmax_np(x, temp: float = 1.0) -> np.ndarray:
    """Numerically stable softmax with temperature."""
    z = np.asarray(x, dtype=float) / max(temp, EPS)
    z -= np.max(z)
    p = np.exp(z)
    s = p.sum()
    return p / s if s > 0 and np.isfinite(s) else np.ones_like(p) / len(p)

def softmax_probs(scores, beta: float) -> np.ndarray:
    """
    Softmax(beta * score). Beta plays the role of inverse temperature.
    """
    s = np.asarray(scores, dtype=float)
    if s.size == 0:
        return np.array([])
    z = beta * s
    z -= np.max(z)
    ez = np.exp(z)
    probs = ez / np.sum(ez)
    probs = np.clip(probs, 0.0, 1.0)
    ssum = probs.sum()
    return probs / ssum if ssum > 0.0 else np.ones_like(probs) / len(probs)

def safe_log_reward(score: float) -> float:
    """Log of a positive reward-like score with a small floor."""
    return float(np.log(max(float(score), EPS)))

def compute_logA_from_rewards(sa: float, sb: float, beta1: float, beta2: float, coefficient: float) -> float:
    """
    logA for PT-style swap if score is used as reward r(x).
    logA = (beta1 - beta2) * (log r(x_b) - log r(x_a))
    """
    return (beta1 - beta2) * coefficient * (safe_log_reward(sb) - safe_log_reward(sa))

def accept_from_logA(logA: float) -> bool:
    """Metropolis accept using log-probability comparison."""
    logu = float(np.log(np.random.rand()))
    return logu <= min(0.0, float(logA))

def percentile_cap(arr: np.ndarray, p_low=5.0, p_high=95.0) -> float:
    """Symmetric clipping magnitude decided by percentiles."""
    pl, ph = np.percentile(arr, [p_low, p_high])
    return float(max(abs(pl), abs(ph), 1.0))

def scale_logw_for_01(logw: np.ndarray, clip_c: float = 2.0, scale_k: float = 0.2,
                      keep_zero_center: bool = True) -> np.ndarray:
    """
    Clamp logw to [-clip_c, clip_c], then linearly map to [-scale_k, +scale_k].
    This keeps reweight impact bounded for scores ~ O(1).
    """
    lw = np.clip(logw, -clip_c, clip_c)
    if keep_zero_center:
        return (lw / max(clip_c, 1e-12)) * scale_k
    return (lw / max(clip_c, 1e-12)) * scale_k
