#!/usr/bin/env python
"""
Experiments for the Scientific Reports manuscript "Threshold calibration and
event localization rather than detector architecture limit sequence-based zero
trust access decisions".

The reference model and data pipeline are those of the original study:
  * CERT r4.2 logon.csv + http.csv only; one categorical feature per event
    (Logoff = 0, Logon = 1, HTTP request = 2), no scaling;
  * 80/20 user-level split with GroupShuffleSplit(random_state=42);
  * sliding windows of T = 10 events built per user after the split;
  * 500,000 training windows sampled with np.random.seed(42);
  * LSTM(32, relu) -> RepeatVector -> LSTM(32, relu) -> TimeDistributed(Dense(1));
  * Adam, MSE, batch 512, <= 10 epochs, early stopping (patience 3), val split 0.1;
  * EVT/POT with a 90th-percentile initial threshold and q = 0.01.

What is changed (the protocol fixes requested by the reviewers):
  * every model (LSTM, GRU, Bi-LSTM, 1D-CNN, Dense AE, Isolation Forest, OC-SVM)
    gets its thresholds from its OWN training-set scores (EVT and a static 99th
    percentile); nothing is estimated on evaluation data;
  * all neural baselines share the same hyperparameters as the LSTM;
  * the Dense AE uses a linear output layer (the original sigmoid output could not
    reproduce the value 2 and invalidated the ablation);
  * with raw CSVs (--data-dir) benign evaluation windows are sampled across all
    held-out users; with the saved arrays (--arrays-dir) the original
    single-user evaluation set is used and a many-user hold-out check is added;
  * five seeds for every model, mean +/- SD, bootstrap CIs (user clusters with
    raw CSVs, blocks of consecutive windows with saved arrays);
  * new experiments: controlled drift, EVT sensitivity and GPD
    goodness of fit, SEL localization accuracy against attribution baselines,
    latency breakdown up to a REST controller endpoint, adaptive/mimicry
    anomalies, training-data contamination and window-size checks.

Usage
-----
  python zt_rerun.py --arrays-dir . --out results                  # from saved arrays
  python zt_rerun.py --data-dir r4.2 --out results                 # from raw CSVs
  python zt_rerun.py --data-dir r4.2 --out results_quick --quick   # ~25 min check
  python zt_rerun.py ... --stages main,sens                        # subset
  python zt_rerun.py --out results --stages report                 # rebuild outputs

Every stage writes a JSON checkpoint to <out>/; completed stages and seeds are
skipped on a re-run unless --force is given.

--backend numpy replaces the neural models by a tiny linear autoencoder. It
exists ONLY for smoke-testing the pipeline without TensorFlow; never report its
numbers.
"""
import argparse
import json
import os
import platform
import sys
import threading
import time
from dataclasses import dataclass, field, asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pandas as pd

TOKENS = {"Logoff": 0, "Logon": 1, "http_request": 2}
TOKEN_NAMES = {0: "Logoff", 1: "Logon", 2: "HTTP"}
NN_MODELS = ["lstm", "gru", "bilstm", "cnn1d", "dense"]
ALL_MODELS = NN_MODELS + ["iforest", "ocsvm"]
MODEL_LABEL = {
    "lstm": "LSTM-AE (reference)", "gru": "GRU-AE", "bilstm": "Bi-LSTM-AE",
    "cnn1d": "1D-CNN-AE", "dense": "Dense-AE", "iforest": "Isolation Forest",
    "ocsvm": "One-class SVM",
}


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class Config:
    data_dir: str = "r4.2"
    arrays_dir: str = None  # folder with X_train_lstm.pkl, X_test_eval.pkl, y_true.pkl
    out: str = "results"
    backend: str = "keras"
    seeds: list = field(default_factory=lambda: [42, 100, 2024, 777, 999])
    T: int = 10
    n_train: int = 500_000
    n_test_normal: int = 9_000
    n_anom: int = 1_000
    epochs: int = 10
    batch: int = 512
    val_split: float = 0.1
    patience: int = 3
    t_init_pct: float = 90.0
    q: float = 0.01
    classical_subset: int = 50_000
    eval_seed: int = 2026
    n_boot: int = 1000
    # drift
    drift_train_months: int = 6
    drift_shift_start: int = 9
    drift_shift_len: int = 3
    drift_p_max: float = 0.05
    drift_normals_per_month: int = 3000
    drift_buffer: int = 50_000
    # SEL
    n_sel: int = 2000
    # contamination / window size
    contam_rates: list = field(default_factory=lambda: [0.01, 0.05])
    window_sizes: list = field(default_factory=lambda: [5, 20])
    force: bool = False
    quick: bool = False

    def apply_quick(self):
        self.quick = True
        self.seeds = [42, 100]
        self.n_train = 100_000
        self.epochs = 3
        self.classical_subset = 10_000
        self.n_boot = 200
        self.drift_normals_per_month = 1000
        self.drift_buffer = 10_000
        self.n_sel = 500
        self.contam_rates = [0.05]
        self.window_sizes = [20]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def jdump(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=_json_default)


def jload(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
class EventData:
    """All events sorted by (user, time), with per-user offsets."""

    def __init__(self, users, user_codes, ts, act):
        self.users = users                 # sorted array of user IDs (str)
        order = np.lexsort((ts, user_codes))
        self.ucode = user_codes[order]
        self.ts = ts[order]                # int64 seconds since epoch
        self.act = act[order].astype(np.int8)
        counts = np.bincount(self.ucode, minlength=len(users))
        self.offset = np.concatenate([[0], np.cumsum(counts)[:-1]])
        self.count = counts

    def n_windows(self, u, T, stride=1):
        n = self.count[u] - T  # original code: range(len(actions) - T)
        if n <= 0:
            return 0
        return (n + stride - 1) // stride

    def gather(self, users_idx, starts, T):
        """Windows for (user, start) pairs -> (N, T) int8 plus end timestamps."""
        base = self.offset[users_idx] + starts
        idx = base[:, None] + np.arange(T)[None, :]
        return self.act[idx], self.ts[idx[:, -1]]


def load_cert(data_dir, chunksize=5_000_000):
    """Load logon.csv and http.csv in chunks (memory-safe on 16 GB)."""
    user_index = {}
    u_parts, t_parts, a_parts = [], [], []

    def codes_for(series):
        vals = series.astype(str).to_numpy()
        out = np.empty(len(vals), dtype=np.int32)
        for i, v in enumerate(vals):
            c = user_index.get(v)
            if c is None:
                c = len(user_index)
                user_index[v] = c
            out[i] = c
        return out

    def to_seconds(s):
        d = pd.to_datetime(s, format="%m/%d/%Y %H:%M:%S")
        return d.to_numpy().astype("datetime64[s]").astype(np.int64)

    log("Loading logon.csv ...")
    for ch in pd.read_csv(os.path.join(data_dir, "logon.csv"),
                          usecols=["date", "user", "activity"], chunksize=chunksize):
        u_parts.append(codes_for(ch["user"]))
        t_parts.append(to_seconds(ch["date"]))
        a_parts.append(ch["activity"].map({"Logoff": 0, "Logon": 1}).to_numpy(np.int8))
    log("Loading http.csv (large; several minutes) ...")
    nread = 0
    for i, ch in enumerate(pd.read_csv(os.path.join(data_dir, "http.csv"),
                                       usecols=["date", "user"], chunksize=chunksize)):
        nread += len(ch)
        u_parts.append(codes_for(ch["user"]))
        t_parts.append(to_seconds(ch["date"]))
        a_parts.append(np.full(len(ch), 2, dtype=np.int8))
        log(f"  http rows read: {nread:,}")
    ucode = np.concatenate(u_parts)
    ts = np.concatenate(t_parts)
    act = np.concatenate(a_parts)
    # re-code users so that code order == sorted user ID order (as pandas groupby)
    names = np.array(list(user_index.keys()))
    order = np.argsort(names)
    remap = np.empty(len(names), dtype=np.int32)
    remap[order] = np.arange(len(names))
    return EventData(names[order], remap[ucode], ts, act)


def user_split(ev):
    from sklearn.model_selection import GroupShuffleSplit
    gss = GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=42)
    users = ev.users
    tr, te = next(gss.split(users, groups=users))
    return np.sort(tr), np.sort(te)


def enumerate_windows(ev, users_idx, T, stride=1):
    counts = np.array([ev.n_windows(u, T, stride) for u in users_idx], dtype=np.int64)
    return counts


def sample_global(ev, users_idx, T, n, rng_legacy=True, seed=42, stride=1):
    """Reproduces the original sampling: windows enumerated user by user (sorted)
    and np.random.seed(seed); np.random.choice(N, n, replace=False)."""
    counts = enumerate_windows(ev, users_idx, T, stride)
    total = int(counts.sum())
    if rng_legacy:
        np.random.seed(seed)
        if total > n:
            g = np.random.choice(total, n, replace=False)
        else:
            g = np.arange(total)
    else:
        rng = np.random.default_rng(seed)
        g = rng.choice(total, min(n, total), replace=False)
    cum = np.concatenate([[0], np.cumsum(counts)])
    pos = np.searchsorted(cum, g, side="right") - 1
    starts = (g - cum[pos]) * stride
    X, tend = ev.gather(users_idx[pos], starts, T)
    return X, tend, users_idx[pos], total


def original_first_windows(ev, test_users, T, n):
    """The original evaluation set: the first n windows in user-sorted order."""
    us, ss = [], []
    for u in test_users:
        w = ev.n_windows(u, T)
        if w <= 0:
            continue
        take = min(w, n - len(ss))
        us.extend([u] * take)
        ss.extend(range(take))
        if len(ss) >= n:
            break
    us = np.array(us)
    X, tend = ev.gather(us, np.array(ss), T)
    return X, us


def stratified_windows(ev, users_idx, T, n, rng, stride=1, t_lo=None, t_hi=None,
                       return_starts=False):
    """n windows spread as evenly as possible across users (optionally restricted
    to windows whose last event lies in [t_lo, t_hi))."""
    pools = []
    for u in users_idx:
        w = ev.n_windows(u, T, stride)
        if w <= 0:
            pools.append(np.empty(0, dtype=np.int64))
            continue
        starts = np.arange(w, dtype=np.int64) * stride
        if t_lo is not None:
            end_ts = ev.ts[ev.offset[u] + starts + T - 1]
            starts = starts[(end_ts >= t_lo) & (end_ts < t_hi)]
        pools.append(starts)
    sizes = np.array([len(p) for p in pools])
    active = np.flatnonzero(sizes > 0)
    if len(active) == 0:
        empty = (np.empty((0, T), np.int8), np.empty(0), np.empty(0, np.int64))
        return empty + (np.empty(0, np.int64),) if return_starts else empty
    alloc = np.zeros(len(users_idx), dtype=np.int64)
    remaining = min(n, int(sizes.sum()))
    while remaining > 0:
        cand = active[alloc[active] < sizes[active]]
        if len(cand) == 0:
            break
        share = max(remaining // len(cand), 1)
        for u in rng.permutation(cand):
            add = min(share, sizes[u] - alloc[u], remaining)
            alloc[u] += add
            remaining -= add
            if remaining == 0:
                break
    us, ss = [], []
    for i in np.flatnonzero(alloc):
        pick = rng.choice(pools[i], alloc[i], replace=False)
        us.append(np.full(alloc[i], users_idx[i]))
        ss.append(pick)
    us = np.concatenate(us)
    ss = np.concatenate(ss)
    X, tend = ev.gather(us, ss, T)
    if return_starts:
        return X, tend, us, ss
    return X, tend, us


def bigram_model(X, alpha=1.0):
    C = np.full((3, 3), alpha)
    a, b = X[:, :-1].ravel(), X[:, 1:].ravel()
    np.add.at(C, (a, b), 1)
    P = C / C.sum(1, keepdims=True)
    uni = np.bincount(X.ravel(), minlength=3) + alpha
    return P, uni / uni.sum(), C - alpha


def anomalies_random(n, T, rng):
    """Original anomaly type: tokens drawn uniformly from {Logoff, HTTP}."""
    return rng.choice([0, 2], size=(n, T)).astype(np.int8)


def anomalies_single(Xb, P, rng):
    """One event per benign window replaced by the least plausible token given its
    neighbours; returns windows and the ground-truth position."""
    X = Xb.copy()
    n, T = X.shape
    pos = rng.integers(0, T, n)
    logP = np.log(P)
    for i in range(n):
        t = pos[i]
        best, best_s = None, np.inf
        for tok in range(3):
            if tok == X[i, t]:
                continue
            s = 0.0
            if t > 0:
                s += logP[X[i, t - 1], tok]
            if t < T - 1:
                s += logP[tok, X[i, t + 1]]
            if s < best_s:
                best, best_s = tok, s
        X[i, t] = best
    return X, pos


def anomalies_mimicry(n, T, P, uni, rare, rng):
    """Benign-like sequences sampled from the first-order Markov chain fitted on
    training windows, with exactly one rare transition (a, b) inserted."""
    X = np.zeros((n, T), dtype=np.int8)
    for i in range(n):
        x = [rng.choice(3, p=uni)]
        for _ in range(T - 1):
            x.append(rng.choice(3, p=P[x[-1]]))
        p = rng.integers(0, T - 1)
        x[p], x[p + 1] = rare
        X[i] = x
    return X


def apply_reauth_shift(ev, us, starts_end_idx, T, p, rng):
    """Benign workflow change: after each event insert (Logoff, Logon) with
    probability p; keep the last T events. Applied to the 2T events ending at the
    window end so the result is a valid local rewrite of the stream."""
    out = np.empty((len(us), T), dtype=np.int8)
    for i in range(len(us)):
        hi = starts_end_idx[i] + 1
        lo = max(ev.offset[us[i]], hi - 2 * T)
        seq = []
        for tok in ev.act[lo:hi]:
            seq.append(int(tok))
            if p > 0 and rng.random() < p:
                seq.extend([0, 1])
        out[i] = seq[-T:]
    return out


# --------------------------------------------------------------------------- #
# Scoring / thresholds / metrics
# --------------------------------------------------------------------------- #
def evt_threshold(scores, t_pct=90.0, q=0.01, return_params=False):
    from scipy.stats import genpareto
    scores = np.asarray(scores, dtype=np.float64)
    t0 = np.percentile(scores, t_pct)
    peaks = scores[scores > t0] - t0
    N, Nt = len(scores), len(peaks)
    if Nt < 10:
        z, xi, beta = t0, np.nan, np.nan
    else:
        xi, _, beta = genpareto.fit(peaks, floc=0)
        r = q * N / Nt
        if r >= 1:  # q not inside the modelled tail: z_q undefined for this t_init
            z = np.nan
        elif abs(xi) > 1e-12:
            z = t0 + (beta / xi) * (r ** (-xi) - 1)
        else:
            z = t0 - beta * np.log(r)
    if return_params:
        return float(z), dict(t_init=float(t0), xi=float(xi), beta=float(beta), N=N, Nt=Nt)
    return float(z)


def metrics(y, s, thr):
    from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
    y = np.asarray(y).astype(int)
    pred = (s > thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    P = tp / (tp + fp) if tp + fp else 0.0
    R = tp / (tp + fn) if tp + fn else 0.0
    F1 = 2 * P * R / (P + R) if P + R else 0.0
    den = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / den if den else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    out = dict(threshold=float(thr), TP=tp, FP=fp, TN=tn, FN=fn, precision=P, recall=R,
               f1=F1, mcc=mcc, fpr=fpr, accuracy=(tp + tn) / len(y))
    if len(np.unique(y)) == 2:
        out["roc_auc"] = float(roc_auc_score(y, s))
        out["pr_auc"] = float(average_precision_score(y, s))
        f, t, _ = roc_curve(y, s)
        for target in (0.001, 0.01):
            out[f"tpr_at_fpr_{target}"] = float(np.interp(target, f, t))
    return out


def cluster_bootstrap(y, s, thr, groups, n_boot, rng):
    """CI by resampling benign windows clustered by user and anomalies i.i.d."""
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y)
    neg = np.flatnonzero(y == 0); pos = np.flatnonzero(y == 1)
    g_neg = groups[neg]
    ug = np.unique(g_neg)
    by_g = {g: neg[g_neg == g] for g in ug}
    aucs, f1s, mccs = [], [], []
    for _ in range(n_boot):
        gs = rng.choice(ug, len(ug), replace=True)
        idx_n = np.concatenate([by_g[g] for g in gs])
        idx_p = rng.choice(pos, len(pos), replace=True)
        idx = np.concatenate([idx_n, idx_p])
        m = metrics(y[idx], s[idx], thr)
        aucs.append(roc_auc_score(y[idx], s[idx])); f1s.append(m["f1"]); mccs.append(m["mcc"])
    ci = lambda a: [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]
    return dict(roc_auc_ci=ci(aucs), f1_ci=ci(f1s), mcc_ci=ci(mccs))


# --------------------------------------------------------------------------- #
# Model backends
# --------------------------------------------------------------------------- #
class KerasBackend:
    name = "keras"

    def __init__(self):
        import tensorflow as tf  # noqa: F401
        from tensorflow import keras
        self.tf = tf
        self.keras = keras

    def set_seed(self, seed):
        self.keras.utils.set_random_seed(seed)

    def build(self, name, T, F=1):
        k = self.keras
        L = k.layers
        inp = k.Input(shape=(T, F))
        if name == "lstm":
            x = L.LSTM(32, activation="relu")(inp)
            x = L.RepeatVector(T)(x)
            x = L.LSTM(32, activation="relu", return_sequences=True)(x)
            out = L.TimeDistributed(L.Dense(F))(x)
        elif name == "gru":
            x = L.GRU(32, activation="relu")(inp)
            x = L.RepeatVector(T)(x)
            x = L.GRU(32, activation="relu", return_sequences=True)(x)
            out = L.TimeDistributed(L.Dense(F))(x)
        elif name == "bilstm":
            x = L.Bidirectional(L.LSTM(32, activation="relu"))(inp)
            x = L.RepeatVector(T)(x)
            x = L.Bidirectional(L.LSTM(32, activation="relu", return_sequences=True))(x)
            out = L.TimeDistributed(L.Dense(F))(x)
        elif name == "cnn1d":
            x = L.Conv1D(32, 3, activation="relu", padding="same")(inp)
            x = L.MaxPooling1D(2, padding="same")(x)
            x = L.Conv1D(16, 3, activation="relu", padding="same")(x)
            x = L.UpSampling1D(2)(x)
            if (T % 2) == 1:
                x = L.Cropping1D((0, 1))(x)
            out = L.Conv1D(F, 3, activation="linear", padding="same")(x)
        elif name == "dense":
            x = L.Reshape((T * F,))(inp)
            x = L.Dense(64, activation="relu")(x)
            x = L.BatchNormalization()(x)
            x = L.Dense(16, activation="relu")(x)
            x = L.BatchNormalization()(x)
            x = L.Dense(64, activation="relu")(x)
            x = L.Dense(T * F, activation="linear")(x)  # linear output (bug fix)
            out = L.Reshape((T, F))(x)
        else:
            raise ValueError(name)
        m = k.Model(inp, out, name=name)
        m.compile(optimizer=k.optimizers.Adam(1e-3), loss="mse")
        return m

    def fit(self, model, X, cfg):
        k = self.keras
        Xf = X[..., None].astype(np.float32)
        es = k.callbacks.EarlyStopping(monitor="val_loss", patience=cfg.patience,
                                       restore_best_weights=True)
        h = model.fit(Xf, Xf, epochs=cfg.epochs, batch_size=cfg.batch,
                      validation_split=cfg.val_split, callbacks=[es], verbose=2)
        return {"loss": [float(v) for v in h.history["loss"]],
                "val_loss": [float(v) for v in h.history["val_loss"]]}

    def recon(self, model, X, batch=4096):
        Xf = X[..., None].astype(np.float32)
        return model.predict(Xf, batch_size=batch, verbose=0)[..., 0]

    def n_params(self, model):
        return int(model.count_params())

    def save(self, model, path):
        model.save(path + ".keras")

    def load(self, path):
        return self.keras.models.load_model(path + ".keras")

    def exists(self, path):
        return os.path.exists(path + ".keras")

    def saliency(self, model, X):
        tf = self.tf
        x = tf.convert_to_tensor(X[..., None].astype(np.float32))
        with tf.GradientTape() as tape:
            tape.watch(x)
            y = model(x, training=False)
            loss = tf.reduce_mean(tf.square(x - y), axis=[1, 2])
        g = tape.gradient(loss, x).numpy()[..., 0]
        return np.abs(g)

    def single_forward_fns(self, model, T):
        tf = self.tf
        spec = tf.TensorSpec(shape=(None, T, 1), dtype=tf.float32)
        compiled = tf.function(lambda x: model(x, training=False), input_signature=[spec])

        def eager(x):
            return model(x, training=False).numpy()

        def graph(x):
            return compiled(tf.constant(x)).numpy()

        def predict(x):
            return model.predict(x, verbose=0)
        return {"eager_call": eager, "tf_function": graph, "keras_predict": predict}


class NumpyBackend:
    """Smoke-test stand-in: PCA autoencoder. Not a substitute for the real models."""
    name = "numpy"

    class _M:
        def __init__(self, k):
            self.k = k

    def set_seed(self, seed):
        np.random.seed(seed)

    def build(self, name, T, F=1):
        return self._M({"lstm": 3, "gru": 3, "bilstm": 4, "cnn1d": 2, "dense": 5}[name])

    def fit(self, m, X, cfg):
        Xf = X.astype(np.float64)
        m.mu = Xf.mean(0)
        _, _, Vt = np.linalg.svd(Xf[:20000] - m.mu, full_matrices=False)
        m.W = Vt[: m.k].T
        return {"loss": [0.3, 0.1, 0.05][: cfg.epochs], "val_loss": [0.25, 0.09, 0.05][: cfg.epochs]}

    def recon(self, m, X, batch=None):
        Xf = X.astype(np.float64) - m.mu
        return Xf @ m.W @ m.W.T + m.mu

    def n_params(self, m):
        return int(m.W.size + m.mu.size)

    def save(self, m, path):
        np.savez(path + ".npz", mu=m.mu, W=m.W, k=m.k)

    def load(self, path):
        d = np.load(path + ".npz")
        m = self._M(int(d["k"])); m.mu, m.W = d["mu"], d["W"]
        return m

    def exists(self, path):
        return os.path.exists(path + ".npz")

    def saliency(self, m, X, eps=1e-3):
        base = ((X - self.recon(m, X)) ** 2).mean(1)
        out = np.zeros(X.shape)
        for t in range(X.shape[1]):
            Xp = X.astype(np.float64).copy(); Xp[:, t] += eps
            out[:, t] = np.abs((((Xp - self.recon(m, Xp)) ** 2).mean(1) - base) / eps)
        return out

    def single_forward_fns(self, m, T):
        f = lambda x: self.recon(m, x[..., 0])[..., None]
        return {"eager_call": f, "tf_function": f, "keras_predict": f}


def get_backend(name):
    return KerasBackend() if name == "keras" else NumpyBackend()


def window_mse(X, R):
    return ((X.astype(np.float64) - R) ** 2).mean(1)


def step_errors(X, R):
    return (X.astype(np.float64) - R) ** 2


# --------------------------------------------------------------------------- #
# Shared data preparation (cached)
# --------------------------------------------------------------------------- #
class Prepared:
    pass


_EV = {}


def get_ev(cfg):
    if "ev" not in _EV:
        _EV["ev"] = load_cert(cfg.data_dir)
    return _EV["ev"]


def prepare(cfg, T=None, need_ev=False):
    if cfg.arrays_dir:
        return prepare_arrays(cfg)
    T = T or cfg.T
    cache = os.path.join(cfg.out, f"cache_T{T}_n{cfg.n_train}.npz")
    meta_path = os.path.join(cfg.out, f"cache_T{T}_n{cfg.n_train}.json")
    p = Prepared()
    p.T = T
    if os.path.exists(cache) and not need_ev and not getattr(cfg, "_rebuild_cache", False):
        d = np.load(cache)
        for k in d.files:
            setattr(p, k, d[k])
        p.meta = jload(meta_path)
        return p
    ev = get_ev(cfg)
    tr_users, te_users = user_split(ev)
    p.ev, p.tr_users, p.te_users = ev, tr_users, te_users
    if os.path.exists(cache) and not getattr(cfg, "_rebuild_cache", False):
        d = np.load(cache)
        for k in d.files:
            setattr(p, k, d[k])
        p.meta = jload(meta_path)
        return p
    log(f"Preparing windows (T={T}) ...")
    # training windows + original anomaly set: identical RNG sequence to the original
    Xtr, _, utr, total = sample_global(ev, tr_users, T, cfg.n_train, rng_legacy=True, seed=42)
    anomA = np.random.choice([0, 2], size=(cfg.n_anom, T, 1))[..., 0].astype(np.int8)
    rng = np.random.default_rng(cfg.eval_seed)
    Xn, tn, un = stratified_windows(ev, te_users, T, cfg.n_test_normal, rng)
    Xo, uo = original_first_windows(ev, te_users, T, cfg.n_test_normal)
    Xno, _, uno = stratified_windows(ev, te_users, T, cfg.n_test_normal, rng, stride=T)
    P, uni, C = bigram_model(Xtr)
    flat = [(C[a, b], (a, b)) for a in range(3) for b in range(3)]
    rare_count, rare = min(flat)
    Xsb, _, usb = stratified_windows(ev, te_users, T, cfg.n_sel, rng)
    anomB, posB = anomalies_single(Xsb, P, rng)
    anomC = anomalies_mimicry(cfg.n_anom, T, P, uni, rare, rng)
    p.Xtr, p.utr, p.anomA, p.Xn, p.tn, p.un = Xtr, utr, anomA, Xn, tn, un
    p.Xo, p.uo, p.Xno, p.uno = Xo, uo, Xno, uno
    p.anomB, p.posB, p.Xsb, p.anomC, p.P, p.uni = anomB, posB, Xsb, anomC, P, uni
    tok = np.bincount(ev.act, minlength=3)
    p.meta = dict(
        n_events=int(len(ev.act)), n_users=int(len(ev.users)),
        n_train_users=int(len(tr_users)), n_test_users=int(len(te_users)),
        n_train_windows_total=total, n_train_sampled=int(len(Xtr)),
        n_test_normal=int(len(Xn)), n_test_users_represented=int(len(np.unique(un))),
        n_orig_normal_users=int(len(np.unique(uo))), n_nonoverlap_users=int(len(np.unique(uno))),
        n_anom=int(len(anomA)), date_start=str(pd.to_datetime(ev.ts.min(), unit="s").date()),
        date_end=str(pd.to_datetime(ev.ts.max(), unit="s").date()),
        token_share={TOKEN_NAMES[i]: float(tok[i] / tok.sum()) for i in range(3)},
        train_token_share={TOKEN_NAMES[i]: float(uni[i]) for i in range(3)},
        rare_bigram=f"{TOKEN_NAMES[rare[0]]}->{TOKEN_NAMES[rare[1]]}",
        rare_bigram_count=int(rare_count),
        rare_bigram_share=float(rare_count / C.sum()),
        bigram_P=P.tolist(),
        anomA_share_with_logon=float((anomA == 1).any(1).mean()),
        normal_share_all_http=float((Xn == 2).all(1).mean()),
    )
    arrays = {k: getattr(p, k) for k in ["Xtr", "utr", "anomA", "Xn", "tn", "un", "Xo", "uo",
                                         "Xno", "uno", "anomB", "posB", "Xsb", "anomC", "P", "uni"]}
    np.savez_compressed(cache, **arrays)
    jdump(p.meta, meta_path)
    return p



# --------------------------------------------------------------------------- #
# Arrays mode: the raw CERT CSVs are not available; use the saved arrays
# --------------------------------------------------------------------------- #
def _load_pkl(path):
    import joblib
    return joblib.load(path)


def benign_stream(Xn):
    """Rebuild the event stream behind consecutive stride-1 windows. A new segment
    starts wherever window i+1 is not window i shifted by one event (user boundary)."""
    T = Xn.shape[1]
    S = list(Xn[0]); seg_start = [0]; ends = [T - 1]; seg_of = [0]; seg = 0
    for i in range(1, len(Xn)):
        if np.array_equal(Xn[i, :-1], Xn[i - 1, 1:]):
            S.append(int(Xn[i, -1]))
        else:
            seg += 1
            seg_start.append(len(S))
            S.extend(int(v) for v in Xn[i])
        ends.append(len(S) - 1)
        seg_of.append(seg)
    return np.array(S, dtype=np.int8), np.array(ends), np.array(seg_of), np.array(seg_start)


def prepare_arrays(cfg):
    if hasattr(prepare_arrays, "_p"):
        return prepare_arrays._p
    d = cfg.arrays_dir
    Xtr = np.asarray(_load_pkl(os.path.join(d, "X_train_lstm.pkl")))
    Xev = np.asarray(_load_pkl(os.path.join(d, "X_test_eval.pkl")))
    y = np.asarray(_load_pkl(os.path.join(d, "y_true.pkl"))).astype(int).ravel()
    Xtr = Xtr.reshape(len(Xtr), -1).astype(np.int8)
    Xev = Xev.reshape(len(Xev), -1).astype(np.int8)
    T = Xtr.shape[1]
    assert T == cfg.T, f"saved windows have length {T}, expected {cfg.T}"
    if len(Xtr) > cfg.n_train:  # quick mode
        Xtr = Xtr[np.random.default_rng(0).choice(len(Xtr), cfg.n_train, replace=False)]
    p = Prepared()
    p.T = T
    p.Xtr = Xtr
    p.Xn = Xev[y == 0]
    p.anomA = Xev[y == 1]
    S, ends, seg_of, seg_start = benign_stream(p.Xn)
    p.stream, p.stream_ends, p.seg_of, p.seg_start = S, ends, seg_of, seg_start
    # block ids for the bootstrap: blocks of T consecutive windows inside a segment
    p.un = seg_of * 100000 + np.arange(len(p.Xn)) // T
    p.Xo, p.uo = p.Xn, p.un
    p.Xno = p.Xn[::T]; p.uno = p.un[::T]
    rng = np.random.default_rng(cfg.eval_seed)
    P, uni, C = bigram_model(Xtr)
    flat = [(C[a, b], (a, b)) for a in range(3) for b in range(3)]
    rare_count, rare = min(flat)
    sel_idx = rng.choice(len(p.Xn), min(cfg.n_sel, len(p.Xn)), replace=False)
    p.Xsb = p.Xn[sel_idx]
    p.anomB, p.posB = anomalies_single(p.Xsb, P, rng)
    p.anomC = anomalies_mimicry(len(p.anomA), T, P, uni, rare, rng)
    p.P, p.uni = P, uni
    tok = np.bincount(Xtr.ravel(), minlength=3)
    p.meta = dict(
        source="saved arrays (raw CERT logs not available)",
        n_users=1000, n_train_users=800, n_test_users=200,
        n_train_windows_total=None, n_train_sampled=int(len(Xtr)),
        n_test_normal=int(len(p.Xn)), n_test_users_represented=None,
        n_benign_segments=int(seg_of.max() + 1),
        n_orig_normal_users=int(seg_of.max() + 1),
        n_nonoverlap_users=None, n_anom=int(len(p.anomA)),
        anomA_only_logoff_http=float(np.isin(p.anomA, [0, 2]).all(1).mean()),
        date_start=None, date_end=None,
        token_share={TOKEN_NAMES[i]: float(tok[i] / tok.sum()) for i in range(3)},
        train_token_share={TOKEN_NAMES[i]: float(uni[i]) for i in range(3)},
        rare_bigram=f"{TOKEN_NAMES[rare[0]]}->{TOKEN_NAMES[rare[1]]}",
        rare_bigram_count=int(rare_count), rare_bigram_share=float(rare_count / C.sum()),
        bigram_P=P.tolist(),
        normal_share_all_http=float((p.Xn == 2).all(1).mean()),
        n_unique_benign=int(len(np.unique(p.Xn, axis=0))),
        n_events=None,
    )
    log(f"Arrays loaded: train {Xtr.shape}, benign test {p.Xn.shape}, anomalies {p.anomA.shape}, "
        f"benign windows form {p.meta['n_benign_segments']} contiguous segment(s); "
        f"{100 * p.meta['anomA_only_logoff_http']:.1f}% of anomalies use only Logoff/HTTP")
    prepare_arrays._p = p
    return p


def shift_from_stream(S, seg_start, end, T, p_ins, rng):
    lo = max(seg_start, end - 2 * T + 1)
    seq = []
    for tok in S[lo:end + 1]:
        seq.append(int(tok))
        if p_ins > 0 and rng.random() < p_ins:
            seq.extend([0, 1])
    return seq[-T:]


def stage_drift_arrays(cfg, be):
    """Time-ordered drift on the consecutive held-out windows (no raw timestamps)."""
    path = os.path.join(cfg.out, "stage_drift.json")
    if os.path.exists(path) and not cfg.force:
        return jload(path)
    from sklearn.metrics import roc_auc_score
    p = prepare(cfg)
    T = p.T
    m = be.load(os.path.join(cfg.out, "models", f"lstm_{cfg.seeds[0]}"))
    s_tr = window_mse(p.Xtr, be.recon(m, p.Xtr))
    z_fixed, par = evt_threshold(s_tr, cfg.t_init_pct, cfg.q, return_params=True)
    s99 = float(np.percentile(s_tr, 99))
    rng = np.random.default_rng(cfg.eval_seed + 1)
    n_blocks, shift_start, shift_len = 12, 3, 3
    blocks = np.array_split(np.arange(len(p.Xn)), n_blocks)
    seed_buf = list(s_tr[-cfg.drift_buffer:])
    buf = {"evt_spot": list(seed_buf), "evt_incl": list(seed_buf), "rolling99": list(seed_buf)}
    thr = {"static99": s99, "evt_fixed": z_fixed, "evt_spot": z_fixed, "evt_incl": z_fixed, "rolling99": s99}
    rows = []
    for k, idx in enumerate(blocks):
        if k < shift_start:
            pk, phase = 0.0, "before"
        elif k < shift_start + shift_len:
            pk, phase = cfg.drift_p_max * (k - shift_start + 1) / shift_len, "during"
        else:
            pk, phase = cfg.drift_p_max, "after"
        Xs = np.array([shift_from_stream(p.stream, p.seg_start[p.seg_of[i]], p.stream_ends[i], T, pk, rng)
                       for i in idx], dtype=np.int8)
        ev_i, op_i = np.arange(0, len(idx), 2), np.arange(1, len(idx), 2)
        Xe = Xs[ev_i]
        nA = max(int(round(len(Xe) / 9)), 1)
        A = anomalies_random(nA, T, rng)
        X = np.r_[Xe, A]
        y = np.r_[np.zeros(len(Xe)), np.ones(nA)].astype(int)
        s = window_mse(X, be.recon(m, X))
        Xb = np.r_[p.Xn[idx][ev_i], A]
        s_noshift = window_mse(Xb, be.recon(m, Xb))
        row = dict(month=k, phase=phase, p_shift=pk, n_normal=int(len(Xe)), n_anom=nA,
                   auc_shift=float(roc_auc_score(y, s)), auc_noshift=float(roc_auc_score(y, s_noshift)))
        for name, z in thr.items():
            mt = metrics(y, s, z)
            row[name] = dict(threshold=z, fpr=mt["fpr"], recall=mt["recall"],
                             precision=mt["precision"], f1=mt["f1"], mcc=mt["mcc"])
        rows.append(row)
        Xo = Xs[op_i]
        nO = max(int(round(0.005 * len(Xo))), 1)
        Xop = np.r_[Xo, anomalies_random(nO, T, rng)]
        so = window_mse(Xop, be.recon(m, Xop))
        buf["evt_spot"] = (buf["evt_spot"] + list(so[so <= thr["evt_spot"]]))[-cfg.drift_buffer:]
        thr["evt_spot"] = evt_threshold(np.array(buf["evt_spot"]), cfg.t_init_pct, cfg.q)
        buf["evt_incl"] = (buf["evt_incl"] + list(so))[-cfg.drift_buffer:]
        thr["evt_incl"] = evt_threshold(np.array(buf["evt_incl"]), cfg.t_init_pct, cfg.q)
        buf["rolling99"] = (buf["rolling99"] + list(so))[-cfg.drift_buffer:]
        thr["rolling99"] = float(np.percentile(buf["rolling99"], 99))
        log(f"[drift] block {k + 1} ({phase}, p={pk:.3f}) AUC={row['auc_shift']:.4f} FPR fixed="
            f"{row['evt_fixed']['fpr']:.4f} incl={row['evt_incl']['fpr']:.4f} | recall incl={row['evt_incl']['recall']:.3f}")
    phases = {}
    for ph in ["before", "during", "after"]:
        rr = [r for r in rows if r["phase"] == ph]
        phases[ph] = {name: {mk: float(np.mean([r[name][mk] for r in rr]))
                             for mk in ["fpr", "recall", "precision", "f1", "mcc"]} for name in thr}
        phases[ph]["auc_shift"] = float(np.mean([r["auc_shift"] for r in rr]))
        phases[ph]["auc_noshift"] = float(np.mean([r["auc_noshift"] for r in rr]))
    res = dict(rows=rows, phases=phases, train_evt=par | {"z_q": z_fixed}, static99=s99,
               n_train=int(len(p.Xtr)), unit="block", n_blocks=n_blocks,
               shift_start=shift_start, shift_len=shift_len,
               windows_per_block=int(np.mean([len(b) for b in blocks])))
    jdump(res, path)
    return res


def stage_window_arrays(cfg, be):
    """Shorter window (T=5) built from the last five events of every saved window."""
    path = os.path.join(cfg.out, "stage_window.json")
    res = jload(path) if os.path.exists(path) and not cfg.force else {}
    if "5" in res:
        return res
    p = prepare(cfg)
    T = 5
    rng = np.random.default_rng(cfg.eval_seed + 5)
    Xtr, Xn = p.Xtr[:, -T:], p.Xn[:, -T:]
    A = anomalies_random(len(p.anomA), T, rng)
    be.set_seed(cfg.seeds[0])
    m = be.build("lstm", T)
    t0 = time.time(); be.fit(m, Xtr, cfg); tt = time.time() - t0
    s_tr = window_mse(Xtr, be.recon(m, Xtr))
    z = evt_threshold(s_tr, cfg.t_init_pct, cfg.q)
    X = np.r_[Xn, A]
    y = np.r_[np.zeros(len(Xn)), np.ones(len(A))].astype(int)
    res["5"] = metrics(y, window_mse(X, be.recon(m, X)), z) | {"train_seconds": tt}
    log(f"[window] T=5 AUC={res['5']['roc_auc']:.4f} F1={res['5']['f1']:.4f}")
    jdump(res, path)
    return res

# --------------------------------------------------------------------------- #
# Stage: main comparison (all models x seeds)
# --------------------------------------------------------------------------- #
def score_classical(name, p, cfg, seed):
    from sklearn.ensemble import IsolationForest
    from sklearn.svm import OneClassSVM
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(p.Xtr), min(cfg.classical_subset, len(p.Xtr)), replace=False)
    Xs = p.Xtr[idx].astype(np.float64)
    if name == "iforest":
        m = IsolationForest(n_estimators=100, contamination=0.01, random_state=seed, n_jobs=-1)
    else:
        m = OneClassSVM(kernel="rbf", gamma="scale", nu=0.01)
    t0 = time.time(); m.fit(Xs); fit_s = time.time() - t0
    sc = lambda X: -m.decision_function(X.astype(np.float64))
    return m, sc(Xs), sc, fit_s


def stage_main(cfg, be):
    path = os.path.join(cfg.out, "stage_main.json")
    res = jload(path) if os.path.exists(path) and not cfg.force else {"runs": {}}
    p = prepare(cfg)
    res["meta"] = p.meta
    y = np.r_[np.zeros(len(p.Xn)), np.ones(len(p.anomA))].astype(int)
    Xev = np.r_[p.Xn, p.anomA]
    groups = np.r_[p.un, -np.arange(1, len(p.anomA) + 1)]
    os.makedirs(os.path.join(cfg.out, "models"), exist_ok=True)
    os.makedirs(os.path.join(cfg.out, "scores"), exist_ok=True)
    for seed in cfg.seeds:
        for name in ALL_MODELS:
            key = f"{name}_{seed}"
            if key in res["runs"]:
                continue
            log(f"[main] {name} seed={seed}")
            run = {"model": name, "seed": seed}
            if name in NN_MODELS:
                be.set_seed(seed)
                mpath = os.path.join(cfg.out, "models", key)
                m = be.build(name, p.T)
                t0 = time.time()
                hist = be.fit(m, p.Xtr, cfg)
                run["train_seconds"] = time.time() - t0
                run["epochs_run"] = len(hist["loss"])
                run["history"] = hist
                run["n_params"] = be.n_params(m)
                be.save(m, mpath)
                s_tr = window_mse(p.Xtr, be.recon(m, p.Xtr))
                s_ev = window_mse(Xev, be.recon(m, Xev))
                extra = {
                    "orig": window_mse(p.Xo, be.recon(m, p.Xo)),
                    "nonoverlap": window_mse(p.Xno, be.recon(m, p.Xno)),
                    "B": window_mse(p.anomB, be.recon(m, p.anomB)),
                    "C": window_mse(p.anomC, be.recon(m, p.anomC)),
                } if name == "lstm" else {}
            else:
                m, s_tr, sc, fit_s = score_classical(name, p, cfg, seed)
                run["train_seconds"] = fit_s
                s_ev = sc(Xev)
                extra = {}
            z, params = evt_threshold(s_tr, cfg.t_init_pct, cfg.q, return_params=True)
            s99 = float(np.percentile(s_tr, 99))
            run["evt"] = params | {"z_q": z}
            run["static99"] = s99
            run["metrics_evt"] = metrics(y, s_ev, z)
            run["metrics_s99"] = metrics(y, s_ev, s99)
            if name == "lstm":
                yo = np.r_[np.zeros(len(p.Xo)), np.ones(len(p.anomA))]
                so = np.r_[extra["orig"], s_ev[len(p.Xn):]]
                run["metrics_orig_protocol"] = metrics(yo, so, z)
                yno = np.r_[np.zeros(len(p.Xno)), np.ones(len(p.anomA))]
                sno = np.r_[extra["nonoverlap"], s_ev[len(p.Xn):]]
                run["metrics_nonoverlap"] = metrics(yno, sno, z)
                for tag in ("B", "C"):
                    ya = np.r_[np.zeros(len(p.Xn)), np.ones(len(extra[tag]))]
                    sa = np.r_[s_ev[: len(p.Xn)], extra[tag]]
                    run[f"metrics_anom{tag}"] = metrics(ya, sa, z)
                if seed == cfg.seeds[0]:
                    rng = np.random.default_rng(seed)
                    run["bootstrap_evt"] = cluster_bootstrap(y, s_ev, z, groups, cfg.n_boot, rng)
            np.savez_compressed(os.path.join(cfg.out, "scores", key + ".npz"),
                                s_ev=s_ev.astype(np.float32), s_tr_sample=s_tr[:50000].astype(np.float32))
            res["runs"][key] = run
            jdump(res, path)
    jdump(res, path)
    return res


# --------------------------------------------------------------------------- #
# Stage: EVT sensitivity and goodness of fit
# --------------------------------------------------------------------------- #
def stage_sens(cfg, be):
    from scipy import stats
    path = os.path.join(cfg.out, "stage_sens.json")
    if os.path.exists(path) and not cfg.force:
        return jload(path)
    p = prepare(cfg)
    seed = cfg.seeds[0]
    m = be.load(os.path.join(cfg.out, "models", f"lstm_{seed}"))
    s_tr = window_mse(p.Xtr, be.recon(m, p.Xtr))
    Xev = np.r_[p.Xn, p.anomA]
    y = np.r_[np.zeros(len(p.Xn)), np.ones(len(p.anomA))].astype(int)
    s_ev = window_mse(Xev, be.recon(m, Xev))
    grid = []
    for tp in [80, 85, 90, 95, 97.5]:
        for q in [1e-4, 1e-3, 5e-3, 1e-2, 2e-2, 5e-2]:
            z, par = evt_threshold(s_tr, tp, q, return_params=True)
            if np.isnan(z):
                continue
            me = metrics(y, s_ev, z)
            emp = float(np.quantile(s_tr, 1 - q))
            mm = metrics(y, s_ev, emp)
            grid.append(dict(t_init_pct=tp, q=q, z_q=z, xi=par["xi"], beta=par["beta"],
                             evt_f1=me["f1"], evt_mcc=me["mcc"], evt_fpr=me["fpr"],
                             evt_recall=me["recall"], evt_precision=me["precision"],
                             emp_threshold=emp, emp_f1=mm["f1"], emp_mcc=mm["mcc"],
                             emp_fpr=mm["fpr"], emp_recall=mm["recall"]))
    # goodness of fit at the default initial threshold
    t0 = np.percentile(s_tr, cfg.t_init_pct)
    peaks = s_tr[s_tr > t0] - t0
    xi, _, beta = stats.genpareto.fit(peaks, floc=0)
    rng = np.random.default_rng(seed)
    sub = rng.choice(peaks, min(5000, len(peaks)), replace=False)
    ks = stats.kstest(sub, "genpareto", args=(xi, 0, beta))
    n_unique = int(len(np.unique(np.round(peaks, 8))))
    try:
        sub_ad = rng.choice(peaks, min(1000, len(peaks)), replace=False)
        ad = stats.goodness_of_fit(stats.genpareto, sub_ad, known_params={"loc": 0},
                                   statistic="ad", n_mc_samples=99, random_state=1)
        ad_res = dict(statistic=float(ad.statistic), pvalue=float(ad.pvalue))
    except Exception as e:  # pragma: no cover
        ad_res = dict(error=str(e))
    # bootstrap uncertainty of z_q
    zs = []
    for _ in range(100 if not cfg.quick else 20):
        bs = rng.choice(s_tr, len(s_tr), replace=True)
        zs.append(evt_threshold(bs, cfg.t_init_pct, cfg.q))
    qq_p = np.linspace(0.01, 0.99, 99)
    qq = dict(theoretical=stats.genpareto.ppf(qq_p, xi, 0, beta).tolist(),
              empirical=np.quantile(peaks, qq_p).tolist())
    mrl_u = np.quantile(s_tr, np.linspace(0.80, 0.995, 30))
    mrl = dict(u=mrl_u.tolist(), mean_excess=[float((s_tr[s_tr > u] - u).mean()) for u in mrl_u])
    res = dict(grid=grid, gof=dict(xi=float(xi), beta=float(beta), n_exceed=int(len(peaks)),
                                   n_unique_exceed=n_unique, ks_stat=float(ks.statistic),
                                   ks_pvalue=float(ks.pvalue), ad=ad_res,
                                   z_boot_ci=[float(np.percentile(zs, 2.5)), float(np.percentile(zs, 97.5))],
                                   z_point=evt_threshold(s_tr, cfg.t_init_pct, cfg.q)),
               qq=qq, mrl=mrl)
    jdump(res, path)
    return res


# --------------------------------------------------------------------------- #
# Stage: SEL localization accuracy and fidelity
# --------------------------------------------------------------------------- #
def stage_sel(cfg, be):
    path = os.path.join(cfg.out, "stage_sel.json")
    if os.path.exists(path) and not cfg.force:
        return jload(path)
    p = prepare(cfg)
    seed = cfg.seeds[0]
    m = be.load(os.path.join(cfg.out, "models", f"lstm_{seed}"))
    main = jload(os.path.join(cfg.out, "stage_main.json"))
    z = main["runs"][f"lstm_{seed}"]["evt"]["z_q"]
    X, pos = p.anomB, p.posB
    n, T = X.shape
    rng = np.random.default_rng(seed)
    out = {}

    def topk(scores, k):
        order = np.argsort(-scores, axis=1)[:, :k]
        return float((order == pos[:, None]).any(1).mean())

    t0 = time.perf_counter(); R = be.recon(m, X); t_fwd = time.perf_counter() - t0
    t0 = time.perf_counter(); E = step_errors(X, R); t_sel = time.perf_counter() - t0
    methods = {"SEL (timestep reconstruction error)": (E, t_sel / n)}
    # random
    methods["Random position"] = (rng.random((n, T)), 0.0)
    # bigram surprisal (non-neural, uses training statistics only)
    logP = np.log(p.P)
    t0 = time.perf_counter()
    S = np.zeros((n, T))
    S[:, 1:] += -logP[X[:, :-1], X[:, 1:]]
    S[:, :-1] += -logP[X[:, :-1], X[:, 1:]]
    methods["Bigram surprisal"] = (S, (time.perf_counter() - t0) / n)
    # occlusion: replace each step with the most frequent training token
    mode_tok = int(np.argmax(p.uni))
    t0 = time.perf_counter()
    base = window_mse(X, R)
    occ = np.zeros((n, T))
    for t in range(T):
        Xo = X.copy(); Xo[:, t] = mode_tok
        occ[:, t] = base - window_mse(Xo, be.recon(m, Xo))
    methods["Occlusion"] = (occ, (time.perf_counter() - t0) / n)
    # gradient saliency
    t0 = time.perf_counter()
    G = np.concatenate([be.saliency(m, X[i:i + 512]) for i in range(0, n, 512)])
    methods["Gradient saliency"] = (G, (time.perf_counter() - t0) / n)
    table = []
    for k_, (sc, per) in methods.items():
        table.append(dict(method=k_, top1=topk(sc, 1), top3=topk(sc, 3),
                          seconds_per_window=per))
    out["localization"] = table
    out["forward_seconds_per_window_batched"] = t_fwd / n
    # fidelity: repair the flagged step with the rounded reconstruction
    flagged = np.flatnonzero(base > z)
    out["n_windows"] = int(n)
    out["n_flagged"] = int(len(flagged))
    out["detection_rate_single_event"] = float(len(flagged) / n)
    fid = {}
    for k_, (sc, _) in methods.items():
        if len(flagged) == 0:
            break
        Xr = X[flagged].copy()
        tsel = np.argmax(sc[flagged], 1)
        rec = np.clip(np.rint(R[flagged, tsel]), 0, 2).astype(np.int8)
        same = rec == Xr[np.arange(len(flagged)), tsel]
        rec[same] = np.where(Xr[np.arange(len(flagged)), tsel][same] == 2, 1, 2)
        Xr[np.arange(len(flagged)), tsel] = rec
        fid[k_] = float((window_mse(Xr, be.recon(m, Xr)) <= z).mean())
    out["fidelity_repair_success"] = fid
    # positional accuracy profile for SEL
    sel_hit = (np.argmax(E, 1) == pos)
    out["sel_top1_by_position"] = [float(sel_hit[pos == t].mean()) if (pos == t).any() else None
                                   for t in range(T)]
    # one illustrative example (real model output)
    i_ex = int(flagged[0]) if len(flagged) else 0
    out["example"] = dict(tokens=X[i_ex].tolist(), errors=E[i_ex].tolist(),
                          true_pos=int(pos[i_ex]), benign_tokens=p.Xsb[i_ex].tolist(),
                          benign_errors=step_errors(p.Xsb[i_ex:i_ex + 1],
                                                    be.recon(m, p.Xsb[i_ex:i_ex + 1]))[0].tolist(),
                          z_q=z)
    jdump(out, path)
    return out


# --------------------------------------------------------------------------- #
# Stage: controlled chronological drift
# --------------------------------------------------------------------------- #
def month_index(ts, t0):
    d = pd.DatetimeIndex(pd.to_datetime(np.asarray(ts), unit="s"))
    d0 = pd.to_datetime(t0, unit="s")
    return np.asarray((d.year - d0.year) * 12 + (d.month - d0.month))


def stage_drift(cfg, be):
    if cfg.arrays_dir:
        return stage_drift_arrays(cfg, be)
    path = os.path.join(cfg.out, "stage_drift.json")
    if os.path.exists(path) and not cfg.force:
        return jload(path)
    p = prepare(cfg, need_ev=True)
    ev, T = p.ev, p.T
    t_start = int(ev.ts.min())
    d0 = pd.to_datetime(t_start, unit="s")
    month_start = lambda k: int((pd.Timestamp(year=d0.year, month=d0.month, day=1)
                                 + pd.DateOffset(months=k)).timestamp())
    n_months = int(month_index(np.array([ev.ts.max()]), t_start)[0]) + 1
    rng = np.random.default_rng(cfg.eval_seed + 1)
    # training: training users, windows ending in the first k months
    lo, hi = month_start(0), month_start(cfg.drift_train_months)
    Xtr, _, _ = stratified_windows(ev, p.tr_users, T, cfg.n_train, rng, t_lo=lo, t_hi=hi)
    be.set_seed(cfg.seeds[0])
    m = be.build("lstm", T)
    hist = be.fit(m, Xtr, cfg)
    s_tr = window_mse(Xtr, be.recon(m, Xtr))
    z_fixed, par = evt_threshold(s_tr, cfg.t_init_pct, cfg.q, return_params=True)
    s99 = float(np.percentile(s_tr, 99))
    seed_buf = list(s_tr[-cfg.drift_buffer:])
    buf = {"evt_spot": list(seed_buf), "evt_incl": list(seed_buf), "rolling99": list(seed_buf)}
    thr = {"static99": s99, "evt_fixed": z_fixed, "evt_spot": z_fixed, "evt_incl": z_fixed,
           "rolling99": s99}
    from sklearn.metrics import roc_auc_score
    rows = []
    for k in range(cfg.drift_train_months, n_months):
        lo, hi = month_start(k), month_start(k + 1)
        if k < cfg.drift_shift_start:
            pk = 0.0; phase = "before"
        elif k < cfg.drift_shift_start + cfg.drift_shift_len:
            pk = cfg.drift_p_max * (k - cfg.drift_shift_start + 1) / cfg.drift_shift_len; phase = "during"
        else:
            pk = cfg.drift_p_max; phase = "after"
        Xb, _, ub, sb = stratified_windows(ev, p.te_users, T, 2 * cfg.drift_normals_per_month, rng,
                                           t_lo=lo, t_hi=hi, return_starts=True)
        if len(Xb) < 200:
            continue
        perm = rng.permutation(len(Xb)); half = len(Xb) // 2
        ev_i, op_i = perm[:half], perm[half:]
        ends = ev.offset[ub] + sb + T - 1
        Xs = apply_reauth_shift(ev, ub, ends, T, pk, rng) if pk > 0 else Xb
        # evaluation sample: 10% injected anomalies
        Xe = Xs[ev_i]
        nA = max(int(round(len(Xe) / 9)), 1)
        A = anomalies_random(nA, T, rng)
        X = np.r_[Xe, A]
        y = np.r_[np.zeros(len(Xe)), np.ones(nA)].astype(int)
        s = window_mse(X, be.recon(m, X))
        s_noshift = window_mse(np.r_[Xb[ev_i], A], be.recon(m, np.r_[Xb[ev_i], A]))
        row = dict(month=k, phase=phase, p_shift=pk, n_normal=int(len(Xe)), n_anom=nA,
                   auc_shift=float(roc_auc_score(y, s)), auc_noshift=float(roc_auc_score(y, s_noshift)))
        for name, z in thr.items():
            mt = metrics(y, s, z)
            row[name] = dict(threshold=z, fpr=mt["fpr"], recall=mt["recall"],
                             precision=mt["precision"], f1=mt["f1"], mcc=mt["mcc"])
        rows.append(row)
        # unlabelled operational stream for the month: same benign shift, 0.5% anomalies
        Xo = Xs[op_i]
        nO = max(int(round(0.005 * len(Xo))), 1)
        so = window_mse(np.r_[Xo, anomalies_random(nO, T, rng)],
                        be.recon(m, np.r_[Xo, anomalies_random(nO, T, rng)]))
        # SPOT-style: windows flagged by the threshold in force are treated as alarms
        buf["evt_spot"] = (buf["evt_spot"] + list(so[so <= thr["evt_spot"]]))[-cfg.drift_buffer:]
        thr["evt_spot"] = evt_threshold(np.array(buf["evt_spot"]), cfg.t_init_pct, cfg.q)
        # inclusive: every unlabelled window of the month enters the buffer
        buf["evt_incl"] = (buf["evt_incl"] + list(so))[-cfg.drift_buffer:]
        thr["evt_incl"] = evt_threshold(np.array(buf["evt_incl"]), cfg.t_init_pct, cfg.q)
        buf["rolling99"] = (buf["rolling99"] + list(so))[-cfg.drift_buffer:]
        thr["rolling99"] = float(np.percentile(buf["rolling99"], 99))
        log(f"[drift] month {k} ({phase}, p={pk:.3f}) AUC={row['auc_shift']:.4f} FPR fixed="
            f"{row['evt_fixed']['fpr']:.4f} spot={row['evt_spot']['fpr']:.4f} incl={row['evt_incl']['fpr']:.4f}"
            f" | recall fixed={row['evt_fixed']['recall']:.3f} incl={row['evt_incl']['recall']:.3f}")
    phases = {}
    for ph in ["before", "during", "after"]:
        rr = [r for r in rows if r["phase"] == ph]
        if not rr:
            continue
        phases[ph] = {}
        for name in thr:
            phases[ph][name] = {mk: float(np.mean([r[name][mk] for r in rr]))
                                for mk in ["fpr", "recall", "precision", "f1", "mcc"]}
        phases[ph]["auc_shift"] = float(np.mean([r["auc_shift"] for r in rr]))
        phases[ph]["auc_noshift"] = float(np.mean([r["auc_noshift"] for r in rr]))
    res = dict(rows=rows, phases=phases, train_evt=par | {"z_q": z_fixed}, static99=s99,
               history=hist, n_train=int(len(Xtr)), n_months=n_months)
    jdump(res, path)
    return res


# --------------------------------------------------------------------------- #
# Stage: latency breakdown
# --------------------------------------------------------------------------- #
class _CtrlServer(ThreadingHTTPServer):
    request_queue_size = 256
    daemon_threads = True


class _Ctrl(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    disable_nagle_algorithm = True

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n))
        ack = json.dumps({"status": "quarantine_scheduled", "user": body.get("user")}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(ack)))
        self.end_headers()
        self.wfile.write(ack)

    def log_message(self, *a):
        pass


def _pct(a):
    a = np.asarray(a) * 1000.0
    return dict(median_ms=float(np.median(a)), p95_ms=float(np.percentile(a, 95)),
                p99_ms=float(np.percentile(a, 99)), mean_ms=float(np.mean(a)))


def stage_latency(cfg, be):
    import http.client
    from concurrent.futures import ThreadPoolExecutor
    path = os.path.join(cfg.out, "stage_latency.json")
    if os.path.exists(path) and not cfg.force:
        return jload(path)
    p = prepare(cfg)
    seed = cfg.seeds[0]
    m = be.load(os.path.join(cfg.out, "models", f"lstm_{seed}"))
    main = jload(os.path.join(cfg.out, "stage_main.json"))
    z = main["runs"][f"lstm_{seed}"]["evt"]["z_q"]
    T = p.T
    fns = be.single_forward_fns(m, T)
    X = np.r_[p.Xn[:500], p.anomA[:500]].astype(np.float32)[..., None]
    res = {"hardware": hardware_info(), "single_window": {}, "batched": {}}
    reps = 300 if cfg.quick else 1000
    for name, fn in fns.items():
        r = reps if name != "keras_predict" else min(200, reps)
        for i in range(20):
            fn(X[i:i + 1])
        ts = []
        for i in range(r):
            x = X[i % len(X): i % len(X) + 1]
            t0 = time.perf_counter(); fn(x); ts.append(time.perf_counter() - t0)
        res["single_window"][name] = _pct(ts)
    g = fns["tf_function"]
    for B in [1, 8, 64, 512, 4096]:
        xb = np.repeat(X, int(np.ceil(B / len(X))), axis=0)[:B]
        g(xb)
        ts = []
        for _ in range(30 if B <= 512 else 10):
            t0 = time.perf_counter(); g(xb); ts.append(time.perf_counter() - t0)
        d = _pct(ts); d["per_window_ms"] = d["median_ms"] / B
        res["batched"][str(B)] = d
    # SEL and threshold comparison for one window
    R1 = g(X[:1])[..., 0]
    ts = []
    for _ in range(5000):
        t0 = time.perf_counter()
        e = (X[0, :, 0] - R1[0]) ** 2
        flag = e.mean() > z
        tmax = int(np.argmax(e))
        ts.append(time.perf_counter() - t0)
    res["sel_and_compare"] = _pct(ts)
    # EVT refit cost
    s_tr = window_mse(p.Xtr, be.recon(m, p.Xtr))
    res["evt_refit"] = {}
    for nb in [10_000, 50_000, 100_000]:
        buf = s_tr[:nb]
        ts = []
        for _ in range(10):
            t0 = time.perf_counter(); evt_threshold(buf, cfg.t_init_pct, cfg.q); ts.append(time.perf_counter() - t0)
        res["evt_refit"][str(nb)] = _pct(ts)
    # REST controller round trip
    srv = _CtrlServer(("127.0.0.1", 0), _Ctrl)
    port = srv.server_address[1]
    th = threading.Thread(target=srv.serve_forever, daemon=True); th.start()

    def payload(i, tmax, err):
        return json.dumps({"user": f"U{i:05d}", "mac": "00:00:00:00:00:00", "ip": "10.0.0.1",
                           "event": TOKEN_NAMES[int(round(float(X[i % len(X), tmax, 0])))],
                           "timestep": tmax, "window_error": float(err),
                           "severity": float(err / z), "action": "quarantine_vlan"}).encode()

    def connect():
        import socket
        c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
        c.connect()
        c.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        return c

    def worker(nreq, wid):
        conn = connect()
        lat = []
        for j in range(nreq):
            body = payload(wid * 1000 + j, 3, 0.5)
            t0 = time.perf_counter()
            conn.request("POST", "/quarantine", body, {"Content-Type": "application/json"})
            r = conn.getresponse(); r.read()
            lat.append(time.perf_counter() - t0)
        conn.close()
        return lat

    res["rest_roundtrip"] = {}
    for conc in [1, 10, 50]:
        nreq = 200 if conc == 1 else max(20, 2000 // conc)
        with ThreadPoolExecutor(conc) as ex:
            lats = sum(ex.map(worker, [nreq] * conc, range(conc)), [])
        res["rest_roundtrip"][str(conc)] = _pct(lats)
    # end-to-end loop (window ready -> controller acknowledgement), batch size 1
    conn = connect()
    comp = {"inference": [], "score_sel": [], "serialize": [], "rest": [], "total": []}
    for i in range(300 if cfg.quick else 1000):
        x = X[i % len(X): i % len(X) + 1]
        t0 = time.perf_counter()
        r = g(x)[..., 0]
        t1 = time.perf_counter()
        e = (x[0, :, 0] - r[0]) ** 2; err = e.mean(); tmax = int(np.argmax(e))
        t2 = time.perf_counter()
        body = payload(i, tmax, err)
        t3 = time.perf_counter()
        conn.request("POST", "/quarantine", body, {"Content-Type": "application/json"})
        rr = conn.getresponse(); rr.read()
        t4 = time.perf_counter()
        comp["inference"].append(t1 - t0); comp["score_sel"].append(t2 - t1)
        comp["serialize"].append(t3 - t2); comp["rest"].append(t4 - t3); comp["total"].append(t4 - t0)
    conn.close()
    res["end_to_end_single"] = {k: _pct(v) for k, v in comp.items()}
    # concurrent users: one batched inference for N windows, then N concurrent POSTs
    res["end_to_end_concurrent"] = {}
    for N in [10, 100, 500]:
        tot = []
        for rep in range(5):
            xb = np.repeat(X, int(np.ceil(N / len(X))), axis=0)[:N]
            t0 = time.perf_counter()
            rb = g(xb)[..., 0]
            e = (xb[..., 0] - rb) ** 2
            tm = np.argmax(e, 1); er = e.mean(1)
            with ThreadPoolExecutor(min(N, 50)) as ex:
                list(ex.map(lambda k: worker(1, k), range(N)))
            tot.append(time.perf_counter() - t0)
        d = _pct(tot); d["users"] = N
        res["end_to_end_concurrent"][str(N)] = d
    srv.shutdown()
    res["not_measured"] = ("OpenFlow flow-rule installation, VLAN reassignment on physical or "
                           "emulated switches, and controller-to-switch propagation were not measured.")
    jdump(res, path)
    return res


def hardware_info():
    info = dict(platform=platform.platform(), processor=platform.processor(),
                machine=platform.machine(), cpu_count=os.cpu_count(),
                python=sys.version.split()[0])
    try:
        import psutil
        info["ram_gb"] = round(psutil.virtual_memory().total / 2**30, 1)
    except Exception:
        pass
    for mod in ["numpy", "pandas", "scipy", "sklearn", "tensorflow", "keras", "matplotlib"]:
        try:
            info[mod] = __import__(mod).__version__
        except Exception:
            pass
    return info


# --------------------------------------------------------------------------- #
# Stage: training contamination and window size
# --------------------------------------------------------------------------- #
def stage_contam(cfg, be):
    path = os.path.join(cfg.out, "stage_contam.json")
    res = jload(path) if os.path.exists(path) and not cfg.force else {}
    p = prepare(cfg)
    seed = cfg.seeds[0]
    Xev = np.r_[p.Xn, p.anomA]
    y = np.r_[np.zeros(len(p.Xn)), np.ones(len(p.anomA))].astype(int)
    for rate in cfg.contam_rates:
        key = str(rate)
        if key in res:
            continue
        rng = np.random.default_rng(seed + 7)
        Xtr = p.Xtr.copy()
        k = int(rate * len(Xtr))
        idx = rng.choice(len(Xtr), k, replace=False)
        Xtr[idx] = anomalies_random(k, p.T, rng)
        be.set_seed(seed)
        m = be.build("lstm", p.T)
        be.fit(m, Xtr, cfg)
        s_tr = window_mse(Xtr, be.recon(m, Xtr))
        z = evt_threshold(s_tr, cfg.t_init_pct, cfg.q)
        s_ev = window_mse(Xev, be.recon(m, Xev))
        res[key] = metrics(y, s_ev, z)
        log(f"[contam] rate={rate} AUC={res[key]['roc_auc']:.4f} F1={res[key]['f1']:.4f}")
        jdump(res, path)
    return res


def stage_window(cfg, be):
    if cfg.arrays_dir:
        return stage_window_arrays(cfg, be)
    path = os.path.join(cfg.out, "stage_window.json")
    res = jload(path) if os.path.exists(path) and not cfg.force else {}
    seed = cfg.seeds[0]
    for T in cfg.window_sizes:
        key = str(T)
        if key in res:
            continue
        p = prepare(cfg, T=T)
        be.set_seed(seed)
        m = be.build("lstm", T)
        t0 = time.time(); be.fit(m, p.Xtr, cfg); tt = time.time() - t0
        s_tr = window_mse(p.Xtr, be.recon(m, p.Xtr))
        z = evt_threshold(s_tr, cfg.t_init_pct, cfg.q)
        Xev = np.r_[p.Xn, p.anomA]
        y = np.r_[np.zeros(len(p.Xn)), np.ones(len(p.anomA))].astype(int)
        s_ev = window_mse(Xev, be.recon(m, Xev))
        res[key] = metrics(y, s_ev, z) | {"train_seconds": tt}
        log(f"[window] T={T} AUC={res[key]['roc_auc']:.4f} F1={res[key]['f1']:.4f}")
        jdump(res, path)
    return res



def stage_holdout(cfg, be):
    """Benign diversity check: hold out 20% of the saved training windows (many users,
    not user-disjoint), train on the rest, and measure false alarms on the held-out part."""
    path = os.path.join(cfg.out, "stage_holdout.json")
    if os.path.exists(path) and not cfg.force:
        return jload(path)
    p = prepare(cfg)
    rng = np.random.default_rng(cfg.seeds[0] + 11)
    perm = rng.permutation(len(p.Xtr)); cut = int(0.8 * len(perm))
    Xa, Xb = p.Xtr[perm[:cut]], p.Xtr[perm[cut:]]
    n_eval = min(9000, len(Xb))
    Xb = Xb[rng.choice(len(Xb), n_eval, replace=False)]
    be.set_seed(cfg.seeds[0])
    m = be.build("lstm", p.T)
    be.fit(m, Xa, cfg)
    s_tr = window_mse(Xa, be.recon(m, Xa))
    z = evt_threshold(s_tr, cfg.t_init_pct, cfg.q)
    s99 = float(np.percentile(s_tr, 99))
    X = np.r_[Xb, p.anomA]
    y = np.r_[np.zeros(len(Xb)), np.ones(len(p.anomA))].astype(int)
    s = window_mse(X, be.recon(m, X))
    res = dict(n_train=int(len(Xa)), n_benign=int(len(Xb)),
               n_unique_benign=int(len(np.unique(Xb, axis=0))),
               share_all_http=float((Xb == 2).all(1).mean()),
               evt=metrics(y, s, z), s99=metrics(y, s, s99))
    log(f"[holdout] AUC={res['evt']['roc_auc']:.4f} FPR(EVT)={res['evt']['fpr']:.4f} "
        f"unique benign windows={res['n_unique_benign']}")
    jdump(res, path)
    return res


# --------------------------------------------------------------------------- #
# Report: tables, figures, placeholders
# --------------------------------------------------------------------------- #
def f4(x):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.4f}"


def ms(vals, nd=4):
    a = np.asarray(vals, dtype=float)
    if len(a) == 1:
        return f"{a[0]:.{nd}f}"
    return f"{a.mean():.{nd}f} ± {a.std(ddof=1):.{nd}f}"


def fmt_ms_value(x):
    if x < 0.001:
        return "<0.001"
    if x < 0.01:
        return f"{x:.4f}"
    if x >= 100:
        return f"{x:.0f}"
    if x >= 1:
        return f"{x:.2f}"
    return f"{x:.3f}"


PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]
LINESTYLES = ["-", "--", "-.", ":", (0, (5, 1)), (0, (3, 1, 1, 1)), (0, (1, 1))]


def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.color": "#e6e6e3", "grid.linewidth": 0.6,
                         "axes.edgecolor": "#888888", "lines.linewidth": 1.6,
                         "savefig.bbox": "tight", "pdf.fonttype": 42})
    return plt


def _save(fig, out, name):
    figdir = os.path.join(out, "figures"); os.makedirs(figdir, exist_ok=True)
    fig.savefig(os.path.join(figdir, name + ".pdf"))
    fig.savefig(os.path.join(figdir, name + ".png"), dpi=600)


def report(cfg):
    plt = _style()
    out = cfg.out
    ph = {}
    tables = {}
    main = jload(os.path.join(out, "stage_main.json"))
    meta = main["meta"]
    fmt_int = lambda v: "n/a" if v is None else f"{v:,}"
    ph.update({
        "DATA.SOURCE": meta.get("source", "raw CERT r4.2 logs"),
        "DATA.N_EVENTS": fmt_int(meta.get("n_events")), "DATA.N_USERS": fmt_int(meta["n_users"]),
        "DATA.N_TRAIN_USERS": str(meta["n_train_users"]), "DATA.N_TEST_USERS": str(meta["n_test_users"]),
        "DATA.N_TRAIN_WINDOWS_TOTAL": fmt_int(meta.get("n_train_windows_total")),
        "DATA.N_TRAIN_SAMPLED": f"{meta['n_train_sampled']:,}",
        "DATA.N_TEST_NORMAL": f"{meta['n_test_normal']:,}",
        "DATA.N_TEST_USERS_REPRESENTED": fmt_int(meta.get("n_test_users_represented")),
        "DATA.N_ORIG_NORMAL_USERS": str(meta["n_orig_normal_users"]),
        "DATA.N_BENIGN_SEGMENTS": fmt_int(meta.get("n_benign_segments")),
        "DATA.N_ANOM": f"{meta['n_anom']:,}",
        "DATA.DATE_START": str(meta.get("date_start")), "DATA.DATE_END": str(meta.get("date_end")),
        "DATA.SHARE_HTTP": f"{100 * meta['token_share']['HTTP']:.1f}%",
        "DATA.SHARE_LOGON": f"{100 * meta['token_share']['Logon']:.1f}%",
        "DATA.SHARE_LOGOFF": f"{100 * meta['token_share']['Logoff']:.1f}%",
        "DATA.NORMAL_ALL_HTTP": f"{100 * meta['normal_share_all_http']:.1f}%",
        "DATA.RARE_BIGRAM": meta["rare_bigram"],
        "DATA.RARE_BIGRAM_SHARE": f"{100 * meta['rare_bigram_share']:.4f}%",
    })
    if "n_unique_benign" in meta:
        ph["DATA.N_UNIQUE_BENIGN"] = f"{meta['n_unique_benign']:,}"
    if "anomA_only_logoff_http" in meta:
        ph["DATA.ANOM_ONLY_LOGOFF_HTTP"] = f"{100 * meta['anomA_only_logoff_http']:.1f}%"
    runs = main["runs"]
    seeds = sorted({r["seed"] for r in runs.values()})
    ph["RUN.N_SEEDS"] = str(len(seeds))
    cfg_saved = jload(os.path.join(out, "config.json"))
    ph["DATA.CLASSICAL_SUBSET"] = f"{cfg_saved['classical_subset']:,}"
    ph["RUN.SEEDS"] = ", ".join(str(s) for s in seeds)
    # --- main table
    rows = [["Model", "Params", "ROC-AUC", "PR-AUC", "Precision", "Recall", "F1", "MCC", "FPR",
             "F1 (static P99)", "MCC (static P99)"]]
    for name in ALL_MODELS:
        rr = [runs[f"{name}_{s}"] for s in seeds if f"{name}_{s}" in runs]
        if not rr:
            continue
        g = lambda key, sub="metrics_evt": [r[sub][key] for r in rr]
        K = name.upper()
        for mk, lab in [("roc_auc", "AUC"), ("pr_auc", "PRAUC"), ("precision", "P"),
                        ("recall", "R"), ("f1", "F1"), ("mcc", "MCC"), ("fpr", "FPR")]:
            ph[f"MAIN.{K}.{lab}"] = ms(g(mk))
        ph[f"MAIN.{K}.S99.F1"] = ms(g("f1", "metrics_s99"))
        ph[f"MAIN.{K}.S99.MCC"] = ms(g("mcc", "metrics_s99"))
        ph[f"MAIN.{K}.S99.FPR"] = ms(g("fpr", "metrics_s99"))
        ph[f"MAIN.{K}.S99.P"] = ms(g("precision", "metrics_s99"))
        params = rr[0].get("n_params")
        ph[f"MAIN.{K}.PARAMS"] = f"{params:,}" if params else "–"
        rows.append([MODEL_LABEL[name], ph[f"MAIN.{K}.PARAMS"], ph[f"MAIN.{K}.AUC"], ph[f"MAIN.{K}.PRAUC"],
                     ph[f"MAIN.{K}.P"], ph[f"MAIN.{K}.R"], ph[f"MAIN.{K}.F1"], ph[f"MAIN.{K}.MCC"],
                     ph[f"MAIN.{K}.FPR"], ph[f"MAIN.{K}.S99.F1"], ph[f"MAIN.{K}.S99.MCC"]])
    tables["main"] = rows
    L = [runs[f"lstm_{s}"] for s in seeds if f"lstm_{s}" in runs]
    ph["LSTM.ZQ"] = ms([r["evt"]["z_q"] for r in L])
    ph["LSTM.XI"] = ms([r["evt"]["xi"] for r in L], 3)
    ph["LSTM.BETA"] = ms([r["evt"]["beta"] for r in L])
    ph["LSTM.S99"] = ms([r["static99"] for r in L])
    ph["LSTM.NT"] = f"{L[0]['evt']['Nt']:,}"
    ph["LSTM.EPOCHS"] = ms([r["epochs_run"] for r in L], 1)
    ph["LSTM.TRAIN_MIN"] = ms([r["train_seconds"] / 60 for r in L], 1)
    ph["LSTM.TP"] = ms([r["metrics_evt"]["TP"] for r in L], 1)
    ph["LSTM.FP"] = ms([r["metrics_evt"]["FP"] for r in L], 1)
    ph["LSTM.TPR_AT_FPR001"] = ms([r["metrics_evt"].get("tpr_at_fpr_0.001", np.nan) for r in L])
    ph["LSTM.TPR_AT_FPR01"] = ms([r["metrics_evt"].get("tpr_at_fpr_0.01", np.nan) for r in L])
    b = L[0].get("bootstrap_evt", {})
    if b:
        ph["LSTM.AUC_CI"] = "{:.4f}–{:.4f}".format(*b["roc_auc_ci"])
        ph["LSTM.MCC_CI"] = "{:.4f}–{:.4f}".format(*b["mcc_ci"])
        ph["LSTM.F1_CI"] = "{:.4f}–{:.4f}".format(*b["f1_ci"])
    ph["REPRO.LSTM.AUC"] = ms([r["metrics_orig_protocol"]["roc_auc"] for r in L])
    ph["REPRO.LSTM.F1"] = ms([r["metrics_orig_protocol"]["f1"] for r in L])
    ph["REPRO.LSTM.MCC"] = ms([r["metrics_orig_protocol"]["mcc"] for r in L])
    ph["NONOVL.LSTM.AUC"] = ms([r["metrics_nonoverlap"]["roc_auc"] for r in L])
    ph["NONOVL.LSTM.F1"] = ms([r["metrics_nonoverlap"]["f1"] for r in L])
    ph["NONOVL.USERS"] = fmt_int(meta.get("n_nonoverlap_users"))
    # paired t-tests across seeds (LSTM vs others) on MCC and AUC
    from scipy.stats import ttest_rel
    tt_rows = [["Comparison", "ΔAUC (mean)", "p (AUC)", "ΔMCC (mean)", "p (MCC)"]]
    for name in ALL_MODELS[1:]:
        common = [s for s in seeds if f"{name}_{s}" in runs and f"lstm_{s}" in runs]
        if len(common) < 2:
            continue
        a = lambda n_, k: np.array([runs[f"{n_}_{s}"]["metrics_evt"][k] for s in common])
        da = a("lstm", "roc_auc") - a(name, "roc_auc"); dm = a("lstm", "mcc") - a(name, "mcc")
        pa = ttest_rel(a("lstm", "roc_auc"), a(name, "roc_auc")).pvalue
        pm = ttest_rel(a("lstm", "mcc"), a(name, "mcc")).pvalue
        tt_rows.append([f"LSTM-AE vs {MODEL_LABEL[name]}", f"{da.mean():+.4f}", f4(float(pa)),
                        f"{dm.mean():+.4f}", f4(float(pm))])
    tables["ttest"] = tt_rows
    # adversary / robustness table
    adv = [["Evaluation condition", "ROC-AUC", "Recall at EVT threshold", "F1", "MCC"]]
    for tag, lab in [("metrics_evt", "Random Logoff/HTTP sequences (main set)"),
                     ("metrics_anomB", "Single implausible event in a benign window"),
                     ("metrics_anomC", f"Markov-mimicry sequence with one {meta['rare_bigram']} transition"),
                     ("metrics_orig_protocol", "Original benign set (first windows in user order)"),
                     ("metrics_nonoverlap", "Non-overlapping benign windows (stride T)")]:
        if tag == "metrics_orig_protocol" and str(meta.get("source", "")).startswith("saved arrays"):
            continue  # identical to the main set when running from the saved arrays
        adv.append([lab, ms([r[tag]["roc_auc"] for r in L]), ms([r[tag]["recall"] for r in L]),
                    ms([r[tag]["f1"] for r in L]), ms([r[tag]["mcc"] for r in L])])
        short = {"metrics_anomB": "ADV.B", "metrics_anomC": "ADV.C"}.get(tag)
        if short:
            ph[f"{short}.AUC"] = ms([r[tag]["roc_auc"] for r in L])
            ph[f"{short}.RECALL"] = ms([r[tag]["recall"] for r in L])
    p_c = os.path.join(out, "stage_contam.json")
    if os.path.exists(p_c):
        c = jload(p_c)
        for rate, mt in c.items():
            adv.append([f"Training set contaminated with {100 * float(rate):.0f}% anomalies (seed {seeds[0]})",
                        f4(mt["roc_auc"]), f4(mt["recall"]), f4(mt["f1"]), f4(mt["mcc"])])
            ph[f"CONTAM.{int(round(100 * float(rate)))}.AUC"] = f4(mt["roc_auc"])
            ph[f"CONTAM.{int(round(100 * float(rate)))}.F1"] = f4(mt["f1"])
    p_w = os.path.join(out, "stage_window.json")
    if os.path.exists(p_w):
        w = jload(p_w)
        for T, mt in w.items():
            adv.append([f"Window length T = {T} (seed {seeds[0]})", f4(mt["roc_auc"]), f4(mt["recall"]),
                        f4(mt["f1"]), f4(mt["mcc"])])
            ph[f"WIN.{T}.AUC"] = f4(mt["roc_auc"]); ph[f"WIN.{T}.F1"] = f4(mt["f1"])
            ph[f"WIN.{T}.MCC"] = f4(mt["mcc"])
    p_h = os.path.join(out, "stage_holdout.json")
    if os.path.exists(p_h):
        h = jload(p_h)
        adv.append([f"Benign windows held out from the training pool (many users, not user-disjoint; seed {seeds[0]})",
                    f4(h["evt"]["roc_auc"]), f4(h["evt"]["recall"]), f4(h["evt"]["f1"]), f4(h["evt"]["mcc"])])
        ph.update({"HOLD.AUC": f4(h["evt"]["roc_auc"]), "HOLD.FPR": f4(h["evt"]["fpr"]),
                   "HOLD.F1": f4(h["evt"]["f1"]), "HOLD.MCC": f4(h["evt"]["mcc"]),
                   "HOLD.S99.FPR": f4(h["s99"]["fpr"]), "HOLD.NUNIQUE": f"{h['n_unique_benign']:,}",
                   "HOLD.N": f"{h['n_benign']:,}", "HOLD.ALLHTTP": f"{100 * h['share_all_http']:.1f}%"})
    tables["robustness"] = adv
    # --- figures from main
    scores_dir = os.path.join(out, "scores")
    y = np.r_[np.zeros(meta["n_test_normal"]), np.ones(meta["n_anom"])]
    from sklearn.metrics import roc_curve, confusion_matrix
    fig, ax = plt.subplots(figsize=(3.5, 3.2))
    for i, name in enumerate(ALL_MODELS):
        f_ = os.path.join(scores_dir, f"{name}_{seeds[0]}.npz")
        if not os.path.exists(f_):
            continue
        s = np.load(f_)["s_ev"]
        fpr, tpr, _ = roc_curve(y, s)
        ax.plot(fpr, tpr, color=PALETTE[i], linestyle=LINESTYLES[i],
                label=f"{MODEL_LABEL[name]} ({runs[f'{name}_{seeds[0]}']['metrics_evt']['roc_auc']:.4f})")
    ax.plot([0, 1], [0, 1], color="#bbbbbb", lw=0.8)
    ax.set_xscale("symlog", linthresh=1e-3); ax.set_xlim(0, 1)
    ax.set_xlabel("False positive rate (symlog scale)"); ax.set_ylabel("True positive rate")
    ax.legend(fontsize=6.5, loc="lower right", frameon=False)
    _save(fig, out, "fig_roc"); plt.close(fig)
    # confusion matrices at EVT threshold
    fig, axes = plt.subplots(2, 4, figsize=(7.2, 3.8))
    for ax, name in zip(axes.ravel(), ALL_MODELS):
        r = runs[f"{name}_{seeds[0]}"]["metrics_evt"]
        cm = np.array([[r["TN"], r["FP"]], [r["FN"], r["TP"]]])
        ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.sum() * 0.9)
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, f"{v:,}", ha="center", va="center", fontsize=8,
                    color="white" if v > cm.sum() * 0.45 else "#222222")
        ax.set_xticks([0, 1], ["Benign", "Anomaly"], fontsize=7)
        ax.set_yticks([0, 1], ["Benign", "Anomaly"], fontsize=7)
        ax.set_title(f"{MODEL_LABEL[name]}\nMCC {r['mcc']:.3f}", fontsize=7.5)
        ax.grid(False)
    axes.ravel()[-1].axis("off")
    fig.supxlabel("Predicted label", fontsize=8); fig.supylabel("True label", fontsize=8)
    fig.tight_layout(); _save(fig, out, "fig_confusion"); plt.close(fig)
    # LSTM loss curve
    h = runs[f"lstm_{seeds[0]}"]["history"]
    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    ep = np.arange(1, len(h["loss"]) + 1)
    ax.plot(ep, h["loss"], color=PALETTE[0], label="Training loss")
    ax.plot(ep, h["val_loss"], color=PALETTE[1], linestyle="--", label="Validation loss")
    ax.set_xlabel("Epoch"); ax.set_ylabel("MSE"); ax.legend(frameon=False)
    _save(fig, out, "fig_loss_lstm"); plt.close(fig)
    # error distribution with thresholds
    s = np.load(os.path.join(scores_dir, f"lstm_{seeds[0]}.npz"))["s_ev"]
    r0 = runs[f"lstm_{seeds[0]}"]
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    bins = np.logspace(np.log10(max(s.min(), 1e-6)), np.log10(s.max() + 1e-9), 60)
    ax.hist(s[y == 0], bins=bins, color=PALETTE[0], alpha=0.75, label="Benign windows")
    ax.hist(s[y == 1], bins=bins, color=PALETTE[1], alpha=0.75, label="Synthetic anomalies")
    ax.axvline(r0["evt"]["z_q"], color="#222222", lw=1.2, label=f"EVT $z_q$ = {r0['evt']['z_q']:.4f}")
    ax.axvline(r0["static99"], color="#222222", lw=1.0, ls=":", label=f"Static P99 = {r0['static99']:.4f}")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Window reconstruction error (MSE)"); ax.set_ylabel("Count")
    ax.legend(fontsize=6.5, frameon=False)
    _save(fig, out, "fig_error_distribution"); plt.close(fig)
    ph["FIG.ROC"] = "figures/fig_roc.png"
    # --- sensitivity
    p_s = os.path.join(out, "stage_sens.json")
    if os.path.exists(p_s):
        S = jload(p_s)
        gof = S["gof"]
        ph.update({"GOF.XI": f"{gof['xi']:.3f}", "GOF.BETA": f"{gof['beta']:.4f}",
                   "GOF.NT": f"{gof['n_exceed']:,}", "GOF.NUNIQUE": f"{gof['n_unique_exceed']:,}",
                   "GOF.KS": f"{gof['ks_stat']:.3f}", "GOF.KSP": f"{gof['ks_pvalue']:.3g}",
                   "GOF.AD": f"{gof['ad'].get('statistic', float('nan')):.3f}",
                   "GOF.ADP": f"{gof['ad'].get('pvalue', float('nan')):.3g}",
                   "GOF.ZCI": "{:.4f}–{:.4f}".format(*gof["z_boot_ci"])})
        grid = S["grid"]
        tps = sorted({g["t_init_pct"] for g in grid}); qs = sorted({g["q"] for g in grid})
        def cell(tp, q, k):
            hit = [g for g in grid if g["t_init_pct"] == tp and g["q"] == q]
            return hit[0][k] if hit else np.nan
        M = np.array([[cell(tp, q, "evt_mcc") for q in qs] for tp in tps])
        F = np.array([[cell(tp, q, "evt_fpr") for q in qs] for tp in tps])
        ph["SENS.MCC_RANGE"] = f"{np.nanmin(M):.3f}–{np.nanmax(M):.3f}"
        sub = M[[tps.index(t) for t in tps if 85 <= t <= 95]][:, [qs.index(q) for q in qs if 1e-3 <= q <= 2e-2]]
        ph["SENS.MCC_RANGE_CORE"] = f"{np.nanmin(sub):.3f}–{np.nanmax(sub):.3f}"
        fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5))
        for ax, A, title, cmap in [(axes[0], M, "MCC at $z_q$", "Blues"), (axes[1], F, "FPR at $z_q$", "Oranges")]:
            im = ax.imshow(A, cmap=cmap, aspect="auto")
            ax.set_xticks(range(len(qs)), [f"{q:g}" for q in qs], fontsize=6.5, rotation=45)
            ax.set_yticks(range(len(tps)), [f"{t:g}" for t in tps], fontsize=7)
            ax.set_xlabel("Risk level q"); ax.set_ylabel("Initial threshold percentile")
            vmax = np.nanmax(A) if np.isfinite(np.nanmax(A)) else 1
            for (i, j), v in np.ndenumerate(A):
                txt = "n/a" if np.isnan(v) else (f"{v:.2f}" if title.startswith("MCC") else f"{v:.3f}")
                ax.text(j, i, txt, ha="center", va="center", fontsize=5.5,
                        color="white" if (not np.isnan(v) and v > 0.6 * vmax) else "#222222")
            ax.set_title(title, fontsize=8); ax.grid(False)
        qq = S["qq"]
        axes[2].plot(qq["theoretical"], qq["empirical"], "o", ms=2.5, color=PALETTE[0])
        lim = [0, max(max(qq["theoretical"]), max(qq["empirical"]))]
        axes[2].plot(lim, lim, color="#888888", lw=0.8)
        axes[2].set_xlabel("GPD quantile"); axes[2].set_ylabel("Empirical exceedance quantile")
        axes[2].set_title("GPD quantile–quantile plot", fontsize=8)
        fig.tight_layout(); _save(fig, out, "fig_sensitivity"); plt.close(fig)
        sens_rows = [["t_init (pct)", "q", "z_q", "ξ", "β", "Precision", "Recall", "F1", "MCC", "FPR", "Empirical (1−q) F1"]]
        for g_ in grid:
            if g_["q"] in (1e-3, 1e-2, 5e-2) and g_["t_init_pct"] in (85, 90, 95):
                sens_rows.append([f"{g_['t_init_pct']:g}", f"{g_['q']:g}", f4(g_["z_q"]), f"{g_['xi']:.3f}",
                                  f4(g_["beta"]), f4(g_["evt_precision"]), f4(g_["evt_recall"]),
                                  f4(g_["evt_f1"]), f4(g_["evt_mcc"]), f4(g_["evt_fpr"]), f4(g_["emp_f1"])])
        tables["sensitivity"] = sens_rows
    # --- SEL
    p_l = os.path.join(out, "stage_sel.json")
    if os.path.exists(p_l):
        S = jload(p_l)
        rows = [["Attribution method", "Top-1 accuracy", "Top-3 accuracy", "Repair success", "Time per window (ms)"]]
        for r in S["localization"]:
            rows.append([r["method"], f"{r['top1']:.3f}", f"{r['top3']:.3f}",
                         f"{S['fidelity_repair_success'].get(r['method'], float('nan')):.3f}",
                         fmt_ms_value(1000 * r["seconds_per_window"])])
        tables["sel"] = rows
        sel = S["localization"][0]
        ph["SEL.TOP1"] = f"{sel['top1']:.3f}"; ph["SEL.TOP3"] = f"{sel['top3']:.3f}"
        ph["SEL.N"] = f"{S['n_windows']:,}"; ph["SEL.DET"] = f"{S['detection_rate_single_event']:.3f}"
        ph["SEL.NFLAG"] = f"{S['n_flagged']:,}"
        ph["SEL.REPAIR"] = f"{S['fidelity_repair_success'].get(sel['method'], float('nan')):.3f}"
        for r in S["localization"][1:]:
            k = {"Random position": "RAND", "Bigram surprisal": "BIGRAM", "Occlusion": "OCC",
                 "Gradient saliency": "GRAD"}[r["method"]]
            ph[f"SEL.{k}.TOP1"] = f"{r['top1']:.3f}"
        ex = S["example"]
        fig, ax = plt.subplots(figsize=(3.5, 2.5))
        t = np.arange(1, len(ex["errors"]) + 1)
        ax.plot(t, ex["benign_errors"], marker="o", ms=3.5, color=PALETTE[0], label="Original benign window")
        ax.plot(t, ex["errors"], marker="s", ms=3.5, color=PALETTE[1], ls="--", label="Same window, one event replaced")
        ax.axvline(ex["true_pos"] + 1, color="#888888", lw=0.8, ls=":")
        ax.set_xticks(t, [f"{i}\n{ {0: 'Off', 1: 'On', 2: 'HTTP'}[k] }" for i, k in zip(t, ex["tokens"])], fontsize=6.5)
        ax.set_xlabel("Timestep and event in the modified window"); ax.set_ylabel("$E_t$ (squared error)")
        ax.legend(fontsize=6.5, frameon=False)
        _save(fig, out, "fig_sel_example"); plt.close(fig)
    # --- drift
    p_d = os.path.join(out, "stage_drift.json")
    if os.path.exists(p_d):
        D = jload(p_d)
        rows_d = D["rows"]
        months = [r["month"] + 1 for r in rows_d]  # 1-based month or block index
        unit = D.get("unit", "month")
        sh0 = D.get("shift_start", cfg.drift_shift_start); shl = D.get("shift_len", cfg.drift_shift_len)
        names = [("static99", "Static P99, fixed"), ("evt_fixed", "EVT, fixed"),
                 ("evt_spot", "EVT, alarm-censored refit"), ("evt_incl", "EVT, inclusive refit"),
                 ("rolling99", "Rolling P99, inclusive")]
        fig, axes = plt.subplots(3, 1, figsize=(3.5, 5.4), sharex=True)
        for i, (k, lab) in enumerate(names):
            axes[0].plot(months, [r[k]["threshold"] for r in rows_d], color=PALETTE[i], ls=LINESTYLES[i], label=lab)
            axes[1].plot(months, [r[k]["fpr"] for r in rows_d], color=PALETTE[i], ls=LINESTYLES[i])
            axes[2].plot(months, [r[k]["recall"] for r in rows_d], color=PALETTE[i], ls=LINESTYLES[i])
        for ax in axes:
            ax.axvspan(sh0 + 0.5, sh0 + shl + 0.5,
                       color="#f2f2ef", zorder=0)
        axes[0].set_ylabel("Threshold"); axes[1].set_ylabel("FPR"); axes[2].set_ylabel("Recall")
        axes[2].set_xlabel(("Month of dataset (1 = first month" if unit == "month" else
                            "Consecutive block of held-out windows (1 = earliest") + "; shaded: shift ramp)")
        axes[0].legend(fontsize=6, frameon=False)
        fig.tight_layout(); _save(fig, out, "fig_drift"); plt.close(fig)
        rows = [["Phase", "Strategy", "FPR", "Recall", "Precision", "F1", "MCC"]]
        for phs, dd in D["phases"].items():
            for k, lab in names:
                v = dd[k]
                rows.append([phs, lab, f4(v["fpr"]), f4(v["recall"]), f4(v["precision"]), f4(v["f1"]), f4(v["mcc"])])
                ph[f"DRIFT.{phs.upper()}.{k.upper()}.FPR"] = f4(v["fpr"])
                ph[f"DRIFT.{phs.upper()}.{k.upper()}.RECALL"] = f4(v["recall"])
                ph[f"DRIFT.{phs.upper()}.{k.upper()}.F1"] = f4(v["f1"])
            ph[f"DRIFT.{phs.upper()}.AUC"] = f4(dd["auc_shift"])
        tables["drift"] = rows
        ph["DRIFT.N_TRAIN"] = f"{D['n_train']:,}"
        if D.get("unit") == "block":
            ph["DRIFT.N_BLOCKS"] = str(D["n_blocks"]); ph["DRIFT.WIN_PER_BLOCK"] = f"{D['windows_per_block']:,}"
    # --- latency
    p_t = os.path.join(out, "stage_latency.json")
    if os.path.exists(p_t):
        Lt = jload(p_t)
        hw = Lt["hardware"]
        ph["HW.CPU"] = hw.get("processor") or hw.get("machine", "")
        ph["HW.PLATFORM"] = hw.get("platform", "")
        ph["HW.RAM"] = str(hw.get("ram_gb", "n/a"))
        for k in ["python", "tensorflow", "keras", "numpy", "scipy", "sklearn", "pandas"]:
            ph[f"VER.{k.upper()}"] = str(hw.get(k, "n/a"))
        sw = Lt["single_window"]
        e2e = Lt["end_to_end_single"]
        rows = [["Component (one window, batch size 1)", "Median (ms)", "95th percentile (ms)", "99th percentile (ms)"]]
        lab = {"inference": "LSTM-AE forward pass (compiled graph)", "score_sel": "Window error, EVT comparison and SEL",
               "serialize": "JSON payload construction", "rest": "REST call to controller endpoint (localhost)",
               "total": "Total: window ready to controller acknowledgement"}
        for k, v in e2e.items():
            rows.append([lab[k], fmt_ms_value(v["median_ms"]), fmt_ms_value(v["p95_ms"]), fmt_ms_value(v["p99_ms"])])
        rows.append(["Keras model.predict() on one window (for reference)", fmt_ms_value(sw["keras_predict"]["median_ms"]),
                     fmt_ms_value(sw["keras_predict"]["p95_ms"]), fmt_ms_value(sw["keras_predict"]["p99_ms"])])
        for nb, v in Lt["evt_refit"].items():
            rows.append([f"EVT refit on a {int(nb):,}-window buffer", fmt_ms_value(v["median_ms"]),
                         fmt_ms_value(v["p95_ms"]), fmt_ms_value(v["p99_ms"])])
        for N, v in Lt["end_to_end_concurrent"].items():
            rows.append([f"{N} concurrent users: batched inference + {N} REST calls", fmt_ms_value(v["median_ms"]),
                         fmt_ms_value(v["p95_ms"]), fmt_ms_value(v["p99_ms"])])
        tables["latency"] = rows
        ph["LAT.TOTAL_MED"] = fmt_ms_value(e2e["total"]["median_ms"])
        ph["LAT.TOTAL_P95"] = fmt_ms_value(e2e["total"]["p95_ms"])
        ph["LAT.TOTAL_P99"] = fmt_ms_value(e2e["total"]["p99_ms"])
        ph["LAT.INF_MED"] = fmt_ms_value(e2e["inference"]["median_ms"])
        ph["LAT.SEL_MED"] = fmt_ms_value(e2e["score_sel"]["median_ms"])
        ph["LAT.REST_MED"] = fmt_ms_value(e2e["rest"]["median_ms"])
        ph["LAT.PREDICT_MED"] = fmt_ms_value(sw["keras_predict"]["median_ms"])
        ph["LAT.CONC500_MED"] = fmt_ms_value(Lt["end_to_end_concurrent"]["500"]["median_ms"])
        ph["LAT.CONC500_P95"] = fmt_ms_value(Lt["end_to_end_concurrent"]["500"]["p95_ms"])
        ph["LAT.EVT100K"] = fmt_ms_value(Lt["evt_refit"]["100000"]["median_ms"])
    ph["NOTE.BACKEND"] = cfg.backend
    # write outputs
    jdump(ph, os.path.join(out, "placeholders.json"))
    jdump(tables, os.path.join(out, "tables.json"))
    with open(os.path.join(out, "results_summary.md"), "w", encoding="utf-8") as f:
        if cfg.backend != "keras":
            f.write("> WARNING: numpy smoke-test backend. These numbers are NOT reportable.\n\n")
        f.write("# Re-run results\n\n")
        for k, v in sorted(ph.items()):
            f.write(f"- {k}: {v}\n")
        for tname, rows in tables.items():
            f.write(f"\n## Table: {tname}\n\n")
            f.write("| " + " | ".join(rows[0]) + " |\n|" + "---|" * len(rows[0]) + "\n")
            for r in rows[1:]:
                f.write("| " + " | ".join(str(c) for c in r) + " |\n")
    log(f"Report written to {out}/results_summary.md, placeholders.json, tables.json, figures/")
    return ph, tables


# --------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="r4.2")
    ap.add_argument("--arrays-dir", default=None,
                    help="folder with X_train_lstm.pkl, X_test_eval.pkl and y_true.pkl (no raw CSVs needed)")
    ap.add_argument("--out", default="results")
    ap.add_argument("--backend", default="keras", choices=["keras", "numpy"])
    ap.add_argument("--stages", default="main,sens,sel,latency,drift,contam,window,holdout,report")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    cfg = Config(data_dir=a.data_dir, arrays_dir=a.arrays_dir, out=a.out, backend=a.backend, force=a.force)
    if a.quick:
        cfg.apply_quick()
    os.makedirs(cfg.out, exist_ok=True)
    jdump(asdict(cfg), os.path.join(cfg.out, "config.json"))
    stages = [s.strip() for s in a.stages.split(",") if s.strip()]
    be = get_backend(cfg.backend) if any(s != "report" for s in stages) else None
    t0 = time.time()
    for st in stages:
        log(f"=== stage {st} ===")
        if st == "report":
            report(cfg)
        else:
            globals()[f"stage_{st}"](cfg, be)
    log(f"Done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
