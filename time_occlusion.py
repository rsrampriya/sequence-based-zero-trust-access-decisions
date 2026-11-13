"""Per-window latency of occlusion localization (batch size 1).

For one window, occlusion needs the original window plus T copies in which one
event is replaced by the most frequent training token (HTTP = 2). All T + 1
windows are scored in a single forward call of the compiled model, exactly as
the latency stage scores one window.

Run from the folder that holds X_test_eval.pkl, y_true.pkl and results/:
    python time_occlusion.py
Writes results/occlusion_latency.json and prints the numbers.
"""
import json
import os
import time

import joblib
import numpy as np
import tensorflow as tf
import keras

MODEL = os.path.join("results", "models", "lstm_42.keras")
OUT = os.path.join("results", "occlusion_latency.json")
MODE_TOKEN = 2
N_WARMUP, N_REPS = 200, 2000

X = np.asarray(joblib.load("X_test_eval.pkl")).reshape(-1, 10)
y = np.asarray(joblib.load("y_true.pkl")).astype(int).ravel()
windows = X[y == 1].astype(np.float32)
T = windows.shape[1]

model = keras.models.load_model(MODEL, compile=False)
spec = tf.TensorSpec(shape=(None, T, 1), dtype=tf.float32)
fwd = tf.function(lambda x: model(x, training=False), input_signature=[spec])


def occlusion_batch(w):
    b = np.repeat(w[None, :], T + 1, axis=0)
    for t in range(T):
        b[t + 1, t] = MODE_TOKEN
    return b[..., None]


def one_window(w):
    b = occlusion_batch(w)
    r = fwd(tf.constant(b)).numpy()[..., 0]
    err = ((b[..., 0] - r) ** 2).mean(1)
    drop = err[0] - err[1:]
    return int(np.argmax(drop))


def single_forward(w):
    fwd(tf.constant(w[None, :, None])).numpy()


rng = np.random.default_rng(0)
idx = rng.integers(0, len(windows), N_WARMUP + N_REPS)
for i in idx[:N_WARMUP]:
    one_window(windows[i]); single_forward(windows[i])

occ, base = [], []
for i in idx[N_WARMUP:]:
    t0 = time.perf_counter(); one_window(windows[i]); occ.append(time.perf_counter() - t0)
    t0 = time.perf_counter(); single_forward(windows[i]); base.append(time.perf_counter() - t0)


def summary(v):
    v = np.array(v) * 1000
    return {"median_ms": float(np.median(v)), "p95_ms": float(np.percentile(v, 95)),
            "p99_ms": float(np.percentile(v, 99))}


res = {"occlusion_one_window_incl_forward": summary(occ),
       "plain_forward_one_window": summary(base),
       "n_reps": N_REPS, "T": T, "model": MODEL,
       "tensorflow": tf.__version__, "keras": keras.__version__}
with open(OUT, "w") as f:
    json.dump(res, f, indent=2)
print(json.dumps(res, indent=2))
