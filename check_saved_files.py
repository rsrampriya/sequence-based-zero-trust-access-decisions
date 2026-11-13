"""Check what the saved files in the project folder actually contain.

Run inside the project folder (where X_train_lstm.pkl etc. are):
    python check_saved_files.py
It prints a report and also writes it to check_output.txt. Send that file back.
Takes 2-5 minutes (one pass of the LSTM over the 500,000 training windows).
"""
import glob
import io
import os
import sys
import traceback

import numpy as np

OUT = io.StringIO()


def say(*a):
    msg = " ".join(str(x) for x in a)
    print(msg)
    OUT.write(msg + "\n")


def section(t):
    say("\n" + "=" * 70 + "\n" + t + "\n" + "=" * 70)


def safe(fn):
    try:
        fn()
    except Exception as e:  # keep going, report the problem
        say(f"  !! could not complete: {type(e).__name__}: {e}")
        OUT.write(traceback.format_exc())


def load(path):
    import joblib
    return joblib.load(path)


def describe(obj, indent="  "):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, (list, tuple)) and len(v) <= 10:
                say(f"{indent}{k}: {v}")
            elif isinstance(v, np.ndarray):
                say(f"{indent}{k}: array shape={v.shape} dtype={v.dtype}")
            elif isinstance(v, dict):
                say(f"{indent}{k}:"); describe(v, indent + "  ")
            else:
                say(f"{indent}{k}: {v!r}"[:300])
    elif isinstance(obj, np.ndarray):
        say(f"{indent}array shape={obj.shape} dtype={obj.dtype}")
    else:
        say(f"{indent}{type(obj).__name__}: {repr(obj)[:300]}")


def small_pickles():
    section("1. Small saved files (thresholds and metrics)")
    names = ["cert_evt_parameters.pkl", "evt_parameters.pkl", "threshold_parameters.pkl",
             "math_and_thresholds.pkl", "lstm_multi_seed_metrics.pkl", "dense_multi_seed_metrics.pkl",
             "evaluation_metrics.pkl", "lanl_evt_parameters.pkl", "lanl_recalibrated_evt.pkl"]
    for n in names:
        if not os.path.exists(n):
            say(f"- {n}: not found"); continue
        say(f"- {n} (modified {os.path.getmtime(n):.0f}):")
        safe(lambda: describe(load(n)))


def arrays():
    section("2. Saved arrays")
    for n in ["X_train_lstm.pkl", "X_test_eval.pkl", "y_true.pkl", "all_data_arrays.pkl"]:
        if not os.path.exists(n):
            say(f"- {n}: not found"); continue
        a = load(n)
        if isinstance(a, dict):
            say(f"- {n}: dict with keys {list(a.keys())}")
            describe(a)
            continue
        a = np.asarray(a)
        vals = np.unique(a)[:10]
        say(f"- {n}: shape={a.shape} dtype={a.dtype} distinct values (first 10)={vals}")
    Xtr = np.asarray(load("X_train_lstm.pkl")); Xtr = Xtr.reshape(len(Xtr), -1)
    Xev = np.asarray(load("X_test_eval.pkl")); Xev = Xev.reshape(len(Xev), -1)
    y = np.asarray(load("y_true.pkl")).ravel().astype(int)
    say(f"  y_true counts: benign={int((y == 0).sum())} anomalous={int((y == 1).sum())}")
    say(f"  first anomalous index={int(np.argmax(y == 1))} (expected 9000 if benign windows come first)")
    A, N = Xev[y == 1], Xev[y == 0]
    say(f"  anomalies containing only Logoff(0)/HTTP(2): {100 * np.isin(A, [0, 2]).all(1).mean():.1f}%")
    say(f"  token share in training windows: " + str(np.round(np.bincount(Xtr.ravel().astype(int), minlength=3) / Xtr.size, 4)))
    say(f"  benign test windows made only of HTTP: {100 * (N == 2).all(1).mean():.1f}%")
    cont = np.mean([np.array_equal(N[i, :-1], N[i - 1, 1:]) for i in range(1, len(N))])
    breaks = int(round((1 - cont) * (len(N) - 1)))
    say(f"  consecutive benign windows that overlap by 9 events: {100 * cont:.2f}% "
        f"-> {breaks + 1} contiguous segment(s) (approx. number of users in the benign test set)")
    dup = len(N) - len(np.unique(N, axis=0))
    say(f"  duplicate benign windows: {dup} of {len(N)}")


def models():
    section("3. Saved models")
    from tensorflow import keras
    for n in ["lstm_zta_model.keras", "bilstm_baseline_model.keras", "dense_ablated_model.keras",
              "gru_baseline_model.keras", "lanl_lstm_model.keras"]:
        if not os.path.exists(n):
            say(f"- {n}: not found"); continue
        m = keras.models.load_model(n, compile=False)
        say(f"- {n}: params={m.count_params():,} input={m.input_shape} output={m.output_shape}")
        for L in m.layers:
            cfg = L.get_config()
            inner = cfg.get("layer", {}).get("config", {}) if "layer" in cfg else {}
            act = cfg.get("activation", inner.get("activation"))
            units = cfg.get("units", inner.get("units", cfg.get("filters")))
            say(f"    {L.__class__.__name__:<18} units={units} activation={act} "
                f"return_sequences={cfg.get('return_sequences', inner.get('return_sequences'))}")


def evt_check():
    section("4. Which threshold belongs to lstm_zta_model.keras?")
    from scipy.stats import genpareto
    from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, matthews_corrcoef
    from tensorflow import keras
    m = keras.models.load_model("lstm_zta_model.keras", compile=False)
    Xtr = np.asarray(load("X_train_lstm.pkl")).astype("float32").reshape(-1, 10, 1)
    Xev = np.asarray(load("X_test_eval.pkl")).astype("float32").reshape(-1, 10, 1)
    y = np.asarray(load("y_true.pkl")).ravel().astype(int)
    say("  reconstructing 500,000 training windows (1-3 min) ...")
    s_tr = np.mean((Xtr[..., 0] - m.predict(Xtr, batch_size=4096, verbose=0)[..., 0]) ** 2, axis=1)
    s_ev = np.mean((Xev[..., 0] - m.predict(Xev, batch_size=4096, verbose=0)[..., 0]) ** 2, axis=1)
    t0 = np.percentile(s_tr, 90); pk = s_tr[s_tr > t0] - t0
    xi, _, beta = genpareto.fit(pk, floc=0)
    r = 0.01 * len(s_tr) / len(pk)
    z = t0 + beta / xi * (r ** (-xi) - 1) if abs(xi) > 1e-12 else t0 - beta * np.log(r)
    say(f"  POT threshold fitted on this model's training errors: z_q={z:.6f} (xi={xi:.4f}, beta={beta:.6f})")
    say(f"  static 99th percentile of training errors: {np.percentile(s_tr, 99):.6f}")
    say(f"  ROC-AUC on X_test_eval: {roc_auc_score(y, s_ev):.4f}")
    for name, thr in [("own POT threshold", z), ("0.0195 (value used in Table 6)", 0.0195),
                      ("0.054219", 0.054219), ("static P99", np.percentile(s_tr, 99))]:
        pred = (s_ev > thr).astype(int)
        p, rc, f1, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
        say(f"  {name:<32} thr={thr:.6f}  P={p:.4f} R={rc:.4f} F1={f1:.4f} MCC={matthews_corrcoef(y, pred):.4f}")


def environment():
    section("5. Environment")
    import platform
    say(f"  python {sys.version.split()[0]} | {platform.platform()} | {platform.processor()}")
    for mod in ["numpy", "pandas", "scipy", "sklearn", "tensorflow", "keras", "matplotlib", "joblib"]:
        try:
            say(f"  {mod} {__import__(mod).__version__}")
        except Exception:
            say(f"  {mod} not installed")
    say("  other files: " + ", ".join(sorted(glob.glob("*.pkl")))[:600])


if __name__ == "__main__":
    for fn in (environment, small_pickles, arrays, models, evt_check):
        safe(fn)
    with open("check_output.txt", "w", encoding="utf-8") as f:
        f.write(OUT.getvalue())
    print("\nSaved the report to check_output.txt")
