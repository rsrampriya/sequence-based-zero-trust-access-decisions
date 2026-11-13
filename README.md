# Threshold calibration and event localization for sequence-based zero trust access decisions

Code and derived data for the Scientific Reports manuscript
"Threshold calibration and event localization rather than detector architecture limit
sequence-based zero trust access decisions" (Gupta, Agarwal and Rampriya R S).

The pipeline scores ten-event windows of user activity (logoff = 0, logon = 1, web request = 2)
from the CERT r4.2 insider threat dataset with a reconstruction model, converts the score into an
alert with a threshold fitted on training errors, localizes the event behind the alert, and sends a
structured request to a controller endpoint. `zt_rerun.py` runs every experiment in the paper:

| Stage | What it does | Paper |
|---|---|---|
| `main` | LSTM, GRU, Bi-LSTM, 1D-CNN and dense autoencoders, isolation forest and one-class SVM; five seeds; POT and static 99th-percentile thresholds fitted on each model's own training scores | Table 2, Figs 2–3, Table S1 |
| `sens` | POT sensitivity grid and generalized Pareto goodness of fit | Fig. 4, Table S2 |
| `drift` | Controlled benign drift with fixed and refitted thresholds | Table 3, Fig. 5 |
| `sel` | Localization of a single altered event (SEL, occlusion, gradient, bigram, random) | Table 4, Fig. 6 |
| `contam`, `window`, `holdout` | Training contamination, window length T = 5, many-user benign hold-out | Table 5 |
| `latency` | Decision-path latency up to a local REST endpoint | Table 6 |
| `report` | Tables, placeholders and figures (PDF and 600-dpi PNG) | all |

`time_occlusion.py` measures single-window occlusion latency (Latency section), and
`check_saved_files.py` prints the properties of the saved arrays and model used to reconcile
earlier results (response letter, Table R1).

## Data

The raw CERT r4.2 files are available from the Carnegie Mellon University Software Engineering
Institute (https://doi.org/10.1184/R1/12841247). The experiments in the paper were run from
arrays saved during the original preprocessing, which are archived with this code on Zenodo:

* `X_train_lstm.pkl` – 500,000 training windows of the 800 training users, shape (500000, 10, 1)
* `X_test_eval.pkl` – evaluation windows (9,000 benign, 1,000 synthetic anomalies)
* `y_true.pkl` – labels of the evaluation windows

The benign evaluation windows are consecutive windows of a single held-out user; see the
manuscript (Results, Evaluation setting) for the consequences.

## Reproducing the results

```bash
pip install -r requirements.txt
# put the three .pkl files in the current folder, then
python zt_rerun.py --arrays-dir . --out results            # all stages, about 70 min on a laptop CPU
python time_occlusion.py                                   # needs results/models/lstm_42.keras
```

If the raw CERT CSV files are available instead, use `--data-dir path/to/r4.2`.
Each stage writes a JSON checkpoint to `results/`; completed stages are skipped on a rerun
(`--force` recomputes them). `python zt_rerun.py --out results --stages report` rebuilds tables and
figures from existing checkpoints. Results in the paper were produced with Python 3.13.7,
TensorFlow 2.21.0, Keras 3.15.1, NumPy 2.5.4, SciPy 1.18.1, scikit-learn 1.9.1 and pandas 3.1.1
on an Intel Core i5-13420H CPU with 16 GB RAM, without a GPU.

The archived `results/` folder contains the per-run metrics, scores, trained models and figures of
the reported run.

## Smoke test

The GitHub Actions workflow (`.github/workflows/smoke.yml`) generates a small synthetic dataset in
the CERT file format (`tests/make_synthetic_cert.py`) and runs the complete pipeline with a NumPy
stand-in model, so the code can be checked without the dataset or TensorFlow:

```bash
python tests/make_synthetic_cert.py --out tests/synthetic_r4.2 --users 40 --days 500
python zt_rerun.py --data-dir tests/synthetic_r4.2 --out smoke_results --backend numpy --quick
```

The stand-in model is for testing only; its numbers have no meaning.

## License

MIT License (see `LICENSE`).
