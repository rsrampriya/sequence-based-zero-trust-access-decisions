"""Generate a tiny CERT-r4.2-shaped dataset (logon.csv, http.csv) for smoke tests.

The data are random and carry no meaning; they only exercise the pipeline.
Usage: python tests/make_synthetic_cert.py --out tests/synthetic_r4.2 --users 40
"""
import argparse
import os

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="tests/synthetic_r4.2")
    ap.add_argument("--users", type=int, default=40)
    ap.add_argument("--days", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    os.makedirs(a.out, exist_ok=True)
    start = pd.Timestamp("2010-01-02")
    users = [f"U{rng.integers(1000, 9999)}{chr(65 + i % 26)}{i:03d}" for i in range(a.users)]
    lg, ht = [], []
    for u in users:
        pc = f"PC-{rng.integers(1000, 9999)}"
        for d in range(a.days):
            if rng.random() < 0.28:  # weekends / leave
                continue
            day = start + pd.Timedelta(days=d)
            t_on = day + pd.Timedelta(minutes=int(rng.normal(8 * 60, 30)))
            t_off = day + pd.Timedelta(minutes=int(rng.normal(17 * 60, 40)))
            lg.append((t_on, u, pc, "Logon"))
            n = rng.poisson(12)
            for t in np.sort(rng.uniform(0, (t_off - t_on).total_seconds(), n)):
                ht.append((t_on + pd.Timedelta(seconds=float(t)), u, pc, "http://example.org"))
            if rng.random() < 0.05:  # mid-day re-authentication
                tm = t_on + (t_off - t_on) / 2
                lg.append((tm, u, pc, "Logoff"))
                lg.append((tm + pd.Timedelta(seconds=30), u, pc, "Logon"))
            lg.append((t_off, u, pc, "Logoff"))
    fmt = "%m/%d/%Y %H:%M:%S"
    L = pd.DataFrame(lg, columns=["date", "user", "pc", "activity"])
    L["date"] = L["date"].dt.strftime(fmt)
    L.insert(0, "id", [f"{{L{i}}}" for i in range(len(L))])
    H = pd.DataFrame(ht, columns=["date", "user", "pc", "url"])
    H["date"] = H["date"].dt.strftime(fmt)
    H.insert(0, "id", [f"{{H{i}}}" for i in range(len(H))])
    L.to_csv(os.path.join(a.out, "logon.csv"), index=False)
    H.to_csv(os.path.join(a.out, "http.csv"), index=False)
    print(f"wrote {len(L):,} logon and {len(H):,} http rows for {a.users} users to {a.out}")


if __name__ == "__main__":
    main()
