"""Pull the S&P 500 daily-return panel from WRDS / CRSP and cache it.

Produces /tmp/sp500_panel.npz (an N×T return matrix) consumed by
`sp500_noise_dressing.py`. Requires a WRDS account; credentials are read by the
`wrds` package from ~/.pgpass (no secrets in this file). Set WRDS_USERNAME or
pass --user; defaults to the username in ~/.pgpass.

    python examples/pull_sp500.py --start 2018-01-01 --end 2022-12-31
"""
import argparse
import os

import numpy as np
import pandas as pd
import wrds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2018-01-01")
    ap.add_argument("--end", default="2022-12-31")
    ap.add_argument("--user", default=os.environ.get("WRDS_USERNAME"))
    ap.add_argument("--out", default="/tmp/sp500_panel.npz")
    args = ap.parse_args()

    user = args.user
    if user is None:                       # fall back to the username in ~/.pgpass
        with open(os.path.expanduser("~/.pgpass")) as fh:
            user = fh.readline().strip().split(":")[3]

    db = wrds.Connection(wrds_username=user)
    # stocks continuously in the S&P 500 across the whole window
    members = db.raw_sql(f"""
        select distinct permno from crsp.dsp500list
        where start <= '{args.start}' and ending >= '{args.end}'
    """)
    permnos = members["permno"].astype(int).tolist()
    plist = ",".join(str(p) for p in permnos)
    rets = db.raw_sql(f"""
        select permno, date, ret from crsp.dsf
        where permno in ({plist})
          and date between '{args.start}' and '{args.end}'
          and ret is not null
    """)
    db.close()

    rets["date"] = pd.to_datetime(rets["date"])
    panel = rets.pivot(index="date", columns="permno", values="ret").sort_index()
    panel = panel.dropna(axis=1, how="any")     # keep complete-history stocks only
    np.savez(
        args.out,
        returns=panel.values.T,                 # N x T
        permnos=np.array(panel.columns, dtype=int),
        dates=panel.index.astype(str).values,
    )
    N, T = panel.shape[1], panel.shape[0]
    print(f"saved {args.out}  (N={N} stocks x T={T} days, Q={T/N:.2f})")


if __name__ == "__main__":
    main()
