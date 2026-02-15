#!/usr/bin/env python3
import argparse
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

def load_pkl(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)

def to_numpy(df_or_series):
    if isinstance(df_or_series, pd.DataFrame):
        return df_or_series.to_numpy(dtype=float)
    if isinstance(df_or_series, pd.Series):
        return df_or_series.to_numpy(dtype=float)
    if isinstance(df_or_series, np.ndarray):
        return df_or_series.astype(float)
    raise TypeError(f"Unsupported type: {type(df_or_series)}")

def summarize_samples(samples, name="samples"):
    """
    samples: list[pd.DataFrame] (n_series) or pd.DataFrame (single series)
    Each DF expected shape: (num_samples, horizon)
    """
    if isinstance(samples, pd.DataFrame):
        samples_list = [samples]
    else:
        samples_list = list(samples)

    out = []
    for si, df in enumerate(samples_list):
        a = to_numpy(df)
        # const rows
        const_ratio = float(((np.nanmax(a, axis=1) - np.nanmin(a, axis=1)) == 0).mean())
        uniq = int(len(np.unique(a)))
        mn, mean, mx = float(np.nanmin(a)), float(np.nanmean(a)), float(np.nanmax(a))
        p1, p5, p50, p95, p99 = [float(np.nanpercentile(a, q)) for q in (1,5,50,95,99)]

        out.append({
            "series": si,
            "shape": tuple(a.shape),
            "unique_floats": uniq,
            "const_row_ratio": const_ratio,
            "min": mn, "mean": mean, "max": mx,
            "p1": p1, "p5": p5, "p50": p50, "p95": p95, "p99": p99,
        })
    return out

def summarize_median(median):
    if isinstance(median, pd.Series):
        med_list = [median]
    else:
        med_list = list(median)

    out = []
    for si, ser in enumerate(med_list):
        a = to_numpy(ser)
        out.append({
            "series": si,
            "len": int(len(a)),
            "min": float(np.nanmin(a)),
            "mean": float(np.nanmean(a)),
            "max": float(np.nanmax(a)),
        })
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", help="dataset name like weather (expects outputs/monash/<dataset>.pkl)")
    ap.add_argument("--model", default="ollama/llama2:text", help="model key inside pkl dict")
    ap.add_argument("--pkl", default=None, help="optional direct path to pkl (overrides dataset)")
    args = ap.parse_args()

    if args.pkl:
        pkl_path = Path(args.pkl)
    else:
        pkl_path = Path("outputs/monash") / f"{args.dataset}.pkl"

    if not pkl_path.exists():
        raise SystemExit(f"File not found: {pkl_path}")

    obj = load_pkl(pkl_path)

    if args.model not in obj:
        # sometimes saved as {model: out_dict} or directly out_dict
        if isinstance(obj, dict) and "samples" in obj and "median" in obj:
            data = obj
        else:
            raise SystemExit(f"Model key '{args.model}' not found. Available keys: {list(obj.keys())[:20]}")
    else:
        data = obj[args.model]

    print(f"\n=== PKL ===\npath: {pkl_path}\nmodel: {args.model}")

    # basics
    samples = data.get("samples", None)
    median = data.get("median", None)
    completions_list = data.get("completions_list", None)
    input_strs = data.get("input_strs", None)

    if samples is None:
        print("\n[WARN] no 'samples' found")
    else:
        print("\n=== SAMPLES SUMMARY ===")
        rows = summarize_samples(samples)
        for r in rows:
            print(
                f"series {r['series']}: shape={r['shape']}, unique={r['unique_floats']}, "
                f"const_rows={r['const_row_ratio']:.1%}, "
                f"min/mean/max={r['min']:.3f}/{r['mean']:.3f}/{r['max']:.3f}, "
                f"p1/p50/p99={r['p1']:.3f}/{r['p50']:.3f}/{r['p99']:.3f}"
            )

    if median is None:
        print("\n[WARN] no 'median' found")
    else:
        print("\n=== MEDIAN SUMMARY ===")
        rows = summarize_median(median)
        for r in rows:
            print(f"series {r['series']}: len={r['len']}, min/mean/max={r['min']:.3f}/{r['mean']:.3f}/{r['max']:.3f}")

    # completion stats (optional)
    if completions_list is not None:
        print("\n=== COMPLETIONS (light) ===")
        # completions_list: list(n_series) of list(num_samples) of str
        if isinstance(completions_list, list) and completions_list and isinstance(completions_list[0], list):
            for si in range(min(3, len(completions_list))):
                lens = [len(str(s)) for s in completions_list[si] if s is not None]
                if lens:
                    print(f"series {si}: completion strings count={len(lens)}, char_len min/med/max={min(lens)}/{sorted(lens)[len(lens)//2]}/{max(lens)}")
        else:
            print("unexpected completions_list structure:", type(completions_list))

    if input_strs is not None:
        print("\n=== INPUT_STRS (head) ===")
        if isinstance(input_strs, (list, tuple)) and input_strs:
            for si in range(min(3, len(input_strs))):
                s = str(input_strs[si])
                print(f"series {si} head: {s[:120]!r}")
        else:
            print("unexpected input_strs structure:", type(input_strs))

if __name__ == "__main__":
    main()
