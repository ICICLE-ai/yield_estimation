import re
import math
import numpy as np
import pandas as pd

DEFAULT_WEATHER_AGG_RULES = {
    "dayl": "mean",
    "prcp": "sum",
    "srad": "mean",
    "tmax": "mean",
    "tmin": "mean",
    "vp": "mean",
    "tmean": "mean",
    "gdd": "sum",
    "precip_3day_avg_perday": "mean",
    "precip_7day_avg_perday": "mean",
    "precip_14day_avg_perday": "mean",
}

def load_table(path):
    path = str(path)
    if path.endswith(".h5") or path.endswith(".hdf5"):
        return pd.read_hdf(path)
    if path.endswith(".csv"):
        return pd.read_csv(path)
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported file type: {path}")

def decode_bytes_in_object_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.select_dtypes(include=["object"]).columns:
        if len(out) > 0 and isinstance(out[c].iloc[0], (bytes, bytearray)):
            out[c] = out[c].str.decode("utf-8")
    return out

def find_daily_cols(df: pd.DataFrame, var: str) -> list[str]:
    pat_bracket = re.compile(rf"^{re.escape(var)}.*?_\[(\d+)\]$")
    pat_plain = re.compile(rf"^{re.escape(var)}.*?_(\d+)$")
    hits = []
    for c in df.columns:
        s = str(c)
        m = pat_bracket.match(s) or pat_plain.match(s)
        if m:
            hits.append((int(m.group(1)), c))
    hits.sort(key=lambda x: x[0])
    return [c for _, c in hits]

def daily_to_cumulative_weekly(daily: np.ndarray, agg: str, week_len: int = 7) -> np.ndarray:
    daily = daily.astype(np.float32)
    T = daily.shape[0]
    K = int(math.ceil(T / week_len))
    out = np.zeros(K, dtype=np.float32)

    for w in range(K):
        e = min((w + 1) * week_len, T)
        chunk = daily[:e]
        out[w] = np.nansum(chunk) if agg == "sum" else np.nanmean(chunk)

    return out

def compute_x_stats(dataset, max_samples=20000, seed=0):
    rng = np.random.default_rng(seed)
    n = min(len(dataset), max_samples)
    idxs = rng.choice(len(dataset), size=n, replace=False) if n < len(dataset) else np.arange(len(dataset))

    weather, soil = [], []
    for i in idxs:
        item = dataset[int(i)]
        weather.append(item["weather"].numpy())
        soil.append(item["soil"].numpy())

    W_all = np.stack(weather).astype(np.float32)
    S_all = np.stack(soil).astype(np.float32)

    w_mean = np.nanmean(W_all, axis=(0, 1))
    w_std = np.nanstd(W_all, axis=(0, 1)) + 1e-6
    s_mean = np.nanmean(S_all, axis=0)
    s_std = np.nanstd(S_all, axis=0) + 1e-6

    w_mean = np.where(np.isfinite(w_mean), w_mean, 0.0)
    w_std = np.where((np.isfinite(w_std)) & (w_std > 1e-6), w_std, 1.0)
    s_mean = np.where(np.isfinite(s_mean), s_mean, 0.0)
    s_std = np.where((np.isfinite(s_std)) & (s_std > 1e-6), s_std, 1.0)

    return w_mean, w_std, s_mean, s_std

def compute_y_log_stats(dataset):
    ys = np.array([float(dataset[i]["yield"]) for i in range(len(dataset))], dtype=np.float32)
    ys_log = np.log1p(np.clip(ys, 0.0, None))
    return float(ys_log.mean()), float(ys_log.std() + 1e-6)