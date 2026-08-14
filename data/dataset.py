from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from data.preprocessing import (
    load_table,
    decode_bytes_in_object_cols,
    find_daily_cols,
    daily_to_cumulative_weekly,
    DEFAULT_WEATHER_AGG_RULES,
)

class YieldDataset(Dataset):
    def __init__(
        self,
        data_file: str | Path,
        weather_vars: list[str],
        soil_vars: list[str],
        split: str = "all",
        seed: int = 1234,
        crop: str | None = None,
        years: str | None = None,
        crop_col: str = "crop",
        yield_col: str = "yield",
        field_col: str = "farm_field",
        year_col: str = "year",
        time_agg: str = "weekly_cumulative",
        split_strategy: str = "field",
        val_split: float = 0.2,
        test_split: float = 0.2,
        require_yield: bool = True,
    ):
        self.weather_vars = weather_vars
        self.soil_vars = soil_vars
        self.crop_col = crop_col
        self.yield_col = yield_col
        self.field_col = field_col
        self.year_col = year_col
        self.time_agg = time_agg
        self.crop_map = {"corn": 0, "maize": 0, "soybean": 1, "soy": 1}

        self.w_mean = None
        self.w_std = None
        self.s_mean = None
        self.s_std = None

        df = load_table(data_file)
        df = decode_bytes_in_object_cols(df)
        df.columns = [str(c).strip() for c in df.columns]

        if yield_col != "yield" and yield_col in df.columns:
            df = df.rename(columns={yield_col: "yield"})
            self.yield_col = "yield"

        required = [crop_col, field_col, year_col] + soil_vars
        if require_yield:
            required.append(self.yield_col)

        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing[:20]}")

        if require_yield:
            df[self.yield_col] = pd.to_numeric(df[self.yield_col], errors="coerce")
            df = df.dropna(subset=[self.yield_col]).reset_index(drop=True)
        else:
            if self.yield_col not in df.columns:
                df[self.yield_col] = np.nan

        df[crop_col] = df[crop_col].astype(str).str.strip().str.lower()
        if crop:
            df = df[df[crop_col] == crop.strip().lower()].reset_index(drop=True)

        if years:
            year_list = [int(y.strip()) for y in years.split(",") if y.strip()]
            df = df[df[year_col].isin(year_list)].reset_index(drop=True)

        self.weather_cols_by_var = {}

        for v in weather_vars:
            cols = find_daily_cols(df, v)

            if not cols:
                examples = [c for c in df.columns if str(c).startswith(v)][:10]
                raise ValueError(
                    f"No indexed columns found for weather var '{v}'. "
                    f"Examples: {examples}"
                )

            self.weather_cols_by_var[v] = cols


        if self.time_agg == "weekly":
            # Input data is already weekly.
            self.K = len(
                self.weather_cols_by_var[weather_vars[0]]
            )

        elif self.time_agg == "weekly_cumulative":
            # Existing behavior for daily input.
            self.K = len(
                daily_to_cumulative_weekly(
                    df.loc[
                        0,
                        self.weather_cols_by_var[weather_vars[0]]
                    ].to_numpy(dtype=np.float32),
                    agg=DEFAULT_WEATHER_AGG_RULES.get(
                        weather_vars[0],
                        "mean",
                    ),
                )
            )

        else:
            raise ValueError(
                "time_agg must be 'weekly' or 'weekly_cumulative'"
            )

        df = df.reset_index(drop=True)
        self.df_full = df
        self.indices = self._make_split_indices(df, split, seed, split_strategy, val_split, test_split)

        self.df = df
        self.soil_arr = df[soil_vars].to_numpy(np.float32)
        self.crop_arr = df[crop_col].astype(str).str.strip().str.lower().to_numpy()
        self.y_arr = df[self.yield_col].to_numpy(np.float32)

    def _make_split_indices(self, df, split, seed, split_strategy, val_split, test_split):
        rng = np.random.default_rng(seed)

        if split == "all" or split_strategy == "none":
            return np.arange(len(df))

        if split_strategy == "field":
            fields = df[self.field_col].astype(str).unique()
            rng.shuffle(fields)

            n = len(fields)
            n_test = max(1, int(test_split * n)) if n >= 3 else 0
            n_val = max(1, int(val_split * n)) if n >= 3 else 0
            n_train = n - n_val - n_test

            train_fields = set(fields[:n_train])
            val_fields = set(fields[n_train:n_train + n_val])
            test_fields = set(fields[n_train + n_val:])

            if split == "train":
                keep = train_fields
            elif split == "val":
                keep = val_fields
            elif split == "test":
                keep = test_fields
            else:
                raise ValueError("split must be train, val, test, or all")

            return df.index[df[self.field_col].astype(str).isin(keep)].to_numpy()

        if split_strategy == "random":
            idx = np.arange(len(df))
            rng.shuffle(idx)
            n = len(idx)
            n_test = int(test_split * n)
            n_val = int(val_split * n)
            n_train = n - n_val - n_test

            if split == "train":
                return idx[:n_train]
            if split == "val":
                return idx[n_train:n_train + n_val]
            if split == "test":
                return idx[n_train + n_val:]

        raise ValueError(f"Unknown split_strategy={split_strategy}")

    def set_normalization(self, w_mean, w_std, s_mean, s_std):
        self.w_mean = np.asarray(w_mean, dtype=np.float32)
        self.w_std = np.asarray(w_std, dtype=np.float32)
        self.s_mean = np.asarray(s_mean, dtype=np.float32)
        self.s_std = np.asarray(s_std, dtype=np.float32)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        ridx = int(self.indices[idx])
        row = self.df.loc[ridx]

        weather_vars = []

        for v in self.weather_vars:

            values = row[
                self.weather_cols_by_var[v]
            ].to_numpy(dtype=np.float32)

            if self.time_agg == "weekly":

                # Data is already weekly.
                # Use the 52 weekly values exactly as supplied.
                seq = values

            elif self.time_agg == "weekly_cumulative":

                # Existing behavior for datasets containing daily weather.
                agg = DEFAULT_WEATHER_AGG_RULES.get(v, "mean")

                seq = daily_to_cumulative_weekly(
                    values,
                    agg=agg,
                    week_len=7,
                )

            weather_vars.append(seq)


        if self.time_agg == "weekly":

            # [K, number_of_weather_variables]
            weather = np.stack(
                weather_vars,
                axis=1,
            ).astype(np.float32)

        else:

            # Existing daily -> weekly+cumulative representation
            weather = np.stack(
                weather_vars,
                axis=1,
            ).astype(np.float32)

            weather = weather.reshape(
                weather.shape[0],
                -1,
            ).astype(np.float32)
        
        soil = self.soil_arr[ridx].astype(np.float32)

        if self.w_mean is not None:
            weather = np.where(np.isnan(weather), self.w_mean[None, :], weather)
            weather = (weather - self.w_mean[None, :]) / self.w_std[None, :]

        if self.s_mean is not None:
            soil = np.where(np.isnan(soil), self.s_mean, soil)
            soil = (soil - self.s_mean) / self.s_std

        crop = str(self.crop_arr[ridx]).strip().lower()
        crop_id = self.crop_map.get(crop, 0)

        return {
            "weather": torch.from_numpy(weather),
            "soil": torch.from_numpy(soil),
            "crop_id": torch.tensor(crop_id, dtype=torch.long),
            "yield": torch.tensor(float(self.y_arr[ridx]), dtype=torch.float32),
            "farm_field": str(row[self.field_col]),
            "year": int(row[self.year_col]),
        }