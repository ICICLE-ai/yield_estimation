import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoModel

from data.dataset import YieldDataset
from data.preprocessing import daily_to_cumulative_weekly, DEFAULT_WEATHER_AGG_RULES
from hf.auto import register_yield_autoclass


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--hf_model_dir", required=True)
    p.add_argument("--input_file", default=None)
    p.add_argument("--single_sample_json", default=None)
    p.add_argument("--cutoff", type=int, required=True)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--output_csv", default="inference_predictions.csv")
    return p.parse_args()


def load_single_sample_json(path, cfg, cutoff):
    with open(path, "r") as f:
        sample = json.load(f)

    weather_format = sample.get("weather_format", "weekly_cumulative")

    weather_cols = []
    for v in cfg.weather_vars:
        if v not in sample["weather"]:
            raise ValueError(f"Missing weather variable in JSON: {v}")

        arr = np.asarray(sample["weather"][v], dtype=np.float32)

        if weather_format == "daily":
            agg = DEFAULT_WEATHER_AGG_RULES.get(v, "mean")
            arr = daily_to_cumulative_weekly(arr, agg=agg, week_len=7)
        elif weather_format in ("weekly", "weekly_cumulative"):
            pass
        else:
            raise ValueError(
                "weather_format must be 'daily', 'weekly', "
                "or 'weekly_cumulative'."
            )

        weather_cols.append(arr)

    lengths = [len(x) for x in weather_cols]
    if len(set(lengths)) != 1:
        raise ValueError(f"Weather variable lengths do not match after aggregation: {lengths}")

    weather = np.stack(weather_cols, axis=1).astype(np.float32)

    soil = []
    for v in cfg.soil_vars:
        if v not in sample["soil"]:
            raise ValueError(f"Missing soil variable in JSON: {v}")
        soil.append(float(sample["soil"][v]))

    soil = np.asarray(soil, dtype=np.float32)

    w_mean = np.asarray(cfg.w_mean, dtype=np.float32)
    w_std = np.asarray(cfg.w_std, dtype=np.float32)
    s_mean = np.asarray(cfg.s_mean, dtype=np.float32)
    s_std = np.asarray(cfg.s_std, dtype=np.float32)

    weather = np.where(np.isnan(weather), w_mean[None, :], weather)
    weather = (weather - w_mean[None, :]) / w_std[None, :]

    soil = np.where(np.isnan(soil), s_mean, soil)
    soil = (soil - s_mean) / s_std

    crop_map = {"corn": 0, "maize": 0, "soybean": 1, "soy": 1}
    crop = str(sample.get("crop", "corn")).strip().lower()
    crop_id = crop_map.get(crop, 0)

    t_eff = min(cutoff, weather.shape[0])

    return {
        "weather": torch.from_numpy(weather[:t_eff]).unsqueeze(0),
        "soil": torch.from_numpy(soil).unsqueeze(0),
        "crop_id": torch.tensor([crop_id], dtype=torch.long),
        "t_eff": t_eff,
    }


@torch.no_grad()
def main():
    args = parse_args()
    register_yield_autoclass()

    if args.input_file is None and args.single_sample_json is None:
        raise ValueError("Provide either --input_file or --single_sample_json.")

    if args.input_file is not None and args.single_sample_json is not None:
        raise ValueError("Use only one: --input_file OR --single_sample_json.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AutoModel.from_pretrained(args.hf_model_dir).to(device)
    model.eval()
    cfg = model.config

    if args.single_sample_json is not None:
        sample = load_single_sample_json(args.single_sample_json, cfg, args.cutoff)

        weather = sample["weather"].to(device)
        soil = sample["soil"].to(device)
        crop_id = sample["crop_id"].to(device)
        t_eff = sample["t_eff"]

        out = model(
            weather=weather,
            soil=soil,
            crop_id=crop_id,
            horizon_idx=t_eff,
            causal=True,
            return_sequence=False,
        )

        pred = float(out.predictions.item())

        Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([{
            "sample_idx": 0,
            "cutoff": int(args.cutoff),
            "y_pred": pred,
        }]).to_csv(args.output_csv, index=False)

        print(f"Predicted yield: {pred:.4f}")
        print(f"Saved inference prediction to {args.output_csv}")
        return

    ds = YieldDataset(
        data_file=args.input_file,
        weather_vars=cfg.weather_vars,
        soil_vars=cfg.soil_vars,
        split="all",
        seed=1234,
        crop=None,
        years=None,
        require_yield=False,
    )

    ds.set_normalization(
        cfg.w_mean,
        cfg.w_std,
        cfg.s_mean,
        cfg.s_std,
    )

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    rows = []
    sample_idx = 0

    for batch in loader:
        weather = batch["weather"].to(device)
        soil = batch["soil"].to(device)
        crop_id = batch["crop_id"].to(device)

        t_eff = min(args.cutoff, weather.size(1))

        out = model(
            weather=weather[:, :t_eff, :],
            soil=soil,
            crop_id=crop_id,
            horizon_idx=t_eff,
            causal=True,
            return_sequence=False,
        )

        for pred in out.predictions.detach().cpu().numpy().tolist():
            rows.append({
                "sample_idx": sample_idx,
                "cutoff": int(args.cutoff),
                "y_pred": float(pred),
            })
            sample_idx += 1

    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output_csv, index=False)

    print(f"Saved inference predictions to {args.output_csv}")


if __name__ == "__main__":
    main()