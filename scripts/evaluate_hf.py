import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import json
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoModel

from data.dataset import YieldDataset
from training.engine import evaluate
from hf.auto import register_yield_autoclass


def parse_int_list(x):
    return [int(v.strip()) for v in x.split(",") if v.strip()]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--hf_model_dir", required=True)
    p.add_argument("--test_file", required=True)
    p.add_argument("--cutoffs", default=None)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--output_csv", default="eval_predictions.csv")
    p.add_argument("--metrics_json", default="eval_metrics.json")
    p.add_argument(
        "--time_agg",
        default="weekly",
        choices=["weekly", "weekly_cumulative"],
    )
    return p.parse_args()


def main():
    args = parse_args()
    register_yield_autoclass()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AutoModel.from_pretrained(args.hf_model_dir).to(device)
    model.eval()
    cfg = model.config

    cutoffs = parse_int_list(args.cutoffs) if args.cutoffs else cfg.eval_cutoffs

    test_ds = YieldDataset(
        data_file=args.test_file,
        weather_vars=cfg.weather_vars,
        soil_vars=cfg.soil_vars,
        split="all",
        seed=1234,
        crop=None,
        years=None,
        time_agg=args.time_agg,
    )
    test_ds.set_normalization(
        cfg.w_mean,
        cfg.w_std,
        cfg.s_mean,
        cfg.s_std,
    )

    loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    metrics, rows = evaluate(
        model=model,
        loader=loader,
        device=device,
        y_mean=cfg.y_mean,
        y_std=cfg.y_std,
        cutoffs=cutoffs,
    )

    print(json.dumps(metrics, indent=2))

    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output_csv, index=False)

    with open(args.metrics_json, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved predictions to {args.output_csv}")
    print(f"Saved metrics to {args.metrics_json}")


if __name__ == "__main__":
    main()