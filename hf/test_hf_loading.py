import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import torch

from hf.modeling_yield import YieldForRegression


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--hf_model_dir", required=True)
    return p.parse_args()


def main():
    args = parse_args()

    model = YieldForRegression.from_pretrained(args.hf_model_dir)
    model.eval()

    cfg = model.config

    weather = torch.randn(2, cfg.K, cfg.W)
    soil = torch.randn(2, cfg.S)
    crop_id = torch.zeros(2, dtype=torch.long)

    with torch.no_grad():
        out = model(
            weather=weather,
            soil=soil,
            crop_id=crop_id,
            horizon_idx=16,
        )

    print("logits shape:", out.logits.shape)
    print("predictions shape:", out.predictions.shape)
    print("predictions:", out.predictions)


if __name__ == "__main__":
    main()