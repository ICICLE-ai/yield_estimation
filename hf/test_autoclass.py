import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import torch
from transformers import AutoConfig, AutoModel

from hf.configuration_yield import YieldConfig
from hf.modeling_yield import YieldForRegression


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--hf_model_dir", required=True)
    return p.parse_args()


def main():
    args = parse_args()

    AutoConfig.register("yield-weather-soil", YieldConfig)
    AutoModel.register(YieldConfig, YieldForRegression)

    config = AutoConfig.from_pretrained(args.hf_model_dir)
    model = AutoModel.from_pretrained(args.hf_model_dir)

    weather = torch.randn(2, config.K, config.W)
    soil = torch.randn(2, config.S)
    crop_id = torch.zeros(2, dtype=torch.long)

    model.eval()
    with torch.no_grad():
        out = model(
            weather=weather,
            soil=soil,
            crop_id=crop_id,
            horizon_idx=16,
        )

    print(type(config))
    print(type(model))
    print(out.predictions)


if __name__ == "__main__":
    main()