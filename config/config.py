from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
import json

@dataclass
class TrainConfig:
    mode: str = "train_eval"  # train_eval | train | eval | predict

    train_file: Path | None = None
    val_file: Path | None = None
    test_file: Path | None = None
    predict_file: Path | None = None
    checkpoint_path: Path | None = None
    output_csv: Path | None = None
    single_sample_json: Path | None = None

    weather_vars: list[str] | None = None
    soil_vars: list[str] | None = None

    crop_col: str = "crop"
    yield_col: str = "yield"
    field_col: str = "farm_field"
    year_col: str = "year"

    years: str | None = None
    crop: str | None = None
    time_agg: str = "weekly_cumulative"

    train_cutoffs: list[int] | None = None
    eval_cutoffs: list[int] | None = None
    predict_cutoff: int | None = None

    d_model: int = 64
    nhead: int = 4
    num_layers: int = 4
    dim_ff: int = 128
    dropout: float = 0.4
    pool: str = "last"
    use_crop: bool = True
    crop_emb_dim: int = 8

    epochs: int = 10
    lr: float = 1e-4
    batch_size: int = 64
    weight_decay: float = 1e-3
    seed: int = 1234
    early_stop_patience: int = 3

    split_strategy: str = "field"  # field | random | none
    val_split: float = 0.2
    test_split: float = 0.2

    expt_name: str = "test"
    out_dir: Path = Path("outputs")
    log_dir: Path = Path("runs")

    def validate(self) -> None:
        if self.mode in {"train", "train_eval"}:
            if self.train_file is None:
                raise ValueError("train_file is required for training.")
            if not self.weather_vars:
                raise ValueError("weather_vars is required for training.")
            if not self.soil_vars:
                raise ValueError("soil_vars is required for training.")

        if self.mode in {"eval", "predict"} and self.checkpoint_path is None:
            raise ValueError("checkpoint_path is required for eval/predict.")

        if self.mode == "eval" and self.test_file is None:
            raise ValueError("test_file is required for eval.")

        if self.mode == "predict":
            if self.predict_file is None and self.single_sample_json is None:
                raise ValueError("predict_file or single_sample_json is required.")
            if self.predict_cutoff is None:
                raise ValueError("predict_cutoff is required for predict.")

    def save(self) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        with open(self.out_dir / "config.json", "w") as f:
            json.dump(asdict(self), f, indent=2, default=str)