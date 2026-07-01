import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import json
import random
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModel

from config.config import TrainConfig
from data.dataset import YieldDataset
from data.preprocessing import compute_x_stats, compute_y_log_stats
from training.engine import evaluate
from hf.configuration_yield import YieldConfig
from hf.auto import register_yield_autoclass


def parse_list(x):
    return [v.strip() for v in x.split(",") if v.strip()]


def parse_int_list(x):
    return [int(v.strip()) for v in x.split(",") if v.strip()]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_file", required=True)
    p.add_argument("--val_file", default=None)
    p.add_argument("--test_file", default=None)
    p.add_argument("--weather_vars", required=True)
    p.add_argument("--soil_vars", required=True)
    p.add_argument("--crop", default=None)
    p.add_argument("--years", default=None)
    p.add_argument("--train_cutoffs", default="4,8,12,16,22")
    p.add_argument("--eval_cutoffs", default=None)
    p.add_argument("--out_dir", default="outputs/yield_hf_model")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=1234)
    return p.parse_args()


def train_one_epoch_hf(model, loader, optimizer, device, cutoffs):
    model.train()

    total_loss = 0.0
    n_total = 0
    se_raw = 0.0
    n = 0

    for batch in loader:
        weather = batch["weather"].to(device)
        soil = batch["soil"].to(device)
        crop_id = batch["crop_id"].to(device)
        labels = batch["yield"].to(device)

        t = random.choice(cutoffs)
        t_eff = min(t, weather.size(1))

        out = model(
            weather=weather[:, :t_eff, :],
            soil=soil,
            crop_id=crop_id,
            labels=labels,
            horizon_idx=t_eff,
            causal=True,
            return_sequence=False,
        )

        loss = out.loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        n_total += labels.size(0)

        with torch.no_grad():
            pred = out.predictions
            se_raw += ((pred - labels) ** 2).sum().item()
            n += labels.numel()

    rmse = (se_raw / max(n, 1)) ** 0.5

    return {
        "loss": total_loss / max(n_total, 1),
        "rmse": rmse,
    }


def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def main():
    args = parse_args()
    register_yield_autoclass()

    cfg = TrainConfig(
        mode="train_eval",
        train_file=Path(args.train_file),
        val_file=Path(args.val_file) if args.val_file else None,
        test_file=Path(args.test_file) if args.test_file else None,
        weather_vars=parse_list(args.weather_vars),
        soil_vars=parse_list(args.soil_vars),
        crop=args.crop,
        years=args.years,
        train_cutoffs=parse_int_list(args.train_cutoffs),
        eval_cutoffs=parse_int_list(args.eval_cutoffs) if args.eval_cutoffs else parse_int_list(args.train_cutoffs),
        out_dir=Path(args.out_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
    )

    cfg.validate()
    cfg.save()

    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = YieldDataset(
        data_file=cfg.train_file,
        weather_vars=cfg.weather_vars,
        soil_vars=cfg.soil_vars,
        split="train" if cfg.val_file is None else "all",
        seed=cfg.seed,
        crop=cfg.crop,
        years=cfg.years,
        split_strategy=cfg.split_strategy,
        val_split=cfg.val_split,
        test_split=cfg.test_split,
    )

    if cfg.val_file:
        val_ds = YieldDataset(
            data_file=cfg.val_file,
            weather_vars=cfg.weather_vars,
            soil_vars=cfg.soil_vars,
            split="all",
            seed=cfg.seed,
            crop=cfg.crop,
            years=cfg.years,
        )
    else:
        val_ds = YieldDataset(
            data_file=cfg.train_file,
            weather_vars=cfg.weather_vars,
            soil_vars=cfg.soil_vars,
            split="val",
            seed=cfg.seed,
            crop=cfg.crop,
            years=cfg.years,
            split_strategy=cfg.split_strategy,
            val_split=cfg.val_split,
            test_split=cfg.test_split,
        )

    w_mean, w_std, s_mean, s_std = compute_x_stats(train_ds, seed=cfg.seed)
    y_mean, y_std = compute_y_log_stats(train_ds)

    train_ds.set_normalization(w_mean, w_std, s_mean, s_std)
    val_ds.set_normalization(w_mean, w_std, s_mean, s_std)

    b0 = train_ds[0]
    K, W = b0["weather"].shape
    S = b0["soil"].shape[0]

    hf_config = YieldConfig(
        weather_vars=cfg.weather_vars,
        soil_vars=cfg.soil_vars,
        w_mean=w_mean.tolist(),
        w_std=w_std.tolist(),
        s_mean=s_mean.tolist(),
        s_std=s_std.tolist(),
        y_mean=float(y_mean),
        y_std=float(y_std),
        K=int(K),
        W=int(W),
        S=int(S),
        train_cutoffs=cfg.train_cutoffs,
        eval_cutoffs=cfg.eval_cutoffs,
        d_model=cfg.d_model,
        nhead=cfg.nhead,
        num_layers=cfg.num_layers,
        dim_ff=cfg.dim_ff,
        dropout=cfg.dropout,
        pool=cfg.pool,
        use_crop=cfg.use_crop,
        crop_emb_dim=cfg.crop_emb_dim,
    )

    model = AutoModel.from_config(hf_config).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0)

    best_val = float("inf")
    best_metrics = None

    for epoch in range(1, cfg.epochs + 1):
        train_metrics = train_one_epoch_hf(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            cutoffs=cfg.train_cutoffs,
        )

        val_metrics_by_t, _ = evaluate(
            model=model,
            loader=val_loader,
            device=device,
            y_mean=y_mean,
            y_std=y_std,
            cutoffs=cfg.eval_cutoffs,
        )

        val_rmse = sum(m["rmse"] for m in val_metrics_by_t.values()) / len(val_metrics_by_t)

        print(
            f"Epoch {epoch:03d} | "
            f"train_rmse={train_metrics['rmse']:.4f} | "
            f"val_rmse={val_rmse:.4f}"
        )

        if val_rmse < best_val:
            best_val = val_rmse
            best_metrics = {
                "epoch": epoch,
                "val_rmse": val_rmse,
                "val_by_cutoff": val_metrics_by_t,
            }

            model.save_pretrained(cfg.out_dir)
            hf_config.save_pretrained(cfg.out_dir)
            save_json(cfg.out_dir / "metrics.json", best_metrics)

            print(f"Saved best HF model to {cfg.out_dir}")

    print("Best metrics:")
    print(json.dumps(best_metrics, indent=2))

    if cfg.test_file is not None:
        print("\nRunning final test using best HF model...")

        best_model = AutoModel.from_pretrained(cfg.out_dir).to(device)
        best_model.eval()

        test_ds = YieldDataset(
            data_file=cfg.test_file,
            weather_vars=cfg.weather_vars,
            soil_vars=cfg.soil_vars,
            split="all",
            seed=cfg.seed,
            crop=cfg.crop,
            years=cfg.years,
        )

        test_ds.set_normalization(w_mean, w_std, s_mean, s_std)

        test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0)

        test_metrics, test_rows = evaluate(
            model=best_model,
            loader=test_loader,
            device=device,
            y_mean=y_mean,
            y_std=y_std,
            cutoffs=cfg.eval_cutoffs,
        )

        print("Final test metrics:")
        print(json.dumps(test_metrics, indent=2))

        save_json(cfg.out_dir / "test_metrics.json", test_metrics)
        pd.DataFrame(test_rows).to_csv(cfg.out_dir / "test_predictions.csv", index=False)

        print(f"Saved final test outputs to {cfg.out_dir}")


if __name__ == "__main__":
    main()