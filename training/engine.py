import math
import random
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import r2_score

def train_one_epoch(model, loader, optimizer, device, y_mean, y_std, cutoffs):
    model.train()
    total_loss = 0.0
    n_total = 0

    se_norm = 0.0
    se_raw = 0.0
    n = 0

    for batch in loader:
        weather = batch["weather"].to(device)
        soil = batch["soil"].to(device)
        crop_id = batch["crop_id"].to(device)
        y = batch["yield"].to(device)

        t = random.choice(cutoffs)
        t_eff = min(t, weather.size(1))

        y_log = torch.log1p(torch.clamp(y, min=0.0))
        y_norm = (y_log - y_mean) / y_std

        yhat_norm = model(
            weather[:, :t_eff, :],
            soil,
            crop_id,
            horizon_idx=t_eff,
            causal=True,
            return_sequence=False,
        )

        loss = F.mse_loss(yhat_norm, y_norm)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        total_loss += loss.item() * y.size(0)
        n_total += y.size(0)

        with torch.no_grad():
            yhat = torch.expm1(yhat_norm * y_std + y_mean)
            se_norm += ((yhat_norm - y_norm) ** 2).sum().item()
            se_raw += ((yhat - y) ** 2).sum().item()
            n += y.numel()

    return {
        "loss_norm_mse": total_loss / max(n_total, 1),
        "norm_rmse": math.sqrt(se_norm / max(n, 1)),
        "rmse": math.sqrt(se_raw / max(n, 1)),
    }

@torch.no_grad()
def evaluate(model, loader, device, y_mean, y_std, cutoffs):
    model.eval()
    results = {}
    all_rows = []

    for t in sorted(cutoffs):
        y_true_all = []
        y_pred_all = []

        for batch in loader:
            weather = batch["weather"].to(device)
            soil = batch["soil"].to(device)
            crop_id = batch["crop_id"].to(device)
            y = batch["yield"].to(device)

            t_eff = min(t, weather.size(1))

            out = model(
                weather=weather[:, :t_eff, :],
                soil=soil,
                crop_id=crop_id,
                horizon_idx=t_eff,
                causal=True,
                return_sequence=False,
            )

            if hasattr(out, "predictions"):
                yhat = out.predictions
            else:
                yhat = torch.expm1(out * y_std + y_mean)

            y_true_all.extend(y.detach().cpu().numpy().tolist())
            y_pred_all.extend(yhat.detach().cpu().numpy().tolist())

        y_true = np.asarray(y_true_all, dtype=np.float32)
        y_pred = np.asarray(y_pred_all, dtype=np.float32)
        err = y_pred - y_true

        results[int(t)] = {
            "rmse": float(np.sqrt(np.mean(err ** 2))),
            "bias": float(np.mean(err)),
            "r2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else float("nan"),
        }

        for i, (yt, yp) in enumerate(zip(y_true, y_pred)):
            all_rows.append({
                "cutoff": int(t),
                "sample_idx": i,
                "y_true": float(yt),
                "y_pred": float(yp),
                "error": float(yp - yt),
            })

    return results, all_rows