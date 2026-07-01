import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch import nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import ModelOutput
from dataclasses import dataclass

from hf.configuration_yield import YieldConfig
from models.unimodal_ws_crossattn import UnimodalWS_CrossAttn_TemporalTF


@dataclass
class YieldModelOutput(ModelOutput):
    loss: torch.Tensor | None = None
    logits: torch.Tensor | None = None
    predictions: torch.Tensor | None = None


class YieldForRegression(PreTrainedModel):
    config_class = YieldConfig
    base_model_prefix = "yield_model"

    def __init__(self, config: YieldConfig):
        super().__init__(config)

        self.yield_model = UnimodalWS_CrossAttn_TemporalTF(
            w_dim=config.W,
            soil_dim=config.S,
            d_model=config.d_model,
            nhead=config.nhead,
            num_layers=config.num_layers,
            dim_ff=config.dim_ff,
            dropout=config.dropout,
            use_crop=config.use_crop,
            crop_emb_dim=config.crop_emb_dim,
            max_weeks=max(32, config.K),
            pool=config.pool,
        )

        self.post_init()

    def forward(
        self,
        weather,
        soil,
        crop_id,
        labels=None,
        horizon_idx=None,
        causal=True,
        return_sequence=False,
        return_dict=True,
    ):
        if horizon_idx is None:
            horizon_idx = weather.shape[1]

        logits = self.yield_model(
            weather,
            soil,
            crop_id,
            horizon_idx=horizon_idx,
            causal=causal,
            return_sequence=return_sequence,
        )

        y_mean = torch.tensor(self.config.y_mean, device=logits.device, dtype=logits.dtype)
        y_std = torch.tensor(self.config.y_std, device=logits.device, dtype=logits.dtype)

        predictions = torch.expm1(logits * y_std + y_mean)

        loss = None
        if labels is not None:
            labels_log = torch.log1p(torch.clamp(labels, min=0.0))
            labels_norm = (labels_log - y_mean) / y_std
            loss = nn.functional.mse_loss(logits, labels_norm)

        if not return_dict:
            return (loss, logits, predictions)

        return YieldModelOutput(
            loss=loss,
            logits=logits,
            predictions=predictions,
        )