import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 128, dropout: float = 0.05):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden//2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden//2, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim)
        )

    def forward(self, x):
        return self.net(x)

class UnimodalWS_CrossAttn_TemporalTF(nn.Module):
    def __init__(
        self,
        w_dim: int,
        soil_dim: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_ff: int = 256,
        dropout: float = 0.05,
        use_crop: bool = True,
        crop_emb_dim: int = 8,
        max_weeks: int = 32,
        pool: str = "mean", # Changed default to mean
        farm_emb_dim: int = 8,
        horizon_emb_dim: int = 16,
    ):
        super().__init__()
        assert pool in ["last", "cls", "mean"]
        self.pool = pool
        self.use_crop = use_crop
        self.max_weeks = max_weeks

        crop_in = crop_emb_dim if use_crop else 0
        if use_crop:
            self.crop_emb = nn.Embedding(2, crop_emb_dim)

        self.horizon_emb = nn.Embedding(max_weeks + 1, horizon_emb_dim)
        self.horizon_proj = nn.Linear(horizon_emb_dim, d_model)
        
        # Hard Positional Signal: Week ID projection
        self.week_proj = nn.Linear(1, d_model) 

        self.weather_enc = MLP(w_dim + crop_in, d_model, hidden=d_model, dropout=dropout)
        self.soil_enc = MLP(soil_dim + crop_in, d_model, hidden=d_model, dropout=dropout)

        self.ws_cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.ws_ln = nn.LayerNorm(d_model)

        self.pos_emb = nn.Parameter(torch.zeros(1, max_weeks + 1, d_model))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

        if pool == "cls":
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
            nn.init.trunc_normal_(self.cls_token, std=0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.temporal_tf = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        # Head now takes [Pooled Sequence + Horizon Embedding + Soil Skip Connection]
        self.head = nn.Sequential(
            nn.LayerNorm(d_model + horizon_emb_dim + d_model), 
            nn.Linear(d_model + horizon_emb_dim + d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(
        self,
        weather,
        soil,
        crop_id,
        farm_field_encoded=None, 
        week_mask=None,
        causal=False,
        return_sequence=False,
        horizon_idx=None,
    ):
        B, t, W = weather.shape
        device = weather.device

        # 1. Horizon Processing
        if horizon_idx is None: horizon_idx = torch.tensor(t, device=device).expand(B)
        if not torch.is_tensor(horizon_idx): horizon_idx = torch.tensor(horizon_idx, device=device).expand(B)
        
        h_idx = horizon_idx.long().clamp(min=1, max=self.max_weeks)
        h_emb = self.horizon_emb(h_idx)
        h_tok = self.horizon_proj(h_emb) # [B, D]

        # 2. Input Encoding
        if self.use_crop:
            c = self.crop_emb(crop_id)
            weather_in = torch.cat([weather, c[:, None, :].expand(-1, t, -1)], dim=-1)
            soil_in = torch.cat([soil, c], dim=-1)
        else:
            weather_in, soil_in = weather, soil

        w_tok = self.weather_enc(weather_in)
        s_encoded = self.soil_enc(soil_in) # [B, D] - Saved for skip connection later
        
        # 3. Cross Attention & Horizon 
        # attn_out, _ = self.ws_cross_attn(query=w_tok, key=s_encoded[:, None, :], value=s_encoded[:, None, :])
        # fused = self.ws_ln(w_tok + attn_out)

        fused = self.ws_ln(w_tok + s_encoded[:, None, :])
        fused = fused + h_tok[:, None, :] # Horizon aware tokens

        # 4. Temporal Transformer
        # Add Hard Week Signal + Learned Pos Emb
        weeks = torch.arange(1, t + 1, device=device).float() / self.max_weeks
        week_signal = self.week_proj(weeks[None, :, None].expand(B, -1, -1))
        
        x = fused + week_signal + self.pos_emb[:, :t, :]
        
        if self.pool == "cls":
            x = torch.cat([self.cls_token.expand(B, -1, -1), x], dim=1)
            t_idx = t + 1
        else: t_idx = t

        src_key_padding_mask = ~week_mask.bool() if week_mask is not None else None
        #h = self.temporal_tf(x, src_key_padding_mask=src_key_padding_mask)

        attn_mask = None
        if causal:
            L = x.size(1)
            attn_mask = torch.triu(
                torch.ones(L, L, device=device, dtype=torch.bool),
                diagonal=1,
            )

        h = self.temporal_tf(
            x,
            mask=attn_mask,
            src_key_padding_mask=src_key_padding_mask,
        )

        # 5. Global Mean Pooling
        if self.pool == "mean":
            if week_mask is not None:
                mask = week_mask.unsqueeze(-1).float()
                pooled = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            else:
                pooled = h.mean(dim=1)
        elif self.pool == "cls":
            pooled = h[:, 0, :]
        else: # "last"
            pooled = h[:, -1, :]

        # 6. Final Fusion (Pooled Sequence + Horizon + Soil Skip)
        out = torch.cat([pooled, h_emb, s_encoded], dim=-1)
        return self.head(out).squeeze(-1)

