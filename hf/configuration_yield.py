from transformers import PretrainedConfig


class YieldConfig(PretrainedConfig):
    model_type = "yield-weather-soil"

    def __init__(
        self,
        weather_vars=None,
        soil_vars=None,
        w_mean=None,
        w_std=None,
        s_mean=None,
        s_std=None,
        y_mean=None,
        y_std=None,
        K=None,
        W=None,
        S=None,
        train_cutoffs=None,
        eval_cutoffs=None,
        d_model=128,
        nhead=4,
        num_layers=4,
        dim_ff=256,
        dropout=0.3,
        pool="mean",
        use_crop=True,
        crop_emb_dim=8,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.weather_vars = weather_vars
        self.soil_vars = soil_vars

        self.w_mean = w_mean
        self.w_std = w_std
        self.s_mean = s_mean
        self.s_std = s_std
        self.y_mean = y_mean
        self.y_std = y_std

        self.K = K
        self.W = W
        self.S = S

        self.train_cutoffs = train_cutoffs
        self.eval_cutoffs = eval_cutoffs

        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_ff = dim_ff
        self.dropout = dropout
        self.pool = pool
        self.use_crop = use_crop
        self.crop_emb_dim = crop_emb_dim