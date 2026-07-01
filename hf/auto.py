from transformers import AutoConfig, AutoModel
from hf.configuration_yield import YieldConfig
from hf.modeling_yield import YieldForRegression


def register_yield_autoclass():
    try:
        AutoConfig.register("yield-weather-soil", YieldConfig)
    except ValueError:
        pass

    try:
        AutoModel.register(YieldConfig, YieldForRegression)
    except ValueError:
        pass