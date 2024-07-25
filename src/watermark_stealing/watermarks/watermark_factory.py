from typing import Optional

from transformers import PreTrainedTokenizer

from watermark_stealing.config import MetaConfig, WatermarkConfig, WatermarkScheme
from watermark_stealing.watermarks.base_watermark import BaseWatermark
from watermark_stealing.watermarks.kgw_watermark import KgwWatermark


def get_watermark(
    meta_cfg: MetaConfig,
    watermark_cfg: WatermarkConfig,
    tokenizer: Optional[PreTrainedTokenizer] = None,
    model_name: Optional[str] = None,
) -> BaseWatermark:
    if watermark_cfg.scheme == WatermarkScheme.KGW:
        if tokenizer is None or model_name is None:
            raise ValueError("KGW watermark requires tokenizer and model_name to be passed")
        return KgwWatermark(meta_cfg, watermark_cfg, tokenizer, model_name)
    else:
        raise ValueError(f"Unknown watermark scheme: {watermark_cfg.scheme}")
