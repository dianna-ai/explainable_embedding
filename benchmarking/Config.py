from dataclasses import dataclass
from typing import Union, Optional


@dataclass
class Config:
    experiment_name: str
    mask_selection_range_min: float
    mask_selection_range_max: float
    mask_selection_negative_range_min: float
    mask_selection_negative_range_max: float
    number_of_masks: Union[str, int]
    p_keep: Optional[float]
    feature_res: int
    random_seed: int
    manual_central_value: Optional[float]


original_config_options = Config(
    experiment_name='default',
    mask_selection_range_min=0,  # 0-1
    mask_selection_range_max=0.1,  # 0-1
    mask_selection_negative_range_min=0.9,  # 0-1
    mask_selection_negative_range_max=1,  # 0-1
    number_of_masks=1000,  # auto, [1, -> ]
    p_keep=0.5,  # None (auto), 0 - 1
    feature_res=8,  # [1, ->]
    random_seed=0,
    manual_central_value=0,
)
