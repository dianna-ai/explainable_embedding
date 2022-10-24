import dataclasses
from dataclasses import dataclass
from typing import Optional, Union


@dataclass
class Config:
    experiment_name: str
    mask_selection_method: str
    mask_selection_range_min: float
    mask_selection_range_max: float
    mask_selection_negative_range_min: float
    mask_selection_negative_range_max: float
    mask_distance_power: float
    weight_range_normalization: bool
    distance_metric: str
    number_of_masks: Union[str, int]
    p_keep: Optional[float]
    feature_res: int

original_config_options = Config(
    experiment_name='default',
    mask_selection_method='',  # range, random, all
    mask_selection_range_min=0,  # 0-1
    mask_selection_range_max=0.1,  # 0-1
    mask_selection_negative_range_min=0.9,  # 0-1
    mask_selection_negative_range_max=1,  # 0-1
    mask_distance_power=1,  # 0.5, 1, 2 ... 100
    # Normalize weights to [0, 1] before taking dot product with masks.
    weight_range_normalization=False,  # True, False
    distance_metric='cosine',  # cosine, euclidian, manhatten?
    number_of_masks=1000,  # auto, [1, -> ]
    p_keep=0.5,  # None (auto), 0 - 1
    feature_res=8,  # [1, ->]
)

furthest_masks_config_options = dataclasses.replace(original_config_options,
                                                    mask_selection_range_min=.8,
                                                    mask_selection_range_max=1,
                                                    )

# number_of_masks_configs = [dataclasses.replace() for n_masks in some_range]
# p_keep_configs = [dataclasses.replace() for n_masks in some_range]
# mask_selection_configs = [dataclasses.replace() for n_masks in some_range]
# mask_weighting_configs = [dataclasses.replace() for n_masks in some_range]
