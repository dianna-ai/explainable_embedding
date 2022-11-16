from dataclasses import dataclass
from typing import Union, Optional

from dataclass_wizard import YAMLWizard
from yaml.constructor import ConstructorError


@dataclass
class Config(YAMLWizard):
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

    @classmethod
    def load(cls, path):
        try:
            return cls.from_yaml_file(path)
        except ConstructorError as e:
            with open(path, 'r') as f:
                return cls.from_yaml('\n'.join(f.read().split('\n')[1:]))


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
