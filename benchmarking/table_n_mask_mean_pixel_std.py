""" Creates a table with for each number of masks, the mean of standard deviations per pixel. Output as string and as LaTeX.

Usage:

python table_n_mask_mean_pixel_std.py ../n_masks_output_sample

where ../n_masks_output_sample contains all (n_mask) run output folders.

Example output:
Number of masks Mean STD per pixel
                        bee vs fly flower vs car dogcar vs car
            100           0.105841      0.107238      0.105069
            500           0.047055      0.049860      0.040480
           2000           0.021234      0.024249      0.024703
\begin{tabular}{rrrr}
\toprule
Number of masks & \multicolumn{3}{r}{Mean STD per pixel} \\
 & bee vs fly & flower vs car & dogcar vs car \\
\midrule
100 & 0.105841 & 0.107238 & 0.105069 \\
500 & 0.047055 & 0.049860 & 0.040480 \\
2000 & 0.021234 & 0.024249 & 0.024703 \\
\bottomrule
\end{tabular}
"""

import argparse
from pathlib import Path
from typing import Generator, Iterable

import numpy as np
from pandas import DataFrame


def parse_path_from_args() -> Path:
    parser = argparse.ArgumentParser(description="Process some paths.")
    parser.add_argument('data_folder', type=Path,
                        help='The data folder path containing the outputs of the distance benchmark script.'
                        )
    args = parser.parse_args()
    return args.data_folder


def load_map(saliency_map_path:Path) -> np.ndarray:
    data = np.load(saliency_map_path)
    assert data.shape == (1, 224, 224, 1), \
        f'Expecting shape (batch, width, height, channels) with both batch and channels = 1. Actual shape: {data.shape}'
    return data[0, :, :, 0]


def main():
    data_path = parse_path_from_args()

    case_type = ['image_captioning', 'image_vs_image'][1]
    n_masks_vals = [100, 500, 2000]  # available are [10, 50, 100, 200, 500, 1000, 2000, 5000]

    first_level_index = 'Mean STD per pixel'
    table = DataFrame({('Number of masks', ''): n_masks_vals,
                       (first_level_index, 'bee vs fly'): [get_mean(data_path, case_type, 'bee_vs_fly', n_mask) for
                                                           n_mask in n_masks_vals],
                       (first_level_index, 'flower vs car'): [get_mean(data_path, case_type, 'flower_vs_car', n_mask)
                                                              for n_mask in n_masks_vals],
                       (first_level_index, 'dogcar vs car'): [get_mean(data_path, case_type, 'dogcar_vs_car', n_mask)
                                                              for n_mask in n_masks_vals], }, )
    print(table.to_string(index=False))
    print(table.to_latex(index=False))


def get_mean(data_path: Path, case_type : str, case: str, n_masks: int) -> float:
    file_paths = get_saliency_file_paths(data_path, case_type, case, n_masks)
    saliencies = np.stack([load_map(p) for p in file_paths])
    std_per_pixel = saliencies.std(axis=0)
    return std_per_pixel.mean()


def get_saliency_file_paths(data_path: Path, case_type: str, case:str, n_mask:int) -> Iterable[Path]:
    for saliency_file_path in data_path.glob(f'*/{case_type}/{case}/saliency.npy'):
        run_name = saliency_file_path.relative_to(data_path).parts[0]
        _n, _masks, _sweep, n_masks, _seed, seed, run_id = run_name.split('_')
        if int(n_masks) == n_mask:
            yield saliency_file_path


if __name__ == "__main__":
    main()
