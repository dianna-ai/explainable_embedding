# explainable_embedding
Experiments with regard to explanation of embedded spaces and multi modal models.

## Setup
Clone the `explainable_embedding` repository, `cd` into it and install the additional dependencies for these experiments:

```sh
pip install -r requirements.txt
```

## Choose experiments

In the `benchmarking` folder, you can edit `distance_benchmark.py` to set which experiments to run.
Since we are currently still actively using this repo for running experiments, the file you downloaded may not be set to run all experiments.
The code at the bottom of the file should speak for itself.
To run all experiments used for the paper, `import all_configs from .distance_benchmark_configs` and modify the loop to go over `all_configs`.

## Run experiments

Run experiments with

```sh
python -m benchmarking.distance_benchmark
```

A directory called `output` will be created with the results.
