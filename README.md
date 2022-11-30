# explainable_embedding
Experiments with regard to explanation of embedded spaces and multi modal models.

## Setup
First make sure you have a Python (<3.11) environment with dianna.
Currently, we use a development branch that you can install with:

```sh
git clone https://github.com/dianna-ai/dianna.git
cd dianna
git checkout 279-embeddings
pip install -e .
```

Then clone the `explainable_embedding` repository, `cd` into it and install the additional dependencies for these experiments:

```sh
pip install -r requirements.txt
```

## Run experiments

Run all experiments with

```sh
python benchmarking/distance_benchmark.py
```

A directory called `output` will be created with the results.
