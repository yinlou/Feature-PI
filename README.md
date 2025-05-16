# Feature-PI

This is the code repository for Feature-PI.

## Install Dependencies

Prepare the python environment by executing the scripts below:

```bash
$ cd Feature-PI
$ pip install -r requirements.txt
```

For Mac users, you might want to install `libomp` as required by LightGBM:
```bash
brew install libomp
```

## Download the whole Feature-PI data from HuggingFace

Use the following command to download the whole Feature-PI data from HuggingFace.

```bash
$ cd Feature-PI
$ mkdir data
$ export HF_HUB_DOWNLOAD_TIMEOUT=300
$ huggingface-cli download yinlou/Feature-PI --repo-type dataset --local-dir data
```

This should download the whole data repo from HuggingFace. The data repo is very large (60GB+) and contains lots of `.csv` and `.json` files. It is expected to take couple of hours to finish downloading.  

## Generate Labels (Permutation Feature Importance Scores)

The permutation feature importance scores are pre-computed using random seed 1 to 5 and can be found in each folder (representing a dataset) as `permutation_importance_s[1-5].json`.

To manually generate labels for binary classification datasets with random seed 2, use the following command under `Feature-PI` directory:

```bash
$ ./scripts/generate_labels_binary_classification.sh data/training/binary_classification 2
```

This will create in each folder under `data/training/binary_classification` a json file named `permutation_importance_s2.json` and it should be the same as `permutation_importance_v2.json`. It contains 15 trials of permutation feature importance and is used as labels for learning to estimate feature importance for that dataset.

Similarly, for regression datasets, you can use the following command:
```bash
$ ./scripts/generate_labels_regression.sh data/training/regression 1
```

This command creates in each folder under `data/training/regression` a json file named `permutation_importance_s1.json`. It contains 15 trials of permutation feature importance and is used as labels for learning to estimate feature importance for that dataset.

## Evaluate Feature Importance Socres on Benchmark Set

Suppose we want to evaluate PI-ensemble for `"[UCI]Arrhythmia` in the benchmark set, we could use the following command:

```bash
python benchmark/eval.py -d ./ -f "[UCI]Arrhythmia" -t binary_classification -e pi_ensemble
```

This will go into each cross validation fold under `"[UCI]Arrhythmia` and generate a file called `pi_ensemble_eval_result.json`, which contains AUC (for binary classification problems) on 5%, 10%, 15%, 20%, and 100% of feature subset, ranked according to feature importance.
