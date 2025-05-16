import os
import json
import pandas as pd
import numpy as np
import lightgbm as lgb
import argparse
import datetime

from sklearn.model_selection import KFold
from sklearn.inspection import permutation_importance
import sys

myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + '/../')
from utils.logger import Logger


def labeling_by_permutation(best_params, data, label, data_path):
    results_dict = {}
    columns = data.columns
    for column in columns:
        results_dict[column] = []
    X = data.values
    y = label.values.ravel()
    for i in range(3):
        skf = KFold(n_splits=5, shuffle=True, random_state=skf_random_states[i])

        cv_fold = 0
        n_repeats = 1
        for train_index, test_index in skf.split(X, y):
            cv_fold += 1
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            model = lgb.LGBMRegressor(**best_params)
            model.fit(X_train, y_train)
            r = permutation_importance(model, X_test, y_test, scoring=metric, n_repeats=n_repeats,
                                       random_state=permutation_random_states[i], n_jobs=n_jobs)

            for j in r.importances_mean.argsort()[::-1]:
                results_dict[columns[j]].append(r.importances_mean[j])


    with open(os.path.join(data_path, 'permutation_importance_v%d.json' % seed), 'w') as f:
        json.dump(results_dict, f)


def process_cat(data, categorical_features):
    for feature in categorical_features:
        data[feature] = data[feature].astype('category')
        data[feature] = data[feature].cat.codes
        data[feature] = data[feature].astype('category')
    return data


def run_labeling(data_path):
    label = pd.read_csv(os.path.join(data_path, 'data_y.csv'))

    data = pd.read_csv(os.path.join(data_path, 'data_x.csv'))
    if len(data.columns) > 10000:
        return 0

    categorical_features = list(data.select_dtypes(exclude=np.number).columns)
    data = process_cat(data, categorical_features)

    # Load best parameters
    with open(os.path.join(data_path, 'best_params_s%d.json' % seed), 'r') as f:
        best_params = json.load(f)

    # Start labeling
    labeling_by_permutation(best_params, data, label, data_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_jobs', type=int, default=4)
    parser.add_argument("-s", "--seed", help="seed of the random number generator (default: 7).", type=int, default=7)
    parser.add_argument("--directory", help="directory of regression datasets.", type=str, required=True)
    parser.add_argument("--metric", help="metric for scoring function", type=str,
                        default='neg_mean_absolute_percentage_error')
    args = parser.parse_args()
    n_jobs = args.n_jobs
    seed = args.seed
    directory = args.directory
    metric = args.metric

    skf_random_states = [i + (seed - 1) * 3 for i in [1, 2, 3]]
    permutation_random_states = [i + (seed - 1) * 3 for i in [1, 2, 3]]
    logger = Logger(myPath, 'generate_labels_regression_%s_s%d' % (metric, seed))

    for data_folder in sorted(os.listdir(directory)):
        data_path = os.path.join(directory, data_folder)
        if os.path.isdir(data_path):
            try:
                logger.log(data_path)
                start = datetime.datetime.now()
                logger.log("Start generating labels of [%s]%s at %s." % (directory, data_folder, start))

                run_labeling(data_path)

                logger.log("Finish generating labels of [%s]%s." % (directory, data_folder))
                logger.log("Labels file has been saved into labels_s%d.csv" % seed)
                logger.log("Time spent labeling:%s\n" % (datetime.datetime.now() - start))

            except Exception as e:
                import traceback

                logger.log(data_path)
                logger.log(traceback.format_exc())