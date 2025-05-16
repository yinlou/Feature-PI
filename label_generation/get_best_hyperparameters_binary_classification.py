import os
import json
import pandas as pd
import numpy as np
import lightgbm as lgb
import argparse
import datetime
import sys

myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + '/../')
from utils.logger import Logger
from copy import deepcopy


def get_n_estimators_by_cv(params, dtrain):
    validation_summary = lgb.cv(params,
                                dtrain,
                                num_boost_round=1000,
                                nfold=5,
                                stratified=True,
                                metrics=[metric],
                                callbacks=[
                                    lgb.early_stopping(stopping_rounds=50),
                                    lgb.log_evaluation(period=10)
                                ],
                                seed=seed)
    optimal_n_estimators = len(validation_summary["valid %s-mean" % metric])
    best_score = np.max(validation_summary["valid %s-mean" % metric])
    return optimal_n_estimators, best_score


def gridsearch_tuning(X, y):
    dtrain = lgb.Dataset(data=X, label=y)
    my_params = {
        'objective': 'binary',
        'learning_rate': 0.1,
        'max_depth': 10,
        'verbose': -1,
        'seed': 0,
        'deterministic': True,
        'n_jobs': n_jobs
    }

    num_leaves_choice = [255, 127, 63, 31, 15, 7, 3]
    cv_results = []
    for value in num_leaves_choice:
        my_params_temp = deepcopy(my_params)
        my_params_temp['num_leaves'] = value
        print(my_params_temp)
        n_estimators, best_score = get_n_estimators_by_cv(my_params_temp, dtrain)
        cv_results.append([my_params_temp, n_estimators, best_score])
    cv_results = sorted(cv_results, key=lambda x: x[2])
    best_params = cv_results[-1][0]
    best_params['n_estimators'] = cv_results[-1][1]
    best_score = cv_results[-1][2]
    return best_params, best_score


def process_cat(data, categorical_features):
    for feature in categorical_features:
        data[feature] = data[feature].astype('category')
        data[feature] = data[feature].cat.codes
        data[feature] = data[feature].astype('category')
    return data


def run_tuning(data_path):
    label = pd.read_csv(os.path.join(data_path, 'data_y.csv'))
    num_class = len(label[label.columns[0]].value_counts())
    if num_class != 2:
        return 0

    data = pd.read_csv(os.path.join(data_path, 'data_x.csv'))
    if len(data.columns) > 10000:
        return 0
    categorical_features = list(data.select_dtypes(exclude=np.number).columns)
    data = process_cat(data, categorical_features)

    best_params, best_score = gridsearch_tuning(data, label)
    best_params['importance_type'] = 'gain'

    # Save best parameters
    with open(os.path.join(data_path, 'best_params_s%d.json' % seed), 'w') as f:
        json.dump(best_params, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_jobs', type=int, default=4)
    parser.add_argument("-s", "--seed", help="seed of the random number generator (default: 7).", type=int, default=7)
    parser.add_argument("--directory", help="directory of binary datasets.", type=str, required=True)
    parser.add_argument("--metric", help="metric for scoring function", type=str, default='auc')
    args = parser.parse_args()
    n_jobs = args.n_jobs
    seed = args.seed
    directory = args.directory
    metric = args.metric
 
    logger = Logger(myPath, 'get_best_hyperparameters_binary_classification_s%d' % seed)
    for data_folder in sorted(os.listdir(directory)):
        data_path = os.path.join(directory, data_folder)
        if os.path.isdir(data_path):
            try:
                logger.log(data_path)
                start = datetime.datetime.now()
                logger.log("Start getting best_params of [%s]%s at %s." % (directory, data_folder, start))

                run_tuning(data_path)

                logger.log("Finish getting best_params of [%s]%s." % (directory, data_folder))
                logger.log("Time spent tuning: %s\n" % (datetime.datetime.now() - start))

            except Exception as e:
                import traceback

                logger.log(data_path)
                logger.log(traceback.format_exc())
