"""Main script to run missingness shift experiments.

Example command for synthetic datasets:
python missingness.py --dataset bernoulli1 \
    --num_beta_samples 1 --num_missingness_samples 20 \
    --linreg --xgb --nn --tag tag_name --imputation --verbose

Example command for semi-synthetic datasets: (need to create clean datasets first)
python missingness.py --dataset adult --synth_y \
    --num_beta_samples 5 --num_missingness_samples 20 \
    --linreg --xgb --nn --tag tag_name --imputation --verbose
"""


import argparse
import os
import sys
import time

from collections import defaultdict
import pickle
import random
from tqdm import tqdm

import numpy as np
import numpy.random as npr
import pandas as pd
import scipy
from scipy.special import expit

import sklearn.neighbors._base  # uncomment depending on package versions
sys.modules['sklearn.neighbors.base'] = sklearn.neighbors._base

from sklearn.linear_model import LinearRegression
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor

from missingpy import MissForest


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


def get_raw_data(dataset_name, rng):
    y = None
    if dataset_name == 'bernoulli1':
        N = 10000
        Z = npr.binomial(1, 0.5, size=(N, 1))
        X1 = Z 
        X2 = Z
        X = np.concatenate([X1, X2], axis=1)
        Y = Z + npr.multivariate_normal([0], [[1]], size=(N))
    elif dataset_name == 'bernoulli2':
        N = 10000
        X1 = npr.binomial(1, 0.5, size=(N, 1))
        X2 = expit(2 * X1 + npr.multivariate_normal([0], [[1]], size=(N)))
        X = np.concatenate([X1, X2], axis=1)
        Y = X1 - X2 + npr.multivariate_normal([0], [[1]], size=(N))
    elif dataset_name == 'eicu':
        X = pd.read_csv('data/eicu/48h_features.csv', index='ID').drop('hospital_id', axis=1)
        Y = pd.read_csv('data/eicu/48h_features.csv', index='ID').drop('hospital_id', axis=1)
        X = X.values
        Y = Y.values
    else:
        # get dataset from set of preprocessed datasets
        fname = f'data/preprocessed/{dataset_name}_clean.csv'
        df = pd.read_csv(fname, index_col=False)
        X = df.drop('label', axis=1)
        y = df[['label']]
        y = (y - y.mean()) / y.std()

        # drop low variance columns
        lowvar_cols = (X.std(axis=0) <= 0.05).reset_index()
        lowvar_cols = lowvar_cols[lowvar_cols[0]]['index'].tolist()
        X = X.drop(lowvar_cols, axis=1)

        X = X.values
        y = y.values

    if y is not None:
        X, y = shuffle(X, y)
    else:
        X = shuffle(X)    

    return X, y


def get_data_splits(X, y, ratios=(0.4, 0.1, 0.4, 0.1)):
    assert(len(ratios) == 4)
    N, D = X.shape
    sizes = [int(N * r) for r in ratios]
    cutoffs = np.cumsum(sizes)
    print('Data splits: ', cutoffs, ratios)

    Xs_train, ys_train = X[:cutoffs[0]], y[:cutoffs[0]]
    Xs_test, ys_test = X[cutoffs[0]:cutoffs[1]], y[cutoffs[0]:cutoffs[1]]
    Xt_train, yt_train = X[cutoffs[1]:cutoffs[2]], y[cutoffs[1]:cutoffs[2]]
    Xt_test, yt_test = X[cutoffs[2]:cutoffs[3]], y[cutoffs[2]:cutoffs[3]]

    return Xs_train, Xs_test, Xt_train, Xt_test, ys_train, ys_test, yt_train, yt_test


def get_synth_y(X, synth_beta):
    y = X.dot(synth_beta)
    return y


def get_X_tilde(missing_rates, X, rng):
    X_tilde = X.copy()
    N, D = X.shape
    for i, m in enumerate(missing_rates.flatten()):
        mask = rng.binomial(1, 1 - m, N)
        X_tilde[:, i] = X_tilde[:, i] * mask

    return X_tilde


def compute_r(X_s, X_t):
    # estimate r from relative proportion of positives
    ps = np.maximum((X_s != 0).sum(axis=0) / X_s.shape[0], 1e-9)
    pt = (X_t != 0).sum(axis=0) / X_t.shape[0]
    r = 1 - (pt / ps)
    return r


def add_bias(X_t):
    Nt, Dt = X_t.shape
    one = np.ones((Nt, 1))
    X_t = np.concatenate((one, X_t), axis=1)  # concatenate 1 for intercept
    return X_t


def get_adjusted_linear_model(X_s, y_s, X_t, fit_intercept=True, r=None, version='both'):  # options: s, t, both
    if r is None:
        r = compute_r(X_s, X_t)
        if r is None:
            print('ps = 0')
            return None, None

    if fit_intercept:
        X_t = add_bias(X_t)
        X_s = add_bias(X_s)
        r = np.concatenate([[0], r])[:, np.newaxis]
    else:
        r = r[:, np.newaxis]

    Xy_t = (1 - r) * np.dot(X_s.T, y_s) / len(y_s)

    # estimated from target data
    XX_t1 = np.dot(X_t.T, X_t) / len(X_t)

    # estimated from source data
    XX_t2 = np.dot(X_s.T, X_s) / len(X_s)
    multiplier = (1 - r).dot((1 - r).T)
    np.fill_diagonal(multiplier, 1-r, wrap=False)
    assert(multiplier.shape == XX_t2.shape)
    XX_t2 = np.multiply(XX_t2, multiplier)

    # combine estimates
    weight1 = float(len(X_t)) / float(len(X_t) + len(X_s))
    weight2 = float(len(X_s)) / float(len(X_t) + len(X_s))
    assert(weight1 + weight2 == 1)
    XX_t_combined = np.multiply(weight1, XX_t1) + np.multiply(weight2, XX_t2)

    XX_t = {
        't': XX_t1,
        's': XX_t2,
        'both': XX_t_combined,
    }[version]

    try:
        u, s, v = np.linalg.svd(XX_t)
        if (np.any(s == 0)):
            return None, None
        XX_t_inv = np.dot(v.transpose(), np.dot(np.diag(1.0 / s), u.transpose()))
        beta_t = XX_t_inv.dot(Xy_t)
    except Exception as e:
        print(e)
        return None, None

    def adjusted_linear_model(newx):
        pred = add_bias(newx).dot(beta_t)
        return pred

    return adjusted_linear_model, beta_t


def transform_Xs_to_Xt(X_s, X_t, rng, r=None, loose=False):
    if r is None:
        r = compute_r(X_s, X_t)

    if loose:
        r = np.maximum(r, 0)

    new_Xt = get_X_tilde(r, X_s, rng)
    return new_Xt


def generate_missing_rates(D, rng):
    missing_rates = []

    # sample source missingness
    # sample target missingness where ms < mt
    ms = rng.uniform(low=0., high=0.5, size=(D, 1))
    assert(np.all(ms >= 0) and np.all(ms < 1))
    
    mt = ((1.0 - ms) * rng.uniform(low=0., high=0.5, size=(D, 1))) + ms
    assert(np.all(mt >= 0) and np.all(mt < 1))
    
    missing_rates.append(('ms < mt', ms, mt))

    # sample source missingness
    # sample target missingness where ms ? mt
    ms = rng.uniform(low=0., high=0.9, size=(D, 1))
    assert(np.all(ms >= 0) and np.all(ms < 1))
    
    mt = rng.uniform(low=0., high=0.9, size=(D, 1))
    assert(np.all(mt >= 0) and np.all(mt < 1))
    
    missing_rates.append(('ms ? mt', ms, mt))

    return missing_rates


def generate_opposing_missing_rates(D, rng):
    mrates = []
    for eps in np.arange(0.05, 1, 0.05):
        more = 1 - eps
        less = eps
        ms1 = rng.uniform(low=more, high=more, size=(1, 1))
        ms2 = rng.uniform(low=less, high=less, size=(1, 1))
        ms = np.concatenate([ms1, ms2], axis=0)

        mt1 = rng.uniform(low=less, high=less, size=(1, 1))
        mt2 = rng.uniform(low=more, high=more, size=(1, 1))
        mt = np.concatenate([mt1, mt2], axis=0)
        
        mrates.append(('ms ? mt', ms, mt))
    return mrates


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run missingness experiments.')
    parser.add_argument('--num_beta_samples', type=int, default=20)
    parser.add_argument('--num_missingness_samples', type=int, default=50)
    parser.add_argument('--synth_y', action='store_true')
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--seed', type=int, default=10)
    parser.add_argument('--linreg', action='store_true')
    parser.add_argument('--xgb', action='store_true')
    parser.add_argument('--nn', action='store_true')
    parser.add_argument('--tag', type=str, default=None)
    parser.add_argument('--imputation', action='store_true')
    parser.add_argument('--verbose', action='store_true')

    args = parser.parse_args()

    if args.xgb:
        import xgboost
        from xgboost import XGBRegressor

    start_time = time.time()
    NUM_BETA_SAMPLES = args.num_beta_samples
    NUM_MISSINGNESS_SAMPLES = args.num_missingness_samples
    SYNTH_Y = args.synth_y
    SEED = args.seed
    DATASET_NAME = args.dataset
    TAG = DATASET_NAME if args.tag is None else args.tag

    rng = npr.default_rng(0)
    npr.seed(SEED)
    random.seed(SEED)

    # if not using synthetic Y, no true beta to sample
    if not SYNTH_Y:
        NUM_BETA_SAMPLES = 1

    X, y = get_raw_data(DATASET_NAME, rng)
    N, D = X.shape

    # generate all missingness rates
    if DATASET_NAME in ['bernoulli1', 'bernoulli2']:
        missingness_rates = [generate_opposing_missing_rates(D, rng) for _ in range(NUM_MISSINGNESS_SAMPLES)]
    else:
        missingness_rates = [generate_missing_rates(D, rng) for _ in range(NUM_MISSINGNESS_SAMPLES)]
    
    # for a variety of betas and missingness rates, 
    # want to characterize performance of different methods
    results = []
    for i_beta in range(NUM_BETA_SAMPLES):
        if (y is None) or SYNTH_Y:
            synth_beta = rng.uniform(low=0.0, high=10, size=(D, 1))
            y = get_synth_y(X, synth_beta)
        else:
            synth_beta = None

        data_splits = get_data_splits(X, y, ratios=(0.4, 0.1, 0.4, 0.1))
        Xs_train, Xs_test, Xt_train, Xt_test, ys_tr, ys_te, yt_tr, yt_te = data_splits
        for i_mrate, mrates in tqdm(enumerate(missingness_rates)):
            for (missingness_sampler, ms, mt) in mrates:
                time_elapsed = (time.time() - start_time) / 60.
                if args.verbose:
                    print(f'=========== time (min): {time_elapsed} beta: {i_beta}, missingness: {i_mrate}, {missingness_sampler} ==========')
                
                imputed_fpath = f'imputed_data/{DATASET_NAME}/{i_beta}_{i_mrate}_{missingness_sampler}_imputed_datadict.pkl'
                imputed_X_tr = None
                imputed_Xs_tr, imputed_Xt_tr = None, None
                imputed_Xs_te, imputed_Xt_te = None, None
                if (not args.imputation) or (not os.path.exists(imputed_fpath)):
                    # either not imputing or imputation hasn't already been done
                    Xs_tr = get_X_tilde(ms, Xs_train.copy(), rng)
                    Xs_te = get_X_tilde(ms, Xs_test.copy(), rng)
                    Xt_tr = get_X_tilde(mt, Xt_train.copy(), rng)
                    Xt_te = get_X_tilde(mt, Xt_test.copy(), rng)
                    
                    try:
                        to_keep = []
                        for k in range(Xs_tr.shape[1]):
                            if not np.all(Xs_tr[:, k] == 0):
                                to_keep.append(k)
                        Xs_tr = Xs_tr[:, to_keep]
                        Xs_te = Xs_te[:, to_keep]
                        Xt_tr = Xt_tr[:, to_keep]
                        Xt_te = Xt_te[:, to_keep]
                        ms = ms[to_keep]
                        mt = mt[to_keep]
                    except Exception as e:
                        print('Problem with removing all-zero columns after applying missingness. Skipping.')
                        print(e)
                        continue

                    # imputation
                    if args.imputation:
                        imputer = MissForest(max_iter=5, n_estimators=20, missing_values=0, copy=True, max_depth=3)
                        X_tr_concat = np.concatenate([Xs_tr, Xt_tr], axis=0)
                        try:
                            imputer = imputer.fit(X_tr_concat)
                            imputed_X_tr = imputer.transform(X_tr_concat)
                            imputed_Xs_tr = imputed_X_tr[:len(Xs_tr)]
                            imputed_Xt_tr = imputed_X_tr[len(Xs_tr):]

                            imputed_Xs_te = imputer.transform(Xs_te)
                            imputed_Xt_te = imputer.transform(Xt_te)
                            assert(len(imputed_Xt_tr) == len(Xt_tr))

                            imputed_datadict = {
                                'ms': ms.ravel(), 
                                'mt': mt.ravel(), 
                                'synth_beta': synth_beta.ravel(),
                                'imputed_Xs_tr': imputed_Xs_tr,
                                'imputed_Xt_tr': imputed_Xt_tr,
                                'imputed_Xs_te': imputed_Xs_te,
                                'imputed_Xt_te': imputed_Xt_te,
                                'ys_tr': ys_tr,
                                'yt_tr': yt_tr,
                                'ys_te': ys_te,
                                'yt_te': yt_te,
                            }
                            if not os.path.exists(f'imputed_data/{DATASET_NAME}'):
                                os.makedirs(f'imputed_data/{DATASET_NAME}')

                            print(f'saving to {imputed_fpath}')
                            pickle.dump(imputed_datadict, open(imputed_fpath, 'wb'))
                        except Exception as e:
                            print('Problem with imputation. Skipping.')
                            print(e)
                            continue
                else:
                    dd = pickle.load(open(imputed_fpath, 'rb'))
                    ms = dd['ms']
                    mt = dd['mt']
                    synth_beta = dd['synth_beta']
                    imputed_Xs_tr = dd['imputed_Xs_tr']
                    imputed_Xt_tr = dd['imputed_Xt_tr']
                    imputed_Xs_te = dd['imputed_Xs_te']
                    imputed_Xt_te = dd['imputed_Xt_te']
                    ys_tr = dd['ys_tr']
                    yt_tr = dd['yt_tr']
                    ys_te = dd['ys_te']
                    yt_te = dd['yt_te']

                ## For each of the models below, compute the relevant metrics

                # adjusted linear model
                model, pred_beta = get_adjusted_linear_model(Xs_tr, ys_tr, Xt_tr, r=None)
                if model is None:
                    print('Exception! missingness too high')
                    continue
                yt_tr_preds = model(Xt_tr)
                yt_te_preds = model(Xt_te)
                
                tr_mse = mean_squared_error(yt_tr, yt_tr_preds)
                te_mse = mean_squared_error(yt_te, yt_te_preds)
                results.append({
                    'domain': 'target',
                    'model': 'adjusted linear',
                    'train_score': tr_mse,
                    'test_score': te_mse,
                    'train_y_std': yt_tr.std(),
                    'test_y_std': yt_te.std(),
                    'metric_name': 'MSE',
                    'missingness_sampler': missingness_sampler,
                    'ms': ms.ravel(),
                    'mt': mt.ravel(),
                    'synth_beta': synth_beta if synth_beta is None else synth_beta.ravel(),
                    'pred_beta': pred_beta.ravel(),
                })

                # transform for non-parametric models
                true_r = (1 - ((1 - mt)/(1 - ms)))
                transformed_Xt = transform_Xs_to_Xt(Xs_tr, Xt_tr, rng, r=None, loose=True)
                transformed_yt = ys_tr
                
                model_classes = []
                if args.linreg:
                    model_classes.append(('linreg', LinearRegression))
                if args.xgb:
                    model_classes.append(('xgb', XGBRegressor))
                if args.nn:
                    model_classes.append(('nn', MLPRegressor))
                
                for mname, mclass in model_classes:
                    if args.verbose:
                        print(f'{mname}...')
                    ## Oracle -- model class trained on target labels
                    model = mclass()
                    model.fit(Xt_tr, yt_tr.ravel())
                    yt_tr_preds = model.predict(Xt_tr)  # predict on target domain
                    yt_te_preds = model.predict(Xt_te)
                    tr_mse = mean_squared_error(yt_tr, yt_tr_preds)
                    te_mse = mean_squared_error(yt_te, yt_te_preds)
                    if 'coef_' in model.__dict__.keys():
                        pred_beta = np.concatenate([np.array([model.intercept_]).ravel(), model.coef_.ravel()])

                    results.append({
                        'domain': 'target',
                        'model': f'oracle {mname}',
                        'train_score': tr_mse,
                        'test_score': te_mse,
                        'train_y_std': yt_tr.std(),
                        'test_y_std': yt_te.std(),
                        'metric_name': 'MSE',
                        'missingness_sampler': missingness_sampler,
                        'ms': ms.ravel(),
                        'mt': mt.ravel(),
                        'synth_beta': synth_beta if synth_beta is None else synth_beta.ravel(),
                        'pred_beta': pred_beta,
                    })

                    ## Imputation
                    if args.imputation:
                        model = mclass()
                        model.fit(imputed_Xs_tr, ys_tr.ravel())
                        if 'coef_' in model.__dict__.keys():
                            pred_beta = np.concatenate([np.array([model.intercept_]).ravel(), model.coef_.ravel()])
                        yt_tr_preds = model.predict(imputed_Xt_tr)  # predict on target domain
                        yt_te_preds = model.predict(imputed_Xt_te)
                        tr_mse = mean_squared_error(yt_tr, yt_tr_preds)
                        te_mse = mean_squared_error(yt_te, yt_te_preds)
                        results.append({
                            'domain': 'target',
                            'model': f'imputed {mname}',
                            'train_score': tr_mse,
                            'test_score': te_mse,
                            'train_y_std': yt_tr.std(),
                            'test_y_std': yt_te.std(),
                            'metric_name': 'MSE',
                            'missingness_sampler': missingness_sampler,
                            'ms': ms.ravel(),
                            'mt': mt.ravel(),
                            'synth_beta': synth_beta.ravel(),
                            'pred_beta': pred_beta.ravel(),
                        })
                        ys_tr_preds = model.predict(imputed_Xs_tr)  # predict on source domain
                        ys_te_preds = model.predict(imputed_Xs_te)
                        tr_mse = mean_squared_error(ys_tr, ys_tr_preds)
                        te_mse = mean_squared_error(ys_te, ys_te_preds)
                        results.append({
                            'domain': 'source',
                            'model': f'imputed {mname}',
                            'train_score': tr_mse,
                            'test_score': te_mse,
                            'train_y_std': ys_tr.std(),
                            'test_y_std': ys_te.std(),
                            'metric_name': 'MSE',
                            'missingness_sampler': missingness_sampler,
                            'ms': ms.ravel(),
                            'mt': mt.ravel(),
                            'synth_beta': synth_beta.ravel(),
                            'pred_beta': pred_beta.ravel(),
                        })

                    ## Adjusted for target domain
                    model = mclass()
                    model.fit(transformed_Xt, transformed_yt.ravel())
                    yt_tr_preds = model.predict(Xt_tr)  # predict on target domain
                    yt_te_preds = model.predict(Xt_te)
                    tr_mse = mean_squared_error(yt_tr, yt_tr_preds)
                    te_mse = mean_squared_error(yt_te, yt_te_preds)
                    if 'coef_' in model.__dict__.keys():
                        pred_beta = np.concatenate([np.array([model.intercept_]).ravel(), model.coef_.ravel()])
                    results.append({
                        'domain': 'target',
                        'model': f'transformed {mname}',
                        'train_score': tr_mse,
                        'test_score': te_mse,
                        'train_y_std': yt_tr.std(),
                        'test_y_std': yt_te.std(),
                        'metric_name': 'MSE',
                        'missingness_sampler': missingness_sampler,
                        'ms': ms.ravel(),
                        'mt': mt.ravel(),
                        'synth_beta': synth_beta if synth_beta is None else synth_beta.ravel(),
                        'pred_beta': pred_beta.ravel(),
                    })

                    ## Not adjusted for target domain
                    model = mclass()
                    model.fit(Xs_tr, ys_tr.ravel())
                    yt_tr_preds = model.predict(Xt_tr)  # predict on target domain
                    yt_te_preds = model.predict(Xt_te)
                    tr_mse = mean_squared_error(yt_tr, yt_tr_preds)
                    te_mse = mean_squared_error(yt_te, yt_te_preds)
                    if 'coef_' in model.__dict__.keys():
                        pred_beta = np.concatenate([np.array([model.intercept_]).ravel(), model.coef_.ravel()])
                    results.append({
                        'domain': 'target',
                        'model': f'{mname}',
                        'train_score': tr_mse,
                        'test_score': te_mse,
                        'train_y_std': yt_tr.std(),
                        'test_y_std': yt_te.std(),
                        'metric_name': 'MSE',
                        'missingness_sampler': missingness_sampler,
                        'ms': ms.ravel(),
                        'mt': mt.ravel(),
                        'synth_beta': synth_beta if synth_beta is None else synth_beta.ravel(),
                        'pred_beta': pred_beta.ravel(),
                    })
                    ys_tr_preds = model.predict(Xs_tr)  # predict on source domain
                    ys_te_preds = model.predict(Xs_te)
                    tr_mse = mean_squared_error(ys_tr, ys_tr_preds)
                    te_mse = mean_squared_error(ys_te, ys_te_preds)
                    results.append({
                        'domain': 'source',
                        'model': f'{mname}',
                        'train_score': tr_mse,
                        'test_score': te_mse,
                        'train_y_std': yt_tr.std(),
                        'test_y_std': yt_te.std(),
                        'metric_name': 'MSE',
                        'missingness_sampler': missingness_sampler,
                        'ms': ms.ravel(),
                        'mt': mt.ravel(),
                        'synth_beta': synth_beta if synth_beta is None else synth_beta.ravel(),
                        'pred_beta': pred_beta.ravel(),
                    })

            pd.DataFrame(results).to_csv(f'{TAG}_missingness_results.csv')
