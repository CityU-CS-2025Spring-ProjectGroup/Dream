from sklearn import *

from hyperopt import fmin, hp, STATUS_OK, Trials, tpe
from hyperopt.early_stop import no_progress_loss
from sklearn.metrics import f1_score


def rf_trainer(X_train, y_train, X_val, y_val):
    param_space = {
        'n_estimators':         hp.choice('n_estimators', range(50, 500)),
        'max_depth':            hp.choice('max_depth', range(1, 20)),
        'max_features':         hp.choice('max_features', ['sqrt', 'log2', None]),
        'min_samples_split':    hp.uniform('min_samples_split', 0.01, 1),
        'min_samples_leaf':     hp.uniform('min_samples_leaf', 0.01, 0.5),
        'bootstrap':            hp.choice('bootstrap', [True, False]),
        'criterion':            hp.choice('criterion', ['gini', 'entropy'])
    }

    def optimize_obj(param_space):
        clf = ensemble.RandomForestClassifier(
            n_estimators        = int(param_space['n_estimators']),
            max_depth           = int(param_space['max_depth']),
            max_features        =     param_space['max_features'],
            min_samples_split   =     param_space['min_samples_split'],
            min_samples_leaf    =     param_space['min_samples_leaf'],
            bootstrap           =     param_space['bootstrap'],
            criterion           =     param_space['criterion'],
            random_state        = 4487,
            n_jobs              = 8
        )
        clf.fit(X_train, y_train)

        pos_prob = clf.predict_proba(X_val)[:, 1]
        pred_labels = (pos_prob > 0.5).astype(int)

        return {
            'loss': -f1_score(y_val, pred_labels),
            'status': STATUS_OK
        }

    trials = Trials()
    best_params = fmin(
        fn                      = optimize_obj,
        space                   = param_space,
        algo                    = tpe.suggest,
        max_evals               = 50,
        trials                  = trials,
        early_stop_fn           = no_progress_loss(20)  # stop training if no improvements in 5 trials
    )

    print(f"Best Hyperparameters of RandomForest: {best_params}")

    best_params = {
        'n_estimators':         int(best_params['n_estimators']),
        'max_depth':            int(best_params['max_depth']),
        'max_features':         ['sqrt', 'log2', None][best_params['max_features']],
        'min_samples_split':    best_params['min_samples_split'],
        'min_samples_leaf':     best_params['min_samples_leaf'],
        'bootstrap':            [True, False][best_params['bootstrap']],
        'criterion':            ['gini', 'entropy'][best_params['criterion']]
    }
    
    rf = ensemble.RandomForestClassifier(
        **best_params,
        random_state=1,
        n_jobs=8
    )
    rf.fit(X_train, y_train)

    return rf
