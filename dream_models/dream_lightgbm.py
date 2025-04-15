import lightgbm as lgb

from hyperopt import fmin, hp, STATUS_OK, Trials, tpe
from sklearn.metrics import f1_score


def lightgbm_trainer(X_train, y_train, X_val, y_val):
    param_space = {
        'max_depth':            hp.quniform('max_depth', 2, 6, 1),
        'reg_alpha':            hp.quniform('reg_alpha', 0, 180, 2),
        'reg_lambda':           hp.uniform ('reg_lambda', 0.2, 5),
        'num_leaves':           hp.quniform('num_leaves', 20, 100, 10),
        'n_estimators':         hp.quniform('n_estimators', 50, 300, 10),
        'learning_rate':        hp.uniform ('learning_rate', 0.005, 0.5)
    }

    def optimize_obj(param_space):
        clf = lgb.LGBMClassifier(
            objective           = 'binary',
            scale_pos_weight    = 1.5,
            max_depth           = int(param_space['max_depth']),
            reg_alpha           =     param_space['reg_alpha'],
            reg_lambda          =     param_space['reg_lambda'],
            n_estimators        = int(param_space['n_estimators']),
            learning_rate       =     param_space['learning_rate'],
            num_leaves          = int(param_space['num_leaves']),
            verbose             = 1
        )
        clf.fit(X_train, y_train)

        pos_prob = clf.predict_proba(X_val)[:, 1]
        pred_labels = (pos_prob > 0.5).astype(int)

        return {
            'loss': -f1_score(y_val, pred_labels),
            'status': STATUS_OK
        }
    
    # find best hyperparameters
    trials = Trials()
    best_params = fmin(
        fn                      = optimize_obj, 
        space                   = param_space, 
        algo                    = tpe.suggest, 
        max_evals               = 50, 
        trials                  = trials
    )

    print(f"Best Hyperparameters of LGBM: {best_params}")

    # transform data type
    best_params['max_depth']    = int(best_params['max_depth'])
    best_params['n_estimators'] = int(best_params['n_estimators'])
    best_params['num_leaves']   = int(best_params['num_leaves'])

    # final model
    lgbm = lgb.LGBMClassifier(
        **best_params,
        random_state=1,
        num_iterations=50
    )
    lgbm.fit(X_train, y_train)

    return lgbm
