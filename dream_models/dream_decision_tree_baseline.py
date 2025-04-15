from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK

def decision_tree_trainer(
    X_train_resampled, 
    y_train_resampled, 
    X_val, 
    y_val
):
    space = {
        "max_depth":                hp.quniform("max_depth", 3, 15, 1),
        "min_samples_split":        hp.quniform("min_samples_split", 2, 20, 1),
        "min_samples_leaf":         hp.quniform("min_samples_leaf", 1, 10, 1),
        "criterion":                hp.choice("criterion", ["gini", "entropy"])
    }

    def objective(params):
        params = {
            "max_depth":            int(params["max_depth"]),
            "min_samples_split":    int(params["min_samples_split"]),
            "min_samples_leaf":     int(params["min_samples_leaf"]),
            "criterion":            params["criterion"]
        }

        clf = DecisionTreeClassifier(
            **params
        )

        clf.fit(X_train_resampled, y_train_resampled)

        y_pred = clf.predict(X_val)

        f1 = f1_score(y_val, y_pred)
        return {
            "loss":                 -f1, 
            "status":               STATUS_OK
        }

    trials = Trials()
    best_hyperparams = fmin(
        fn                          = objective,
        space                       = space,
        algo                        = tpe.suggest,
        max_evals                   = 50,
        trials                      = trials
    )

    if "criterion" in best_hyperparams:
        best_hyperparams["criterion"] = ["gini", "entropy"][best_hyperparams["criterion"]]

    best_hyperparams["max_depth"] = int(best_hyperparams["max_depth"])
    best_hyperparams["min_samples_split"] = int(best_hyperparams["min_samples_split"])
    best_hyperparams["min_samples_leaf"] = int(best_hyperparams["min_samples_leaf"])

    print("Best hyperparameters:", best_hyperparams)

    final_dt_model = DecisionTreeClassifier(**best_hyperparams, random_state=42)
    final_dt_model.fit(X_train_resampled, y_train_resampled)

    return final_dt_model