from sklearn.neighbors import KNeighborsClassifier
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from sklearn.metrics import f1_score

def knn_trainer(
    X_train_resampled, 
    y_train_resampled, 
    X_val, 
    y_val, 
):
    param_space = {
        "n_neighbors":      hp.quniform("n_neighbors", 3, 50, 1), 
        "weights":          hp.choice("weights", ["uniform", "distance"]),
        "metric":           hp.choice("metric", ["euclidean", "manhattan", "minkowski"])
    }
    
    def objective(params):
        params["n_neighbors"] = int(params["n_neighbors"])
        
        knn = KNeighborsClassifier(
            n_neighbors     = params["n_neighbors"],
            weights         = params["weights"],
            metric          = params["metric"],
            n_jobs          = -1 
        )
        
        knn.fit(X_train_resampled, y_train_resampled)
        
        y_pred = knn.predict(X_val)
        
        f1 = f1_score(y_val, y_pred)
        
        return {
            "loss":         -f1, 
            "status":       STATUS_OK
        }
    
    trials = Trials()
    best_params = fmin(
        fn                  = objective,
        space               = param_space,
        algo                = tpe.suggest,
        max_evals           = 50,       
        trials              = trials
    )
    
    best_params["weights"] = ["uniform", "distance"][best_params["weights"]]
    best_params["metric"] = ["euclidean", "manhattan", "minkowski"][best_params["metric"]]
    best_params["n_neighbors"] = int(best_params["n_neighbors"])
    
    print("Best hyperparameters:", best_params)
    
    final_knn = KNeighborsClassifier(
        n_neighbors         = best_params["n_neighbors"],
        weights             = best_params["weights"],
        metric              = best_params["metric"],
        n_jobs              = -1
    )

    final_knn.fit(X_train_resampled, y_train_resampled)
    
    return final_knn
