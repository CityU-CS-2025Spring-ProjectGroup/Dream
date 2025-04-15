import pandas as pd
import numpy as np
import gpboost as gpb
from sklearn.metrics import f1_score, cohen_kappa_score
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK


def gpboost_trainer(
        X_train_resampled, 
        group_train_resampled, 
        y_train_resampled, 
        X_val, 
        y_val, 
        group_val
):
    param_space = {
        "max_depth":                hp.quniform("max_depth", 3, 6, 1),
        "learning_rate":            hp.uniform("learning_rate", 0.005, 0.01),
        "num_leaves":               hp.quniform("num_leaves", 20, 200, 20),
        "feature_fraction":         hp.uniform("feature_fraction", 0.5, 0.95),
        "lambda_l2":                hp.uniform("lambda_l2", 1.0, 10.0),
        "lambda_l1":                hp.quniform("lambda_l1", 10, 100, 10),
        "pos_bagging_fraction":     hp.uniform("pos_bagging_fraction", 0.8, 0.95),  # 正样本采样比例
        "neg_bagging_fraction":     hp.uniform("neg_bagging_fraction", 0.6, 0.8),  # 负样本采样比例
        "num_boost_round":          hp.quniform("num_boost_round", 400, 1000, 100),  # 训练轮数
    }

    # 目标函数：用于评估每组超参数的性能
    def objective(space):
        # 提取超参数并转换为模型需要的格式
        params = {
            "objective": "binary",  # 二分类任务
            "max_depth": int(space["max_depth"]),  # 强制转换为整数
            "learning_rate": space["learning_rate"],
            "num_leaves": int(space["num_leaves"]),
            "feature_fraction": space["feature_fraction"],
            "lambda_l2": space["lambda_l2"],
            "lambda_l1": space["lambda_l1"],
            "pos_bagging_fraction": space["pos_bagging_fraction"],
            "neg_bagging_fraction": space["neg_bagging_fraction"],
            "num_boost_round": int(space["num_boost_round"]),
            "verbose": -1,  # 不输出日志
        }
        num_boost_round = params.pop("num_boost_round")  # 提取训练轮数

        # 初始化高斯过程模型（处理随机效应）
        gp_model = gpb.GPModel(
            group_data=group_train_resampled,  # 分组变量作为随机效应
            likelihood="bernoulli_probit"  # 二分类概率模型
        )

        # 加载训练数据
        data_train = gpb.Dataset(data=X_train_resampled, label=y_train_resampled)

        # 训练模型（结合梯度提升树和高斯过程）
        clf = gpb.train(
            params=params,
            train_set=data_train,
            gp_model=gp_model,  # 绑定高斯过程
            num_boost_round=num_boost_round
        )

        # 在验证集上预测
        pred_resp = clf.predict(
            data=X_val,
            group_data_pred=group_val,  # 验证集分组变量
            predict_var=True,  # 预测方差
            pred_latent=False  # 输出概率而非潜在变量
        )
        positive_probabilities = pred_resp["response_mean"]  # 正类概率
        predicted_labels = (positive_probabilities > 0.5).astype(int)  # 转为二分类标签

        # 计算 F1 分数（最大化 F1，因此损失为 -F1）
        f1 = f1_score(y_val, predicted_labels)
        return {"loss": -f1, "status": STATUS_OK}

    # 使用贝叶斯优化搜索最佳超参数（最大尝试 10 组）
    trials = Trials()
    gpb_best_hyperparams = fmin(
        fn=objective,
        space=param_space,
        algo=tpe.suggest,  # 使用 TPE 算法优化
        max_evals=10,
        trials=trials
    )
    print("Best hyperparameters:", gpb_best_hyperparams)

    # 调整超参数类型（部分参数需转为整数）
    gpb_best_hyperparams["max_depth"] = int(gpb_best_hyperparams["max_depth"])
    gpb_best_hyperparams["num_leaves"] = int(gpb_best_hyperparams["num_leaves"])
    gpb_best_hyperparams["num_boost_round"] = int(gpb_best_hyperparams["num_boost_round"])

    # 用最佳参数训练最终模型
    data_train = gpb.Dataset(X_train_resampled, y_train_resampled)
    data_eval = gpb.Dataset(X_val, y_val)
    gp_model = gpb.GPModel(
        group_data=group_train_resampled,
        likelihood="bernoulli_probit"
    )
    gp_model.set_prediction_data(group_data_pred=group_val)  # 绑定验证集分组变量

    # 训练并启用早停（如果验证集性能 10 轮不提升则停止）
    evals_result = {}  # 记录训练过程中的评估结果
    final_gpb_model = gpb.train(
        params=gpb_best_hyperparams,
        train_set=data_train,
        gp_model=gp_model,
        valid_sets=data_eval,
        early_stopping_rounds=10,
        use_gp_model_for_validation=True,  # 验证时使用高斯过程
        evals_result=evals_result,
    )

    return final_gpb_model