import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import *

from scipy.signal.windows import gaussian,  windows
from scipy.signal import  convolve

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from preprocessing.datasets import resample_data, train_test_split


def process_and_resample_data(df, train_sids, val_sids, test_sids, final_features, group_var):
    X_train, y_train, group_train = train_test_split(
        df, train_sids, final_features, group_var
    )

    X_val, y_val, group_val = train_test_split(
        df, val_sids, final_features, group_var
    )

    X_test, y_test, group_test = train_test_split(
        df, test_sids, final_features, group_var
    )

    X_train_resampled, y_train_resampled, group_train_resampled = resample_data(
        X_train, y_train, group_train, group_var
    )

    return (X_train_resampled, y_train_resampled, group_train_resampled), (X_train, y_train, group_train), (X_val, y_val, group_val), (X_test, y_test, group_test)


def missingness_imputation(data):
    """Perform missingness imputation on the given data.

    Parameters
    ----------
    data : array-like
        The data to be imputed.

    Returns
    -------
    interpolated_series : pandas Series
        The imputed data.
    """

    indices = np.arange(len(data))
    series = pd.Series(data, index=indices)
    interpolated_series = series.interpolate(method="linear")
    return interpolated_series


def half_gaussian_kernel(size, std_dev):
    """Create a half Gaussian kernel.

    Parameters
    ----------
    size : int
        The size of the kernel.
    std_dev : float
        The standard deviation of the kernel.

    Returns
    -------
    half_kernel : array
        The half Gaussian kernel.
    """
    full_kernel = gaussian(size, std_dev)
    half_kernel = full_kernel[: size // 2]
    half_kernel /= half_kernel.sum()
    return half_kernel


def apply_half_gaussian_filter(data, kernel_size, std_dev):
    """Apply a half Gaussian filter to the given data.

    Parameters
    ----------
    data : array-like
        The data to be filtered.
    kernel_size : int
        The size of the kernel.
    std_dev : float
        The standard deviation of the kernel.

    Returns
    -------
    filtered_data : array
        The filtered data.
    """
    kernel = half_gaussian_kernel(kernel_size, std_dev)
    filtered_data = convolve(data, kernel, mode="valid")
    left_padding_length = kernel_size // 2 - 1
    filtered_data = np.pad(
        filtered_data, (left_padding_length, 0), "constant", constant_values=(np.nan,)
    )
    return filtered_data


def half_gaussian_filtering(df, columns, kernel_size=40, std_dev=100):
    """Perform half Gaussian filtering on the given DataFrame.

    Parameters
    ----------
    df : pandas DataFrame
        The DataFrame to be filtered.
    columns : list
        The columns to be filtered.
    kernel_size : int
        The size of the kernel.
    std_dev : float
        The standard deviation of the kernel.

    Returns
    -------
    df : pandas DataFrame
        The filtered DataFrame.
    """
    for column in columns:
        interpolated_series = missingness_imputation(
            apply_half_gaussian_filter(df[column], kernel_size, std_dev)
        )
        df["gaussian_{}".format(column)] = interpolated_series
    return df


def apply_gaussian_filter(data, kernel_size, std_dev):
    """Apply a Gaussian filter to the given data.

    Parameters
    ----------
    data : array-like
        The data to be filtered.
    kernel_size : int
        The size of the kernel.
    std_dev : float
        The standard deviation of the kernel.

    Returns
    -------
    filtered_data : array
        The filtered data.
    """
    kernel = windows.gaussian(kernel_size, std_dev, sym=True)
    kernel /= np.sum(kernel)
    filtered_data = convolve(data, kernel, mode="same")
    return filtered_data


def gaussian_filtering(df, columns, kernel_size=40, std_dev=100):
    """Perform Gaussian filtering on the given DataFrame.

    Parameters
    ----------
    df : pandas DataFrame
        The DataFrame to be filtered.
    columns : list
        The columns to be filtered.
    kernel_size : int
        The size of the kernel.
    std_dev : float
        The standard deviation of the kernel.

    Returns
    -------
    df : pandas DataFrame
        The filtered DataFrame.
    """
    for column in columns:
        interpolated_series = missingness_imputation(
            apply_gaussian_filter(df[column], kernel_size, std_dev)
        )
        df["gaussian_{}".format(column)] = interpolated_series
    #         df["gaussian_diff_{}".format(column)] = interpolated_series - df[column]
    return df


def rolling_stds(df, columns, window_size=20):
    """Calculate rolling standard deviations for the given columns in the DataFrame.

    Parameters
    ----------
    df : pandas DataFrame
        The DataFrame for which to calculate rolling standard deviations.
    columns : list
        The columns for which to calculate rolling standard deviations.
    window_size : int
        The size of the rolling window.

    Returns
    -------
    df : pandas DataFrame
        The DataFrame with the rolling standard deviations added.
    """
    for column in columns:
        df["rolling_var_{}".format(column)] = (
            df[column].rolling(window=window_size, min_periods=1).var()
        )
    return df


def add_derivatives(df, features):
    """Add first and second derivatives to the given features in the DataFrame.

    Parameters
    ----------
    df : pandas DataFrame
        The DataFrame to which to add the derivatives.
    features : list
        The features to which to add the derivatives.
    

    Returns
    -------
    df : pandas DataFrame
        The DataFrame with the derivatives added.
    """
    for feature in features:
        # First derivative
        first_derivative_column = "gaussian_" + feature + "_1st_derivative"
        df[first_derivative_column] = np.gradient(df["gaussian_" + feature])

        # Second derivative
        raw_derivative_column = "raw_" + feature + "_1st_derivative"
        df[raw_derivative_column] = df[feature].diff()
    return df


def compute_probabilities(list_sids, df, features_list, model_name, final_model, group_variable):
    """
    Computes the probabilities based on the final features, using a given model. 

    Parameters:
    ----------
    list_sids : list
        A list of subject IDs for which to compute probabilities.
    df : pandas dataframe
        The  DataFrame containing the data from which features will be extracted.
    features_list : list
        A list of strings representing the names of the features to be used in the model.
    model_name : str
        The name of the model to use for prediction. Input is expected to be 'lgb' or 'gpb'.
    final_model: 
        The trained model object to use for predictions.
    group_variable: list
        A list containing the selected variable(s).

    Returns:
    ----------
    list_probabilities_subject : list
        A list of the predicted probabilities for each sleep stage for each subject.
    lengths : list
        A list of integers representing the number of predictions for each subject.
    list_true_stages : list
        A list of the true sleep stages for each subject.
    """
    list_probabilities_subject = []
    list_true_stages = []
    lengths = []


    for sid in list_sids:
        sid_df = df.loc[
            df["sid"] == sid,
            features_list + ["Sleep_Stage", "timestamp_start"] + group_variable,
        ].copy()

        sid_df = sid_df.reset_index(drop=True)
        sid_df = sid_df.dropna()
        x = sid_df.loc[:, features_list].to_numpy()
        group = sid_df.loc[:, group_variable].to_numpy()

        if model_name in {'kNN', 'DT', 'lgb', 'rf'}:
            """KNN、DT、LightGBM and RandomForest"""

            pred_proba = final_model.predict_proba(x)

            sid_df["predicted_Sleep_Stage"] = np.argmax(pred_proba, axis=1)
            sid_df["predicted_Sleep_Stage_Proba_Class_0"] = pred_proba[:, 0]
            sid_df["predicted_Sleep_Stage_Proba_Class_1"] = pred_proba[:, 1]

        elif model_name == 'gpb':
            """GPBoost"""

            pred_resp = final_model.predict(
                data=x, group_data_pred=group, predict_var=True, pred_latent=False
            )
            positive_probabilities = pred_resp["response_mean"]
            negative_probabilities = 1 - positive_probabilities

            sid_df["predicted_Sleep_Stage"] = (positive_probabilities > 0.5).astype(int)

            sid_df["predicted_Sleep_Stage_Proba_Class_0"] = negative_probabilities
            sid_df["predicted_Sleep_Stage_Proba_Class_1"] = positive_probabilities
        
        else:
            print('Wrong model')

        probabilities_subject = sid_df.loc[
            :,
            [
                "predicted_Sleep_Stage_Proba_Class_0",  # Result proba of tree for class 0
                "predicted_Sleep_Stage_Proba_Class_1",  # Result proba of tree for class 1
                "ACC_INDEX",                            # The strongest predictor 1
                "HRV_HFD",                              # The strongest predictor 2
            ],
        ].to_numpy()

        list_probabilities_subject.append(probabilities_subject)
        lengths.append(probabilities_subject.shape[0])
        list_true_stages.append(sid_df.Sleep_Stage.to_numpy())

    return list_probabilities_subject, lengths, list_true_stages


class TimeSeriesDataset(Dataset):
    def __init__(self, data, lengths, labels):
        self.data = data
        self.lengths = lengths
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        length = self.lengths[idx]
        label = self.labels[idx]
        return {
            "sample": torch.tensor(sample, dtype=torch.float),
            "length": torch.tensor(length, dtype=torch.long),
            "label": torch.tensor(label, dtype=torch.long),
        }


class SimpleDataset(Dataset):
    def __init__(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train

    def __len__(self):
        return len(self.x_train)

    def __getitem__(self, idx):
        x = self.x_train[idx]
        y = self.y_train[idx]
        return x, y


def collate_fn(batch):
    # Extract samples, lengths, and labels from the batch
    samples = [item["sample"] for item in batch]
    lengths = [item["length"] for item in batch]
    labels = [item["label"] for item in batch]

    # Pad the samples
    samples_padded = pad_sequence(samples, batch_first=True)

    # Convert lengths to a tensor
    lengths_tensor = torch.tensor(lengths, dtype=torch.long)

    # If labels are sequences, pad them; otherwise, convert to a tensor directly
    if isinstance(labels[0], torch.Tensor) and len(labels[0].shape) > 0:
        # Assuming labels are sequences and need padding
        labels_padded = pad_sequence(labels, batch_first=True)
    else:
        # Assuming labels are single values per sequence
        labels_tensor = torch.tensor(labels, dtype=torch.long)

    return {
        "sample": samples_padded,
        "length": lengths_tensor,
        "label": labels_padded
        if isinstance(labels[0], torch.Tensor) and len(labels[0].shape) > 0
        else labels_tensor,
    }


def calculate_accuracy(y_pred, y_true):
    # Assuming y_pred is already in probability form (e.g., output of softmax)
    predicted_labels = torch.argmax(y_pred, dim=1)
    correct_predictions = (predicted_labels == y_true).float()
    accuracy = correct_predictions.sum() / len(correct_predictions)
    return accuracy


def calculate_metrics(y_test, y_pred_proba, model_name):
    """
    Calculates the classification metrics based on the true labels and 
    predicted probabilities.

    Parameters:
    ----------
    y_test : array
        True labels for the test data.
    y_pred_proba : array
        Predicted probabilities for each class
    model_name: str
        Name of the model to be shown on the results.

    Returns:
    ----------
    result_df : pandas dataframe
        A DataFrame containing the calculated metrics for the given model.
        columns: Model, Precision, Recall, F1 Score, Specificity, AUROC, AUPRC,
                 accuracy
    """
    results = []

    predicted_labels = np.argmax(y_pred_proba, axis=1)

    accuracy = accuracy_score(y_test, predicted_labels)

    cm = confusion_matrix(y_test, predicted_labels)
    true_negatives = cm.sum() - (
        cm[1, :].sum()
        + cm[:, 1].sum()
        - cm[1, 1]
    )
    false_positives = cm[:, 1].sum() - cm[1, 1]
    specificity = true_negatives / (true_negatives + false_positives)

    precision = precision_score(
        y_test,
        predicted_labels,
        labels=[1],
    )
    recall = recall_score(
        y_test,
        predicted_labels,
        labels=[1],
    )  # Recall is the same as sensitivity
    f1 = f1_score(
        y_test,
        predicted_labels,
        labels=[1],
    )

    # Compute AUROC
    auroc = roc_auc_score(y_test, y_pred_proba[:, 1])
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred_proba[:,1])
    precision_recall_auc = auc(recalls, precisions)

    results.append(
        {
            "Model": model_name,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
            "Specificity": specificity,
            "AUROC": auroc,
            'AUPRC': precision_recall_auc,
            "Accuracy": accuracy
        }
    )

    result_df = pd.DataFrame(results)

    return result_df


def calculate_kappa(list_probabilities_subject, list_true_stages):
    """
    Calculates the Cohen's Kappa score of each subject.

    Parameters:
    ----------
    list_probabilities_subject : list
        A list of the predicted probabilities for each sleep stage for each subject.
    list_true_stages : list
        A list of the true sleep stages for each subject.

    Returns:
    ----------
    avg_cp : float
        The average value of the Cohen's Kappa score of all the subject.
    """
    cp = []
    for i, probabilities in enumerate(list_probabilities_subject):
        cp.append(cohen_kappa_score(list_true_stages[i], np.argmax(probabilities[:, :2], axis=1)))
    avg_cp = np.average(cp)
    return avg_cp


def plot_cm(list_probabilities_subject, list_true_stages, model_name):
    """
    Plot the confusion matrix of a model's prediction.

    Parameters:
    ----------
    list_probabilities_subject : list
        A list of the predicted probabilities for each sleep stage for each subject.
    list_true_stages : list
        A list of the true sleep stages for each subject.
    model_name : str
        The name of the model to be shown on the plot. 

    Returns:
    ----------
    The function does not return any value. It generates and displays a confusion matrix
        heatmap plot directly.
    """
    if 'Transformer' in model_name:
        y_test = list_true_stages
        y_pred = list_probabilities_subject
    else:
        y_test = np.concatenate(list_true_stages)
        if 'LSTM' in model_name:
            y_pred = np.argmax(list_probabilities_subject, axis=1)
        else:
            y_pred = np.concatenate(
                [
                    np.argmax(probabilities[:, :2], axis=1)
                    for probabilities in list_probabilities_subject
                ]
            )
    cm = confusion_matrix(y_test, y_pred)
    class_names = ["Sleep", "Wake"]
    cm_percent = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    print(cm)

    # Plotting
    plt.figure(figsize=(4, 3))
    sns.heatmap(
        cm_percent,
        annot=True,
        fmt=".2%",
        cmap="Greens",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    title = str(model_name) + ' Confusion Matrix'
    plt.title(title)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()
