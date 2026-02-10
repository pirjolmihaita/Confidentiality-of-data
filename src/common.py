import numpy as np

def make_suffix(eps, norm):
    return f"_Eps{eps}_Norm{norm}"

def get_unique_classes(y_train, task_type):
    return np.unique(y_train) if task_type == "classification" else None

def get_bounds(norm):
    return (-norm, norm)

def slice_he_features(X_train_proc, X_test_proc, he_n_features_limit=10):
    n_features = X_train_proc.shape[1]
    if n_features > he_n_features_limit:
        return X_train_proc[:, :he_n_features_limit], X_test_proc[:, :he_n_features_limit], n_features, he_n_features_limit
    return X_train_proc, X_test_proc, n_features, n_features

def slice_subset(X, n):
    return X[:n]
