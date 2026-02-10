import time
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    mean_squared_error, mean_absolute_error, r2_score
)

def run_baseline(mm, m_type, X_train_proc, X_test_proc, y_train, y_test, task_type):
    """
    Returnează dict cu:
      - Baseline_Accuracy/F1/Precision/Recall + Baseline_TrainTime/Baseline_InfTime
      SAU
      - Baseline_MSE/MAE/R2 + Baseline_TrainTime/Baseline_InfTime
    Exact ca în codul tău.
    """
    clf = mm.get_baseline_model(m_type)

    t0 = time.time()
    clf.fit(X_train_proc, y_train)
    train_time = time.time() - t0

    t0 = time.time()
    preds = clf.predict(X_test_proc)
    inf_time = time.time() - t0

    if task_type == "classification":
        return {
            "Baseline_Accuracy": accuracy_score(y_test, preds),
            "Baseline_F1": f1_score(y_test, preds, average="weighted", zero_division=0),
            "Baseline_Precision": precision_score(y_test, preds, average="weighted", zero_division=0),
            "Baseline_Recall": recall_score(y_test, preds, average="weighted", zero_division=0),
            "Baseline_TrainTime": train_time,
            "Baseline_InfTime": inf_time,
        }
    else:
        return {
            "Baseline_MSE": mean_squared_error(y_test, preds),
            "Baseline_MAE": mean_absolute_error(y_test, preds),
            "Baseline_R2": r2_score(y_test, preds),
            "Baseline_TrainTime": train_time,
            "Baseline_InfTime": inf_time,
        }
