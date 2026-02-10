import numpy as np
from sklearn.metrics import accuracy_score, log_loss, f1_score, precision_score

def run_simple_mia(model, X_train, y_train, X_test, y_test):
    """
    Runs a simple Membership Inference Attack (MIA) metric: Generalization Gap.
    
    The intuition is that if a model performs significantly better on training data
    than test data (Overfitting), it has "memorized" the training data, making it
    vulnerable to membership inference.
    
    Returns:
        dict: A dictionary containing:
            - train_acc: Accuracy on training set (Proxy for True Positive Rate)
            - test_acc: Accuracy on test set (Proxy for False Positive Rate)
            - privacy_gap: train_acc - test_acc (The Attack Advantage)
            - vulnerability_score: A heuristic score (0-100) of privacy risk.
    """
    
    # 1. Get Predictions
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)
    
    # 2. Calculate Accuracies
    train_acc = accuracy_score(y_train, train_preds)
    test_acc = accuracy_score(y_test, test_preds)
    
    # 3. Calculate Gap (Attack Advantage)
    # If the attacker simply guessed "Member" whenever the model was correct:
    # TPR = Train Acc
    # FPR = Test Acc
    # Advantage = TPR - FPR
    # 3. Calculate Gap (Attack Advantage)
    gap = max(0, train_acc - test_acc)
    
    # 4. Interpret Result
    risk_level = "Low"
    if gap > 0.15:
        risk_level = "High"
    elif gap > 0.05:
        risk_level = "Medium"
    return {
        'train_accuracy': train_acc,
        'test_accuracy': test_acc,
        'attack_advantage': gap,
        'risk_level': risk_level,
        'train_f1': f1_score(y_train, train_preds, average='weighted', zero_division=0),
        'test_f1': f1_score(y_test, test_preds, average='weighted', zero_division=0),
        'train_precision': precision_score(y_train, train_preds, average='weighted', zero_division=0),
        'test_precision': precision_score(y_test, test_preds, average='weighted', zero_division=0)
    }

def run_threshold_mia(model, X_train, y_train, X_test, y_test):
    """
    Comparison of confidence scores (probabilities).
    Usually DP models will have lower confidence on correct predictions than non-DP models.
    """
    # Check if model supports probabilities
    if not hasattr(model, 'predict_proba'):
        return run_simple_mia(model, X_train, y_train, X_test, y_test)

    # Get probabilities for the correct class
    # For each sample, get prob of the true class
    train_probs = model.predict_proba(X_train)
    test_probs = model.predict_proba(X_test)
    
    # Extract prob of the "True" class (assuming binary classification 0/1 for simplicity, 
    # but works for multiclass if we index properly)
    # Simplification: Just take max probability (Confidence)
    train_conf = np.max(train_probs, axis=1).mean()
    test_conf = np.max(test_probs, axis=1).mean()
    
    return {
        'avg_confidence_train': train_conf,
        'avg_confidence_test': test_conf,
        'confidence_gap': max(0, train_conf - test_conf)
    }

def compute_mia_metrics(model, X_train, y_train, X_test, y_test, task_type, prefix='K_Anon'):
    """
    High-level wrapper to run MIA if appropriate (Classification only) and format results.
    """
    metrics = {}
    if task_type == 'classification':
        mia_res = run_simple_mia(model, X_train, y_train, X_test, y_test)
        metrics = {
             f'{prefix}_MIA_Gap': mia_res['attack_advantage'],
             f'{prefix}_MIA_Train_F1': mia_res['train_f1'],
             f'{prefix}_MIA_Test_F1': mia_res['test_f1']
        }
    return metrics
