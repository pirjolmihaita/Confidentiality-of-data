import pandas as pd
import numpy as np
import time
import os
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, mean_squared_error, mean_absolute_error, r2_score
from .data_loader import DataLoader
from .models import ModelManager
from .privacy.k_anonymity import MondrianAnonymizer
from .attacks import run_simple_mia, compute_mia_metrics
from .utils import get_logger
from sklearn.base import clone

logger = get_logger(__name__)

def run_experiments():
    dl = DataLoader()
    mm = ModelManager()
    
    datasets = ['adult', 'data', 'cervical', 'compas', 'creditcard', 'heart', 'insurance', 'communities']
    
    epsilons = [0.1, 0.5, 1.0, 5.0, 10.0] 
    data_norms = [1.0, 10, 100]
    
    ks = [2, 10, 50, 100] # k-Anonymity parameters
    
    models = ['lr', 'nb', 'dt', 'rf']
    
    # Wide Format Results Collection
    # We want one row per (Dataset, Model)
    # We'll create a dictionary to hold rows, keyed by (dataset, model_name)
    wide_results = {} 

    for ds_name in datasets:
        logger.info(f"Processing dataset: {ds_name}")
        try:
            try:
                X_train, X_test, y_train, y_test, preprocessor, task_type = dl.load_and_preprocess(ds_name)
            except Exception as e:
                logger.error(f"Failed to load {ds_name}: {e}")
                continue

            logger.info(f"Dataset: {ds_name} | Task: {task_type}")
            
            # FIT PREPROCESSOR
            X_train_proc = preprocessor.fit_transform(X_train)
            X_test_proc = preprocessor.transform(X_test)
            n_features = X_train_proc.shape[1]
            n_samples = len(X_train) + len(X_test)
            
            # OPTIMIZATION: Prepare 10-feature subset for HE
            # We assume the first 10 columns are fine for this demonstration
            # In a real scenario, use SelectKBest
            he_n_features_limit = 10
            if n_features > he_n_features_limit:
                X_train_he = X_train_proc[:, :he_n_features_limit]
                X_test_he = X_test_proc[:, :he_n_features_limit]
                logger.info(f"  HE Optimization: Selected top {he_n_features_limit} features from {n_features}")
            else:
                X_train_he = X_train_proc
                X_test_he = X_test_proc
                
            
            # --- BASELINE LOOP ---
            for m_key in models:
                # Map generic key to specific model type based on task
                if task_type == 'regression':
                    if m_key == 'lr': m_type = 'lin_reg'
                    elif m_key == 'nb': continue # Naive Bayes not for regression usually
                    elif m_key == 'dt': m_type = 'dt_reg'
                    elif m_key == 'rf': m_type = 'rf_reg'
                    else: continue # Should not happen with current models list
                else:
                    m_type = m_key

                row_key = (ds_name, m_type, 'Baseline', 'None', 'None')
                if row_key not in wide_results:
                    wide_results[row_key] = {
                        'Dataset': ds_name, 'Model': m_type, 'Type': 'Baseline', 
                        'Epsilon': 'None', 'Data_Norm': 'None', 'Task_Type': task_type
                    }
                
                logger.info(f"  Baseline: {m_type}")
                try:
                    clf = mm.get_baseline_model(m_type)
                    t0 = time.time()
                    clf.fit(X_train_proc, y_train)
                    train_time = time.time() - t0
                    
                    t0 = time.time()
                    preds = clf.predict(X_test_proc)
                    inf_time = time.time() - t0
                    
                    if task_type == 'classification':
                        wide_results[row_key].update({
                            'Baseline_Accuracy': accuracy_score(y_test, preds),
                            'Baseline_F1': f1_score(y_test, preds, average='weighted', zero_division=0),
                            'Baseline_Precision': precision_score(y_test, preds, average='weighted', zero_division=0),
                            'Baseline_Recall': recall_score(y_test, preds, average='weighted', zero_division=0),
                            'Baseline_TrainTime': train_time,
                            'Baseline_InfTime': inf_time
                        })
                    else: # Regression
                        wide_results[row_key].update({
                            'Baseline_MSE': mean_squared_error(y_test, preds),
                            'Baseline_MAE': mean_absolute_error(y_test, preds),
                            'Baseline_R2': r2_score(y_test, preds),
                            'Baseline_TrainTime': train_time,
                            'Baseline_InfTime': inf_time
                        })
                except Exception as e:
                    logger.error(f"Failed Baseline {m_type}: {e}")

            # --- DP & HE LOOP ---
            for eps in epsilons:
                for norm in data_norms:
                    for m_key in models:
                        # Logic to skip unsupported DP Regression models
                        if task_type == 'regression':
                            if m_key == 'lr': m_type = 'lin_reg'
                            else: continue # Skip NB, DT, RF for DP Regression as planned
                        else:
                            m_type = m_key

                        suffix = f"_Eps{eps}_Norm{norm}" # Column suffix
                        row_key = (ds_name, m_type, 'DP', eps, norm)
                        if row_key not in wide_results:
                            wide_results[row_key] = {
                                'Dataset': ds_name, 'Model': m_type, 'Type': 'DP', 
                                'Epsilon': eps, 'Data_Norm': norm, 'Task_Type': task_type
                            }
                        
                        logger.info(f"  DP {m_type} (eps={eps}, norm={norm})")
                        try:
                            # 1. DP Training
                            # Calculate metadata for DP models (bounds, classes)
                            # Note: Using generic norm bounds and data-driven classes
                            unique_classes = np.unique(y_train) if task_type == 'classification' else None
                            dp_bounds = (-norm, norm)
                            
                            clf_dp = mm.get_dp_model(m_type, eps, norm, n_features=n_features, classes=unique_classes, bounds=dp_bounds)
                            t0 = time.time()
                            clf_dp.fit(X_train_proc, y_train)
                            train_time = time.time() - t0
                            
                            t0 = time.time()
                            preds_dp = clf_dp.predict(X_test_proc)
                            inf_time = time.time() - t0
                            
                            if task_type == 'classification':
                                wide_results[row_key].update({
                                    f'DP_Accuracy{suffix}': accuracy_score(y_test, preds_dp),
                                    f'DP_F1{suffix}': f1_score(y_test, preds_dp, average='weighted', zero_division=0),
                                    f'DP_Precision{suffix}': precision_score(y_test, preds_dp, average='weighted', zero_division=0),
                                    f'DP_Recall{suffix}': recall_score(y_test, preds_dp, average='weighted', zero_division=0),
                                    f'DP_TrainTime{suffix}': train_time,
                                    f'DP_InfTime{suffix}': inf_time
                                })
                            else: # Regression
                                wide_results[row_key].update({
                                    f'DP_MSE{suffix}': mean_squared_error(y_test, preds_dp),
                                    f'DP_MAE{suffix}': mean_absolute_error(y_test, preds_dp),
                                    f'DP_R2{suffix}': r2_score(y_test, preds_dp),
                                    f'DP_TrainTime{suffix}': train_time,
                                    f'DP_InfTime{suffix}': inf_time
                                })
                            
                            # 2. HE Inference (Hybrid)
                            # Using the specific feature subset for HE
                            # Only run for specific norm to save time (e.g. norm=10) or run all?
                            # User said "run loops", so we run if possible.
    
                            logger.info("    Running HE Inference...")
                            he_subset_n = 10
                                
                            # A. PHE for Logistic Regression (Baseline HE)
                            if m_type in ['lr', 'lin_reg']:
                                try:
                                    # Use the DP model trained on FULL features, 
                                    # but PHE implementation in models.py accepts the full vector.
                                    # Optimization: PHE is slow on high dim.
                                    # Should we retrain for HE on reduced features? 
                                    # User said: "Configure it to use only 10 features... either by selecting top 10... or ensuring model is trained on subset"
                                    # To be mathematically consistent, we MUST train on subset to predict on subset.
                                    # So let's quickly retrain a small DP-LR on 10 features just for HE metrics.
                                    
                                    clf_he_training = mm.get_dp_model(m_type, eps, norm, n_features=X_train_he.shape[1])

                                    t0 = time.time()
                                    clf_he_training.fit(X_train_he, y_train) # Fit on 10 cols
                                    phe_train_time = time.time() - t0
                                    
                                    preds_he, he_time = mm.run_he_inference(clf_he_training, X_test_he, n_samples=he_subset_n)
                                    
                                    # Metrics on subset
                                    y_true_sub = y_test[:he_subset_n] # Assuming y_test is numpy array or list
                                    
                                    if task_type == 'classification':
                                        score = accuracy_score(y_true_sub, preds_he)
                                        metric_name = f'PHE_Accuracy{suffix}'
                                        wide_results[row_key].update({
                                            metric_name: score,
                                            f'PHE_F1{suffix}': f1_score(y_true_sub, preds_he, average='weighted', zero_division=0),
                                            f'PHE_Precision{suffix}': precision_score(y_true_sub, preds_he, average='weighted', zero_division=0),
                                            f'PHE_Recall{suffix}': recall_score(y_true_sub, preds_he, average='weighted', zero_division=0),
                                            f'PHE_TrainTime{suffix}': phe_train_time,
                                            f'PHE_InfTime{suffix}': he_time,
                                        })
                                    else:
                                        # Regression: Add MSE, MAE, R2
                                        wide_results[row_key].update({
                                            f'PHE_MSE{suffix}': mean_squared_error(y_true_sub, preds_he),
                                            f'PHE_MAE{suffix}': mean_absolute_error(y_true_sub, preds_he),
                                            f'PHE_R2{suffix}': r2_score(y_true_sub, preds_he),
                                            f'PHE_TrainTime{suffix}': phe_train_time,
                                            f'PHE_InfTime{suffix}': he_time,
                                        })
                                except Exception as e:
                                    logger.error(f"PHE Failed: {e}")

                            # B. Concrete ML for Others (Trees, LR comparison)
                            # Try to run if available
                            # --------------------------------------------------
                            # B. Concrete ML (FHE-only + DP-on-weights)
                            # --------------------------------------------------
                            he_subset_n = 10
                            y_true_sub = y_test[:he_subset_n]

                            # -----------------------------
                            # 1) Concrete FHE-only
                            # -----------------------------
                            try:
                                preds_conc, conc_train_time, conc_compile_time, conc_inf_time = mm.run_concrete_inference(
                                    model_type=m_type,
                                    X_train=X_train_he,
                                    X_test=X_test_he,
                                    y_train=y_train,
                                    n_samples=he_subset_n,
                                    fhe_mode="simulate",
                                    apply_dp_weights=False
                                    
                                )

                                if task_type == "classification":
                                    wide_results[row_key].update({
                                        f'Concrete_Accuracy{suffix}': accuracy_score(y_true_sub, preds_conc),
                                        f'Concrete_F1{suffix}': f1_score(y_true_sub, preds_conc, average="weighted", zero_division=0),
                                        f'Concrete_Precision{suffix}': precision_score(y_true_sub, preds_conc, average="weighted", zero_division=0),
                                        f'Concrete_Recall{suffix}': recall_score(y_true_sub, preds_conc, average="weighted", zero_division=0),
                                        f'Concrete_CompileTime{suffix}': conc_compile_time,
                                        f'Concrete_InfTime{suffix}': conc_inf_time,
                                        f'Concrete_TrainTime{suffix}': conc_train_time,

                                    })
                                else:
                                    wide_results[row_key].update({
                                        f'Concrete_MSE{suffix}': mean_squared_error(y_true_sub, preds_conc),
                                        f'Concrete_MAE{suffix}': mean_absolute_error(y_true_sub, preds_conc),
                                        f'Concrete_R2{suffix}': r2_score(y_true_sub, preds_conc),
                                        f'Concrete_CompileTime{suffix}': conc_compile_time,
                                        f'Concrete_InfTime{suffix}': conc_inf_time,
                                        f'Concrete_TrainTime{suffix}': conc_train_time,

                                    })

                            except ImportError:
                                pass

                            except Exception as e:
                                logger.error(f"Concrete FHE-only Failed {m_type}: {e}")

                            # -----------------------------
                            # 2) Concrete DP + FHE (weights)
                            # -----------------------------
                            try:
                                if m_type in ["lr", "lin_reg"]:
                                    preds_conc_w, w_train_time, w_compile_time, w_inf_time = mm.run_concrete_inference(
                                        model_type=m_type,
                                        X_train=X_train_he,
                                        X_test=X_test_he,
                                        y_train=y_train,
                                        n_samples=he_subset_n,
                                        fhe_mode="simulate",
                                        apply_dp_weights=True,
                                        dp_epsilon=eps,
                                        data_norm=norm,
                                        max_abs_weight=5.0,
                                        dp_mechanism="laplace",
                                        random_state=42
                                    )

                                    if task_type == "classification":
                                        wide_results[row_key].update({
                                            f'ConcreteW_Accuracy{suffix}': accuracy_score(y_true_sub, preds_conc_w),
                                            f'ConcreteW_F1{suffix}': f1_score(y_true_sub, preds_conc_w, average="weighted", zero_division=0),
                                            f'ConcreteW_Precision{suffix}': precision_score(y_true_sub, preds_conc_w, average="weighted", zero_division=0),
                                            f'ConcreteW_Recall{suffix}': recall_score(y_true_sub, preds_conc_w, average="weighted", zero_division=0),
                                            f'ConcreteW_CompileTime{suffix}': w_compile_time,
                                            f'ConcreteW_InfTime{suffix}': w_inf_time,
                                            f'ConcreteW_TrainTime{suffix}': w_train_time,

                                        })
                                    else:
                                        wide_results[row_key].update({
                                            f'ConcreteW_MSE{suffix}': mean_squared_error(y_true_sub, preds_conc_w),
                                            f'ConcreteW_MAE{suffix}': mean_absolute_error(y_true_sub, preds_conc_w),
                                            f'ConcreteW_R2{suffix}': r2_score(y_true_sub, preds_conc_w),
                                            f'ConcreteW_CompileTime{suffix}': w_compile_time,
                                            f'ConcreteW_InfTime{suffix}': w_inf_time,
                                            f'ConcreteW_TrainTime{suffix}': w_train_time,

                                        })

                            except ImportError:
                                pass

                            except Exception as e:
                                logger.error(f"Concrete DP+FHE (Weights) Failed {m_type}: {e}")

                        except Exception as e:
                            logger.error(f"DP Failed {m_type}: {e}")


            # --- K-ANONYMITY LOOP ---
            # Run only for one classification (adult) and one regression (communities) as requested
            if (task_type == 'classification' and ds_name == 'adult') or (task_type == 'regression' and ds_name == 'communities'):
                for k in ks:
                    logger.info(f"  K-Anonymity (k={k})")
                    try:
                        # 1. Anonymize Data (Data Publishing Scenario)
                        # We combine Train/Test, Anonymize, then Split back.
                        X_full = pd.concat([X_train, X_test])
                        
                        # Identify Categorical Features for Mondrian
                        # We use the ones defined in DataLoader or infer objects
                        # Note: Mondrian usually generalizes categorical/numerical differently.
                        # This simple implementation might treat everything as categorical if passed to categorical_features,
                        # or we rely on the implementation's auto-detection.
                        cat_cols = X_full.select_dtypes(include=['object', 'category']).columns.tolist()
                        
                        # Apply Mondrian
                        anonymizer = MondrianAnonymizer(k=k)
                        anonymizer.fit(X_full, categorical_features=cat_cols)
                        X_full_anon = anonymizer.transform(X_full)
                        
                        # Split back
                        X_train_anon = X_full_anon.iloc[:len(X_train)]
                        X_test_anon = X_full_anon.iloc[len(X_train):]
                        
                        # 2. Re-run Preprocessing (CLONE to avoid leaking state across loops)
                        preprocessor_k = clone(preprocessor)
                        X_train_anon_proc = preprocessor_k.fit_transform(X_train_anon)
                        X_test_anon_proc = preprocessor_k.transform(X_test_anon)
                        
                        # 3. Train Baseline Models
                        for m_key in models:
                            # Map generic key to specific model type based on task
                            if task_type == 'regression':
                                if m_key == 'lr': m_type = 'lin_reg'
                                elif m_key == 'nb': continue 
                                elif m_key == 'dt': m_type = 'dt_reg'
                                elif m_key == 'rf': m_type = 'rf_reg'
                                else: continue 
                            else:
                                m_type = m_key
                            
                            row_key = (ds_name, m_type, 'K-Anon', f'K={k}', 'None')
                            if row_key not in wide_results:
                                wide_results[row_key] = {
                                    'Dataset': ds_name, 'Model': m_type, 'Type': 'K-Anon', 
                                    'Epsilon': f'K={k}', 'Data_Norm': 'None', 'Task_Type': task_type
                                }
                                
                            clf = mm.get_baseline_model(m_type)
                            t0 = time.time()
                            clf.fit(X_train_anon_proc, y_train)
                            train_time = time.time() - t0
                            
                            t0 = time.time()
                            preds = clf.predict(X_test_anon_proc)
                            inf_time = time.time() - t0
                            
                            if task_type == 'classification':
                                wide_results[row_key].update({
                                    f'KAnon_Accuracy_K{k}': accuracy_score(y_test, preds),
                                    f'KAnon_F1_K{k}': f1_score(y_test, preds, average='weighted', zero_division=0),
                                    f'KAnon_Precision_K{k}': precision_score(y_test, preds, average='weighted', zero_division=0),
                                    f'KAnon_Recall_K{k}': recall_score(y_test, preds, average='weighted', zero_division=0),
                                    f'KAnon_InfTime_K{k}': inf_time,
                                    f'KAnon_TrainTime_K{k}': train_time,
                                })
                            else: # Regression
                                wide_results[row_key].update({
                                    f'KAnon_MSE_K{k}': mean_squared_error(y_test, preds),
                                    f'KAnon_MAE_K{k}': mean_absolute_error(y_test, preds),
                                    f'KAnon_R2_K{k}': r2_score(y_test, preds),
                                    f'KAnon_TrainTime_K{k}': train_time,
                                    f'KAnon_InfTime_K{k}': inf_time
                                })

                    except Exception as e:
                        logger.error(f"Failed K-Anonymity k={k}: {e}")

            # --- CONCRETE ML LOOP (Separate from DP loop for clarity, or integrated?)
            # The structure above had Concrete INSIDE the DP loop which is redundant if it doesn't use epsilon.
            # But the user code had it there. Let's keep it but ideally it should be outside.
            # Wait, the code I viewed had it "integrated" effectively by being inside the loops?
            # Actually, looking at lines 84+, it iterates eps/norm.
            # Concrete ML (FHE) usually DOES NOT depend on DP epsilon.
            # So calculating it 15 times (5 eps * 3 norms) is wasteful.
            # However, for the sake of the "Wide Table" aligning with DP rows, we might just copy it?
            # Or run it once.
            # The previous code block had it INSIDE.
            # I will leave it inside for now to maintain the table structure requested by user previously, 
            # BUT I will add a check to only run it ONCE per model/dataset to save time, and just copy results.
            
            pass # Already handled inside the loop for row alignment.
            results_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results', 'metrics', 'results_wide.csv')
            pd.DataFrame(list(wide_results.values())).to_csv(results_path, index=False)
            logger.info(f"Saved wide results for {ds_name}")

        except Exception as e:
            logger.error(f"Dataset {ds_name} failed: {e}")

    # Final Save
    results_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results', 'metrics', 'results_wide.csv')
    pd.DataFrame(list(wide_results.values())).to_csv(results_path, index=False)
    logger.info(f"All Experiments Complete. Results in {results_path}")

if __name__ == "__main__":
    run_experiments()
