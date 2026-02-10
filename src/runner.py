import os
import time
import pandas as pd
import numpy as np

from .data_loader import DataLoader
from .models import ModelManager
from .utils import get_logger

from .common import make_suffix, get_unique_classes, get_bounds, slice_he_features
from .baseline import run_baseline
from .dp import run_dp
from .dp_phe import run_dp_phe
from .concrete import run_concrete_fhe_only, run_concrete_dp_weights
from .k_anonymity import run_k_anonymity_block

logger = get_logger(__name__)

def run_experiments():
    dl = DataLoader()
    mm = ModelManager()

    datasets = ['adult', 'data', 'cervical', 'compas', 'creditcard', 'heart', 'insurance', 'communities']
    epsilons = [0.1, 0.5, 1.0, 5.0, 10.0]
    data_norms = [1.0, 10, 100]
    ks = [2, 10, 50, 100]
    models = ['lr', 'nb', 'dt', 'rf']

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

            # HE/FHE feature subset (top 10 columns)
            he_n_features_limit = 10
            X_train_he, X_test_he, full_n_features, he_used = slice_he_features(
                X_train_proc, X_test_proc, he_n_features_limit=he_n_features_limit
            )
            if full_n_features > he_n_features_limit:
                logger.info(f"  HE Optimization: Selected top {he_n_features_limit} features from {full_n_features}")

            # --- BASELINE LOOP ---
            for m_key in models:
                if task_type == 'regression':
                    if m_key == 'lr': m_type = 'lin_reg'
                    elif m_key == 'nb': 
                        continue
                    elif m_key == 'dt': m_type = 'dt_reg'
                    elif m_key == 'rf': m_type = 'rf_reg'
                    else:
                        continue
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
                    base_metrics = run_baseline(mm, m_type, X_train_proc, X_test_proc, y_train, y_test, task_type)
                    wide_results[row_key].update(base_metrics)
                except Exception as e:
                    logger.error(f"Failed Baseline {m_type}: {e}")

            # --- DP & HE LOOP ---
            for eps in epsilons:
                for norm in data_norms:
                    for m_key in models:

                        # DP supports: classification lr/nb/dt/rf ; regression only lin_reg (as in your code)
                        if task_type == 'regression':
                            if m_key == 'lr': m_type = 'lin_reg'
                            else:
                                continue
                        else:
                            m_type = m_key

                        suffix = make_suffix(eps, norm)
                        row_key = (ds_name, m_type, 'DP', eps, norm)
                        if row_key not in wide_results:
                            wide_results[row_key] = {
                                'Dataset': ds_name, 'Model': m_type, 'Type': 'DP',
                                'Epsilon': eps, 'Data_Norm': norm, 'Task_Type': task_type
                            }

                        logger.info(f"  DP {m_type} (eps={eps}, norm={norm})")

                        try:
                            unique_classes = get_unique_classes(y_train, task_type)
                            dp_bounds = get_bounds(norm)

                            # 1) DP Train + Predict (full features)
                            clf_dp, dp_metrics = run_dp(
                                mm=mm, m_type=m_type, eps=eps, norm=norm,
                                n_features=n_features, classes=unique_classes, bounds=dp_bounds,
                                X_train_proc=X_train_proc, X_test_proc=X_test_proc,
                                y_train=y_train, y_test=y_test,
                                task_type=task_type, suffix=suffix
                            )
                            wide_results[row_key].update(dp_metrics)

                            # 2) PHE (DP + PHE) pe subset 10 features
                            logger.info("    Running HE Inference...")
                            try:
                                phe_metrics = run_dp_phe(
                                    mm=mm, m_type=m_type, eps=eps, norm=norm,
                                    X_train_he=X_train_he, X_test_he=X_test_he,
                                    y_train=y_train, y_test=y_test,
                                    task_type=task_type, suffix=suffix, he_subset_n=10
                                )
                                wide_results[row_key].update(phe_metrics)
                            except Exception as e:
                                logger.error(f"PHE Failed: {e}")

                            # 3) Concrete ML (FHE-only) + ConcreteW (DP+weights)
                            # EXACT ca în codul tău: rămâne în bucla eps/norm
                            try:
                                conc_metrics = run_concrete_fhe_only(
                                    mm=mm, m_type=m_type,
                                    X_train_he=X_train_he, X_test_he=X_test_he,
                                    y_train=y_train, y_test=y_test,
                                    task_type=task_type, suffix=suffix, he_subset_n=10
                                )
                                wide_results[row_key].update(conc_metrics)
                            except ImportError:
                                pass
                            except Exception as e:
                                logger.error(f"Concrete FHE-only Failed {m_type}: {e}")

                            try:
                                concw_metrics = run_concrete_dp_weights(
                                    mm=mm, m_type=m_type, eps=eps, norm=norm,
                                    X_train_he=X_train_he, X_test_he=X_test_he,
                                    y_train=y_train, y_test=y_test,
                                    task_type=task_type, suffix=suffix, he_subset_n=10
                                )
                                wide_results[row_key].update(concw_metrics)
                            except ImportError:
                                pass
                            except Exception as e:
                                logger.error(f"Concrete DP+FHE (Weights) Failed {m_type}: {e}")

                        except Exception as e:
                            logger.error(f"DP Failed {m_type}: {e}")

            # --- K-ANONYMITY BLOCK ---
            try:
                run_k_anonymity_block(
                    mm=mm,
                    preprocessor=preprocessor,
                    X_train=X_train, X_test=X_test,
                    y_train=y_train, y_test=y_test,
                    task_type=task_type,
                    models=models,
                    ks=ks,
                    wide_results=wide_results,
                    ds_name=ds_name
                )
            except Exception as e:
                logger.error(f"K-Anonymity block failed: {e}")

            # Save after each dataset (exact ca la tine)
            results_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                'results', 'metrics', 'results_wide.csv'
            )
            pd.DataFrame(list(wide_results.values())).to_csv(results_path, index=False)
            logger.info(f"Saved wide results for {ds_name}")

        except Exception as e:
            logger.error(f"Dataset {ds_name} failed: {e}")

    # Final save
    results_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'results', 'metrics', 'results_wide.csv'
    )
    pd.DataFrame(list(wide_results.values())).to_csv(results_path, index=False)
    logger.info(f"All Experiments Complete. Results in {results_path}")
