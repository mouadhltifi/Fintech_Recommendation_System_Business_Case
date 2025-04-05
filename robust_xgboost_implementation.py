# PART 4: ROBUST XGBOOST FOR FINANCIAL INVESTMENT PREDICTION

print("\n===== Building Robust XGBoost Models for Financial Prediction =====")

# Import necessary libraries
import xgboost as xgb
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, fbeta_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.utils import class_weight, resample
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
from tqdm import tqdm  # Use regular tqdm instead of tqdm.auto
import sys
import psutil
import gc
import logging
import datetime
import json
import platform
import math
from contextlib import contextmanager
from functools import partial
import warnings
warnings.filterwarnings('ignore')

# Enable interactive plotting for Jupyter notebooks
plt.ion()

# Disable HTML output - use terminal progress bars only
USE_HTML_DISPLAY = False

# ===== OPTIMIZATION LOGGING SETUP =====
# Create directories for saving models, plots, and logs
for directory in ['xgb_models', 'xgb_plots', 'xgb_logs']:
    if not os.path.exists(directory):
        os.makedirs(directory)

# Create optimization log file
log_filename = f"xgb_logs/optimization_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('XGBoostOptimization')

# Add console handler for immediate feedback - only use with minimal messages
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter('%(message)s')
console_handler.setFormatter(console_formatter)
logger.addHandler(console_handler)

# Add system info logging
def log_system_info():
    """Log detailed system information for optimization reference"""
    system_info = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "xgboost_version": xgb.__version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "memory_total": psutil.virtual_memory().total / (1024**3),  # GB
        "cpu_count": psutil.cpu_count(logical=False),
        "cpu_threads": psutil.cpu_count(logical=True)
    }
    
    logger.info(f"SYSTEM INFO: {json.dumps(system_info)}")
    print("System info logged for optimization reference")

# Time profiling context manager
@contextmanager
def timer_log(name):
    """Context manager to time operations and log them"""
    start_time = time.time()
    memory_before = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    
    try:
        yield
    finally:
        end_time = time.time()
        duration = end_time - start_time
        memory_after = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
        memory_diff = memory_after - memory_before
        
        logger.info(f"OPERATION: {name} - Duration: {duration:.2f}s - Memory: {memory_diff:.2f}MB")

# Log system info at startup
log_system_info()

# Create dictionaries to store XGBoost models
xgb_income_models = {}
xgb_accum_models = {}

# Function to print section header
def print_section(title):
    """Print a section header"""
    print(f"\n{'=' * 80}")
    print(f" {title}")
    print(f"{'=' * 80}")

# Display metrics
def display_metrics(metrics, title=None):
    """Display metrics"""
    if title:
        print(f"\n{title}:")
    for metric, value in metrics.items():
        if isinstance(value, float):
            print(f"{metric.replace('_', ' ').title()}: {value:.4f}")
        else:
            print(f"{metric.replace('_', ' ').title()}: {value}")
            
# Progress bar wrapper for iterations
def progress_bar(iterable=None, total=None, desc="Processing", **kwargs):
    """Create a terminal progress bar"""
    return tqdm(iterable, total=total, desc=desc, **kwargs)

# Enhanced feature engineering
def enhanced_feature_engineering(X):
    """
    DISABLED: Feature engineering disabled - now returning original features only
    To enable advanced features, implement the feature engineering in your preprocessing pipeline
    """
    with timer_log("enhanced_feature_engineering"):
        logger.info("Feature engineering disabled - using original features only")
        return X  # Return the original dataframe without any transformations

# Optimize probability threshold 
def optimize_threshold(y_true, y_pred_proba, metric='f1', target_recall=0.6):
    """Find optimal threshold based on specified metric"""
    with timer_log("threshold_optimization"):
        thresholds = np.linspace(0.1, 0.9, 50)
        best_score = 0
        best_threshold = 0.5
        best_precision = 0
        best_recall = 0
        
        scores = []
        precisions = []
        recalls = []
        
        print("Finding optimal threshold...")
        for threshold in progress_bar(thresholds, desc="Optimizing threshold"):
            y_pred = (y_pred_proba >= threshold).astype(int)
            
            # Calculate precision and recall
            prec = precision_score(y_true, y_pred, zero_division=0)
            rec = recall_score(y_true, y_pred, zero_division=0)
            precisions.append(prec)
            recalls.append(rec)
            
            # Calculate requested metric
            if metric == 'f1':
                score = f1_score(y_true, y_pred, zero_division=0)
            elif metric == 'f2':
                score = fbeta_score(y_true, y_pred, beta=2, zero_division=0)  # Favor recall
            elif metric == 'f0.5':
                score = fbeta_score(y_true, y_pred, beta=0.5, zero_division=0)  # Favor precision
            else:
                # Default to balanced metric
                score = (prec + rec) / 2 if prec + rec > 0 else 0
                
            scores.append(score)
            
            # Only consider thresholds meeting minimum recall if specified
            if target_recall is None or rec >= target_recall:
                if score > best_score:
                    best_score = score
                    best_threshold = threshold
                    best_precision = prec
                    best_recall = rec
        
        logger.info(f"Best threshold: {best_threshold:.4f} with {metric}={best_score:.4f} (precision={best_precision:.4f}, recall={best_recall:.4f})")
        
        # Plot threshold vs metrics
        plt.figure(figsize=(10, 6))
        plt.plot(thresholds, precisions, 'b-', label='Precision')
        plt.plot(thresholds, recalls, 'r-', label='Recall')
        plt.plot(thresholds, scores, 'g-', label=f'{metric.upper()} Score')
        plt.axvline(x=best_threshold, color='k', linestyle='--', label=f'Best threshold: {best_threshold:.4f}')
        if target_recall is not None:
            plt.axhline(y=target_recall, color='r', linestyle=':', label=f'Target recall: {target_recall:.2f}')
        plt.xlabel('Threshold')
        plt.ylabel('Score')
        plt.title(f'Threshold vs Metrics (Best {metric}={best_score:.4f})')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"xgb_plots/threshold_optimization_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png", dpi=200)
        plt.close()
        
        # Display results
        threshold_metrics = {
            'best_threshold': best_threshold,
            f'best_{metric}_score': best_score,
            'precision': best_precision,
            'recall': best_recall
        }
        display_metrics(threshold_metrics, title="Threshold Optimization Results")
        
        return best_threshold

# Hyperparameter tuning with cross-validation for XGBoost
def tune_xgboost_params(X, y, n_trials=20, cv=3):
    """Tune XGBoost hyperparameters using cross-validation"""
    with timer_log("hyperparameter_tuning"):
        try:
            import optuna
            from sklearn.model_selection import cross_val_score
            
            # Use progress bar reporting for Optuna
            class TqdmCallback:
                def __init__(self):
                    self._pbar = None
                    
                def __call__(self, study, trial):
                    if self._pbar is None:
                        self._pbar = tqdm(total=n_trials, desc="Hyperparameter tuning")
                    self._pbar.update(1)
                    if trial.value is not None:
                        self._pbar.set_postfix({"best_score": f"{study.best_value:.4f}"})
                    
                def close(self):
                    if self._pbar:
                        self._pbar.close()
                        self._pbar = None
            
            # Display tuning start info
            print(f"Starting hyperparameter optimization with {n_trials} trials and {cv}-fold CV")
            logger.info(f"Starting hyperparameter optimization with {n_trials} trials and {cv}-fold CV")
            
            # Calculate class imbalance for scale_pos_weight
            neg_pos_ratio = np.sum(y == 0) / np.sum(y == 1) if np.sum(y == 1) > 0 else 1.0
            
            def objective(trial):
                param = {
                    'objective': 'binary:logistic',
                    'eval_metric': 'auc',
                    'tree_method': 'hist',  # Efficient for large datasets
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'max_depth': trial.suggest_int('max_depth', 3, 8),
                    'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                    'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
                    'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True),
                    'scale_pos_weight': trial.suggest_float('scale_pos_weight', 
                                                          0.5 * neg_pos_ratio, 1.5 * neg_pos_ratio),
                    'n_estimators': 300  # Will be controlled by early stopping
                }
                
                model = xgb.XGBClassifier(**param)
                
                # Use cross-validation for more robust evaluation
                scores = cross_val_score(
                    model, X, y, 
                    cv=cv, 
                    scoring='roc_auc',
                    n_jobs=-1
                )
                
                return scores.mean()
            
            # Set up callback for progress reporting
            tqdm_callback = TqdmCallback()
            
            # Run optimization
            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=n_trials, callbacks=[tqdm_callback])
            tqdm_callback.close()
            
            # Get best parameters
            best_params = study.best_params
            best_params['n_estimators'] = 500  # Will be limited by early stopping
            best_params['objective'] = 'binary:logistic'
            best_params['eval_metric'] = 'auc'
            best_params['tree_method'] = 'hist'
            
            logger.info(f"Best hyperparameters: {best_params}")
            print(f"Tuning completed with best score: {study.best_value:.4f}")
            
            # Try to plot optimization history
            try:
                plt.figure(figsize=(10, 6))
                optuna.visualization.matplotlib.plot_optimization_history(study)
                plt.title('Hyperparameter Optimization History')
                plt.tight_layout()
                plt.savefig(f"xgb_plots/optuna_history_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png", dpi=200)
                plt.close()
            except:
                logger.info("Optimization history plot failed")
            
            return best_params
            
        except ImportError:
            logger.info("Optuna not available, using default hyperparameters")
            print("Optuna not available, using default parameters")
                
            # Default parameters if optuna is not available
            default_params = {
                'learning_rate': 0.1,
                'max_depth': 6,
                'min_child_weight': 1,
                'gamma': 0,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0,
                'reg_lambda': 1,
                'scale_pos_weight': neg_pos_ratio,
                'n_estimators': 300,
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
                'tree_method': 'hist'
            }
            return default_params
            
        except Exception as e:
            logger.error(f"Error in hyperparameter tuning: {str(e)}")
            print(f"Error in hyperparameter tuning: {str(e)}")
                
            # Default parameters if there's an error
            return {
                'learning_rate': 0.1,
                'max_depth': 6,
                'min_child_weight': 1,
                'gamma': 0,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0,
                'reg_lambda': 1,
                'scale_pos_weight': neg_pos_ratio,
                'n_estimators': 300,
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
                'tree_method': 'hist'
            }

# Create an ensemble of XGBoost models
def create_xgboost_ensemble(X_train, y_train, X_val, y_val, params, n_models=3):
    """Create an ensemble of XGBoost models with different seeds"""
    with timer_log("ensemble_creation"):
        models = []
        val_predictions = []
        
        # Use progress bar for model training
        with progress_bar(total=n_models, desc="Training ensemble") as pbar:
            for i in range(n_models):
                # Create model with different seed
                model_params = params.copy()
                model_params['random_state'] = 42 + i*101
                
                # Vary learning rate slightly to create diversity
                model_params['learning_rate'] = params['learning_rate'] * (0.8 + 0.4*i/n_models)
                
                # Create model
                model = xgb.XGBClassifier(**model_params)
                
                # Train with early stopping
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    early_stopping_rounds=50,
                    verbose=False
                )
                
                # Add to ensemble
                models.append(model)
                val_predictions.append(model.predict_proba(X_val)[:, 1])
                
                # Update progress bar with model info
                pbar.set_postfix({"model": f"{i+1}/{n_models}", "trees": model.get_booster().num_boosted_rounds()})
                pbar.update(1)
                
                logger.info(f"Trained ensemble model {i+1}/{n_models} with seed {model_params['random_state']}")
        
        print(f"Trained ensemble of {len(models)} models")
        
        return {
            'models': models,
            'val_predictions': val_predictions
        }

# Function for evaluating and logging model performance
def evaluate_model(models, X_test, y_test, optimal_threshold=None):
    """Evaluate an ensemble of models and report performance metrics"""
    with timer_log("model_evaluation"):
        # Get ensemble predictions
        test_predictions = []
        
        # Use progress bar for prediction generation
        with progress_bar(total=len(models), desc="Generating predictions") as pbar:
            for i, model in enumerate(models):
                test_predictions.append(model.predict_proba(X_test)[:, 1])
                pbar.update(1)
        
        # Average predictions
        y_pred_proba = np.mean(test_predictions, axis=0)
        
        # Use default threshold if none specified
        if optimal_threshold is None:
            optimal_threshold = 0.5
            
        # Make binary predictions
        y_pred = (y_pred_proba >= optimal_threshold).astype(int)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
        }
        
        # Only calculate AUC if we have both classes in test set
        if len(np.unique(y_test)) > 1:
            metrics['auc'] = roc_auc_score(y_test, y_pred_proba)
            metrics['pr_auc'] = average_precision_score(y_test, y_pred_proba)
        else:
            metrics['auc'] = 0.5  # Default for single class
            metrics['pr_auc'] = 0.5  # Default for single class
        
        logger.info(f"Model evaluation metrics: {json.dumps(metrics)}")
        
        # Display metrics
        display_metrics(metrics, title="Model Performance")
        
        return metrics, y_pred_proba

# Plot feature importance for XGBoost model
def plot_feature_importance(models, feature_names, cluster_id, target_name):
    """Plot and save feature importance from XGBoost models"""
    with timer_log("feature_importance_plot"):
        try:
            # Calculate average feature importance across models
            importance_dict = {}
            
            # Use progress bar for calculating importance
            with progress_bar(total=len(models), desc="Calculating feature importance") as pbar:
                for i, model in enumerate(models):
                    # Get feature importance
                    feature_importance = model.feature_importances_
                    
                    # Add to importance dictionary
                    for i, importance in enumerate(feature_importance):
                        feature_name = feature_names[i]
                        if feature_name in importance_dict:
                            importance_dict[feature_name] += importance
                        else:
                            importance_dict[feature_name] = importance
                    pbar.update(1)
            
            # Average the importance scores
            for feature in importance_dict:
                importance_dict[feature] /= len(models)
            
            # Sort features by importance
            sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
            
            # Plot only top 15 features (if we have that many)
            num_features = min(15, len(sorted_features))
            features = [item[0] for item in sorted_features[:num_features]]
            importances = [item[1] for item in sorted_features[:num_features]]
            
            plt.figure(figsize=(10, 6))
            plt.barh(range(num_features), importances, align='center')
            plt.yticks(range(num_features), features)
            plt.title(f'Feature Importance (Cluster {cluster_id} - {target_name})')
            plt.xlabel('Importance (Gain)')
            plt.tight_layout()
            plt.savefig(f"xgb_plots/cluster_{cluster_id}_{target_name}_importance.png", dpi=200)
            plt.close()
            
            # Display top features
            if len(sorted_features) > 0:
                top_5_features = sorted_features[:5]
                print("\nTop 5 Important Features:")
                for feature, importance in top_5_features:
                    print(f"- {feature}: {importance:.4f}")
            
            # Return the top features and their importance
            return dict(sorted_features[:num_features])
            
        except Exception as e:
            logger.error(f"Error plotting feature importance: {str(e)}")
            print(f"Error plotting feature importance: {str(e)}")
            return {}

# Process each cluster
def process_clusters(df, labels):
    """Process each cluster and train XGBoost models for both target variables"""
    all_results = []
    unique_clusters = np.unique(labels)
    
    # Create outer progress bar for clusters
    with progress_bar(total=len(unique_clusters), desc="Processing clusters") as cluster_pbar:
        for cluster_id in unique_clusters:
            try:
                cluster_pbar.set_description(f"Cluster {cluster_id}")
                logger.info(f"Starting processing for cluster {cluster_id}")
                
                # Filter data for this cluster
                cluster_df = df[df['cluster'] == cluster_id]
                
                # Skip very small clusters
                if len(cluster_df) < 30:
                    logger.info(f"Cluster {cluster_id} has only {len(cluster_df)} samples. Skipping...")
                    print(f"⚠️ Cluster {cluster_id} skipped: only {len(cluster_df)} samples")
                    cluster_pbar.update(1)
                    continue
                
                # For each target variable
                for target_name in ['IncomeInvestment', 'AccumulationInvestment']:
                    with timer_log(f"cluster{cluster_id}_{target_name}_processing"):
                        # Display section header
                        print_section(f"Training XGBoost for {target_name} in Cluster {cluster_id}")
                        logger.info(f"Training model for {target_name} in cluster {cluster_id}")
                        
                        # Create progress bar for processing steps
                        steps = ["Prepare data", "Feature engineering", "Data splitting", "Handle imbalance", 
                                "Scale features", "Tune hyperparameters", "Train ensemble", "Optimize threshold", 
                                "Evaluate model", "Save results"]
                        
                        with progress_bar(total=len(steps), desc=f"Cluster {cluster_id}: {target_name}") as step_pbar:
                            # Step 1: Prepare data
                            step_pbar.set_description("Preparing data")
                            X = cluster_df.drop(['IncomeInvestment', 'AccumulationInvestment', 'cluster'], axis=1)
                            y = cluster_df[target_name]
                            step_pbar.update(1)
                            
                            # Step 2: Feature engineering
                            step_pbar.set_description("Feature engineering")
                            start_time = time.time()
                            X = enhanced_feature_engineering(X)
                            X = pd.get_dummies(X)
                            step_pbar.set_postfix({"shape": str(X.shape), "time": f"{time.time() - start_time:.1f}s"})
                            step_pbar.update(1)
                            
                            # Step 3: Data splitting
                            step_pbar.set_description("Splitting data")
                            X_train_val, X_test, y_train_val, y_test = train_test_split(
                                X, y, test_size=0.2, random_state=42, stratify=y
                            )
                            
                            X_train, X_val, y_train, y_val = train_test_split(
                                X_train_val, y_train_val, test_size=0.25, random_state=42, stratify=y_train_val
                            )
                            
                            class_counts = pd.Series(y_train).value_counts().to_dict()
                            step_pbar.set_postfix({"train": X_train.shape[0], "test": X_test.shape[0]})
                            step_pbar.update(1)
                            
                            # Step 4: Handle imbalanced data
                            step_pbar.set_description("Handling class imbalance")
                            try:
                                from imblearn.over_sampling import SMOTE
                                
                                # Only apply SMOTE if imbalance is significant and dataset is large enough
                                class_ratio = min(
                                    sum(y_train == 0) / len(y_train),
                                    sum(y_train == 1) / len(y_train)
                                )
                                
                                if class_ratio < 0.3 and len(X_train) >= 50:
                                    # Create SMOTE with appropriate sampling strategy
                                    smote = SMOTE(sampling_strategy=0.5, random_state=42)
                                    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
                                    
                                    # Only use if SMOTE didn't create too many samples
                                    if len(X_train_resampled) <= len(X_train) * 3:
                                        X_train = X_train_resampled
                                        y_train = y_train_resampled
                                        step_pbar.set_postfix({"new_size": len(X_train), "ratio": f"{class_ratio:.2f}->{0.5:.2f}"})
                                    else:
                                        step_pbar.set_postfix({"status": "skipped - too many samples"})
                                else:
                                    step_pbar.set_postfix({"ratio": f"{class_ratio:.2f}", "status": "balanced enough"})
                            except Exception as e:
                                logger.info(f"Error applying SMOTE: {str(e)}. Continuing with original data.")
                                step_pbar.set_postfix({"status": "error - using original data"})
                            step_pbar.update(1)
                            
                            # Step 5: Scale features
                            step_pbar.set_description("Scaling features")
                            scaler = StandardScaler()
                            X_train_scaled = scaler.fit_transform(X_train)
                            X_val_scaled = scaler.transform(X_val)
                            X_test_scaled = scaler.transform(X_test)
                            step_pbar.update(1)
                            
                            # Step 6: Tune hyperparameters
                            step_pbar.set_description("Tuning hyperparameters")
                            n_trials = min(20, max(5, len(X_train) // 50))
                            cv_folds = min(5, max(3, len(X_train) // 100))
                            
                            best_params = tune_xgboost_params(
                                X_train_scaled, y_train, 
                                n_trials=n_trials,
                                cv=cv_folds
                            )
                            step_pbar.set_postfix({"trials": n_trials, "cv": cv_folds})
                            step_pbar.update(1)
                            
                            # Step 7: Create ensemble
                            step_pbar.set_description("Training ensemble")
                            n_models = 3 if len(X_train) >= 100 else 1
                            ensemble = create_xgboost_ensemble(
                                X_train_scaled, y_train,
                                X_val_scaled, y_val,
                                best_params,
                                n_models=n_models
                            )
                            step_pbar.set_postfix({"models": n_models})
                            step_pbar.update(1)
                            
                            # Step 8: Optimize threshold
                            step_pbar.set_description("Optimizing threshold")
                            val_predictions_mean = np.mean(ensemble['val_predictions'], axis=0)
                            target_recall = 0.7 if target_name == 'IncomeInvestment' else 0.6
                            
                            optimal_threshold = optimize_threshold(
                                y_val, val_predictions_mean, 
                                metric='f1', 
                                target_recall=target_recall
                            )
                            step_pbar.set_postfix({"threshold": f"{optimal_threshold:.3f}"})
                            step_pbar.update(1)
                            
                            # Step 9: Evaluate model
                            step_pbar.set_description("Evaluating model")
                            metrics, test_predictions = evaluate_model(
                                ensemble['models'], 
                                X_test_scaled, 
                                y_test,
                                optimal_threshold
                            )
                            step_pbar.set_postfix({"f1": f"{metrics['f1']:.3f}"})
                            step_pbar.update(1)
                            
                            # Step 10: Plot feature importance and save results
                            step_pbar.set_description("Saving results")
                            
                            # Plot feature importance
                            top_features = plot_feature_importance(
                                ensemble['models'],
                                X.columns.tolist(),
                                cluster_id,
                                target_name
                            )
                            
                            # Save models and metadata
                            model_dir = f"xgb_models/cluster_{cluster_id}_{target_name}"
                            if not os.path.exists(model_dir):
                                os.makedirs(model_dir)
                            
                            # Save each model in the ensemble
                            for i, model in enumerate(ensemble['models']):
                                model.save_model(f"{model_dir}/model{i}.json")
                            
                            # Save threshold
                            with open(f"{model_dir}/threshold.json", 'w') as f:
                                json.dump({'threshold': float(optimal_threshold)}, f)
                            
                            # Save feature names
                            with open(f"{model_dir}/feature_names.json", 'w') as f:
                                json.dump(X.columns.tolist(), f)
                            
                            # Save scaler
                            from joblib import dump
                            dump(scaler, f"{model_dir}/scaler.joblib")
                            
                            # Store model info
                            model_info = {
                                'models': ensemble['models'],
                                'feature_names': X.columns.tolist(),
                                'optimal_threshold': optimal_threshold,
                                'metrics': metrics,
                                'top_features': top_features,
                                'scaler': scaler
                            }
                            
                            if target_name == 'IncomeInvestment':
                                xgb_income_models[cluster_id] = model_info
                            else:
                                xgb_accum_models[cluster_id] = model_info
                            
                            # Store results for comparison
                            all_results.append({
                                'Model Type': target_name,
                                'Cluster': f'Cluster {cluster_id}',
                                'Accuracy': metrics['accuracy'],
                                'Precision': metrics['precision'],
                                'Recall': metrics['recall'],
                                'F1 Score': metrics['f1'],
                                'AUC': metrics['auc'],
                                'PR AUC': metrics['pr_auc'],
                                'Threshold': optimal_threshold
                            })
                            
                            # Clear memory
                            gc.collect()
                            step_pbar.update(1)
                
                # Update overall cluster progress
                cluster_pbar.update(1)
                
            except Exception as e:
                logger.error(f"Error in cluster {cluster_id}: {str(e)}")
                print(f"Error processing cluster {cluster_id}: {str(e)}")
                import traceback
                traceback.print_exc()
                logger.error(traceback.format_exc())
                cluster_pbar.update(1)
                continue
    
    return all_results

# Train a global model across all clusters
def train_global_models(df):
    """Train global XGBoost models for both target variables"""
    global_results = []
    
    # Create outer progress bar for global models
    with progress_bar(total=2, desc="Training global models") as global_pbar:
        for target_name in ['IncomeInvestment', 'AccumulationInvestment']:
            try:
                global_pbar.set_description(f"Global: {target_name}")
                print_section(f"Training Global XGBoost Model for {target_name}")
                logger.info(f"Training global model for {target_name}")
                
                with timer_log(f"global_{target_name}_processing"):
                    # Create progress bar for processing steps
                    steps = ["Prepare data", "Feature engineering", "Data splitting", 
                             "Scale features", "Tune hyperparameters", "Train ensemble", 
                             "Optimize threshold", "Evaluate model", "Save results"]
                    
                    with progress_bar(total=len(steps), desc=f"Global: {target_name}") as step_pbar:
                        # Step 1: Prepare data
                        step_pbar.set_description("Preparing data")
                        X = df.drop(['IncomeInvestment', 'AccumulationInvestment', 'cluster'], axis=1)
                        y = df[target_name]
                        step_pbar.update(1)
                        
                        # Step 2: Feature engineering
                        step_pbar.set_description("Feature engineering")
                        start_time = time.time()
                        X = enhanced_feature_engineering(X)
                        X = pd.get_dummies(X)
                        step_pbar.set_postfix({"shape": str(X.shape), "time": f"{time.time() - start_time:.1f}s"})
                        step_pbar.update(1)
                        
                        # Step 3: Data splitting
                        step_pbar.set_description("Splitting data")
                        X_train_val, X_test, y_train_val, y_test = train_test_split(
                            X, y, test_size=0.2, random_state=42, stratify=y
                        )
                        
                        X_train, X_val, y_train, y_val = train_test_split(
                            X_train_val, y_train_val, test_size=0.25, random_state=42, stratify=y_train_val
                        )
                        
                        class_counts = pd.Series(y_train).value_counts().to_dict()
                        step_pbar.set_postfix({"train": X_train.shape[0], "val": X_val.shape[0], "test": X_test.shape[0]})
                        step_pbar.update(1)
                        
                        # Step 4: Scale features
                        step_pbar.set_description("Scaling features")
                        scaler = StandardScaler()
                        X_train_scaled = scaler.fit_transform(X_train)
                        X_val_scaled = scaler.transform(X_val)
                        X_test_scaled = scaler.transform(X_test)
                        step_pbar.update(1)
                        
                        # Step 5: Tune hyperparameters
                        step_pbar.set_description("Tuning hyperparameters")
                        best_params = tune_xgboost_params(
                            X_train_scaled, y_train, 
                            n_trials=30,
                            cv=5
                        )
                        step_pbar.update(1)
                        
                        # Step 6: Create ensemble
                        step_pbar.set_description("Training ensemble")
                        ensemble = create_xgboost_ensemble(
                            X_train_scaled, y_train,
                            X_val_scaled, y_val,
                            best_params,
                            n_models=5
                        )
                        step_pbar.update(1)
                        
                        # Step 7: Optimize threshold
                        step_pbar.set_description("Optimizing threshold")
                        val_predictions_mean = np.mean(ensemble['val_predictions'], axis=0)
                        
                        optimal_threshold = optimize_threshold(
                            y_val, val_predictions_mean, 
                            metric='f1', 
                            target_recall=0.6
                        )
                        step_pbar.set_postfix({"threshold": f"{optimal_threshold:.3f}"})
                        step_pbar.update(1)
                        
                        # Step 8: Evaluate model
                        step_pbar.set_description("Evaluating model")
                        metrics, test_predictions = evaluate_model(
                            ensemble['models'], 
                            X_test_scaled, 
                            y_test,
                            optimal_threshold
                        )
                        step_pbar.set_postfix({"f1": f"{metrics['f1']:.3f}"})
                        step_pbar.update(1)
                        
                        # Step 9: Plot feature importance and save results
                        step_pbar.set_description("Saving results")
                        
                        # Plot feature importance
                        top_features = plot_feature_importance(
                            ensemble['models'],
                            X.columns.tolist(),
                            'global',
                            target_name
                        )
                        
                        # Save models and metadata
                        model_dir = f"xgb_models/global_{target_name}"
                        if not os.path.exists(model_dir):
                            os.makedirs(model_dir)
                        
                        # Save each model in the ensemble
                        for i, model in enumerate(ensemble['models']):
                            model.save_model(f"{model_dir}/model{i}.json")
                        
                        # Save threshold
                        with open(f"{model_dir}/threshold.json", 'w') as f:
                            json.dump({'threshold': float(optimal_threshold)}, f)
                        
                        # Save feature names
                        with open(f"{model_dir}/feature_names.json", 'w') as f:
                            json.dump(X.columns.tolist(), f)
                        
                        # Save scaler
                        from joblib import dump
                        dump(scaler, f"{model_dir}/scaler.joblib")
                        
                        # Store model info
                        model_info = {
                            'models': ensemble['models'],
                            'feature_names': X.columns.tolist(),
                            'optimal_threshold': optimal_threshold,
                            'metrics': metrics,
                            'top_features': top_features,
                            'scaler': scaler
                        }
                        
                        if target_name == 'IncomeInvestment':
                            xgb_income_models['global'] = model_info
                        else:
                            xgb_accum_models['global'] = model_info
                        
                        # Store results for comparison
                        global_results.append({
                            'Model Type': target_name,
                            'Cluster': 'Global',
                            'Accuracy': metrics['accuracy'],
                            'Precision': metrics['precision'],
                            'Recall': metrics['recall'],
                            'F1 Score': metrics['f1'],
                            'AUC': metrics['auc'],
                            'PR AUC': metrics['pr_auc'],
                            'Threshold': optimal_threshold
                        })
                        
                        # Clear memory
                        gc.collect()
                        step_pbar.update(1)
                
                # Update global model progress
                global_pbar.update(1)
                
            except Exception as e:
                logger.error(f"Error in global {target_name} model: {str(e)}")
                print(f"Error processing global {target_name} model: {str(e)}")
                import traceback
                traceback.print_exc()
                logger.error(traceback.format_exc())
                global_pbar.update(1)
                continue
    
    return global_results

# Create visualizations for model comparison
def create_model_visualizations(all_results):
    """Create visualizations comparing all models"""
    with timer_log("creating_visualizations"):
        if not all_results:
            logger.info("No results available for visualization")
            print("No results available for visualization")
            return None
        
        print(f"Creating visualizations for {len(all_results)} models...")
        
        # Convert to DataFrame
        results_df = pd.DataFrame(all_results)
        
        # Save results to CSV
        results_csv = f"xgb_logs/model_results_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        results_df.to_csv(results_csv, index=False)
        
        # Create visualizations
        try:
            # Create progress bar for visualization steps
            viz_steps = ["Comparison plots", "Heatmap", "Summary statistics"]
            with progress_bar(total=len(viz_steps), desc="Creating visualizations") as viz_pbar:
                # Step 1: Comparison bar plots
                viz_pbar.set_description("Creating comparison plots")
                plt.figure(figsize=(18, 12))
                
                metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC']
                
                for i, metric in enumerate(metrics_to_plot):
                    plt.subplot(2, 3, i+1)
                    sns.barplot(x='Cluster', y=metric, hue='Model Type', data=results_df)
                    plt.title(f'{metric} by Cluster', fontsize=14)
                    plt.xticks(rotation=45)
                    plt.ylim(0, 1)
                
                # Plot thresholds
                plt.subplot(2, 3, 6)
                sns.barplot(x='Cluster', y='Threshold', hue='Model Type', data=results_df)
                plt.title('Optimization Thresholds', fontsize=14)
                plt.xticks(rotation=45)
                plt.ylim(0, 1)
                
                plt.tight_layout()
                
                comparison_plot = f"xgb_plots/model_comparison_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                plt.savefig(comparison_plot, dpi=300)
                plt.close()
                viz_pbar.update(1)
                
                # Step 2: Create heatmap of F1 scores
                viz_pbar.set_description("Creating heatmap")
                plt.figure(figsize=(14, 8))
                
                # Create pivot table for heatmap
                pivot_table = results_df.pivot_table(
                    index='Cluster', 
                    columns='Model Type', 
                    values='F1 Score'
                )
                
                sns.heatmap(pivot_table, annot=True, cmap='YlGnBu', fmt='.3f', vmin=0, vmax=1)
                plt.title('F1 Score by Cluster and Model Type', fontsize=16)
                plt.tight_layout()
                
                heatmap_file = f"xgb_plots/f1_heatmap_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                plt.savefig(heatmap_file, dpi=300)
                plt.close()
                viz_pbar.update(1)
                
                # Step 3: Calculate and log summary statistics
                viz_pbar.set_description("Computing summary statistics")
                summary_stats = {
                    'models_trained': len(results_df),
                    'avg_metrics': {
                        'accuracy': float(results_df['Accuracy'].mean()),
                        'precision': float(results_df['Precision'].mean()),
                        'recall': float(results_df['Recall'].mean()),
                        'f1': float(results_df['F1 Score'].mean()),
                        'auc': float(results_df['AUC'].mean()),
                    },
                    'best_model': results_df.loc[results_df['F1 Score'].idxmax()].to_dict()
                }
                
                logger.info(f"MODEL_SUMMARY: {json.dumps(summary_stats)}")
                
                # Save summary to JSON
                summary_file = f"xgb_logs/model_summary_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                with open(summary_file, 'w') as f:
                    json.dump(summary_stats, f, indent=4)
                
                # Print summary information
                print("\n=== Model Summary ===")
                print(f"Total models trained: {summary_stats['models_trained']}")
                print("\nAverage metrics across all models:")
                for metric, value in summary_stats['avg_metrics'].items():
                    print(f"{metric.title()}: {value:.4f}")
                
                best_model = summary_stats['best_model']
                print(f"\nBest model by F1 score:")
                print(f"Type: {best_model['Model Type']}, Cluster: {best_model['Cluster']}")
                print(f"F1 Score: {best_model['F1 Score']:.4f}, Precision: {best_model['Precision']:.4f}, Recall: {best_model['Recall']:.4f}")
                
                print(f"\nFiles created:")
                print(f"- Results CSV: {results_csv}")
                print(f"- Comparison Plot: {comparison_plot}")
                print(f"- F1 Heatmap: {heatmap_file}")
                print(f"- Summary JSON: {summary_file}")
                
                viz_pbar.update(1)
                
            return summary_stats
            
        except Exception as e:
            logger.error(f"Error creating visualizations: {str(e)}")
            print(f"Error creating visualizations: {str(e)}")
            import traceback
            traceback.print_exc()
            logger.error(traceback.format_exc())
            return None

# Main function with progress bars
def main(df=None, labels=None):
    """Run the entire XGBoost pipeline with progress bars"""
    print_section("XGBoost Modeling Pipeline")
    
    # Set up data
    if df is None or labels is None:
        # Try to get the data from the global scope
        try:
            global engineered_df
            df_for_xgboost = engineered_df.copy()
            df_for_xgboost['cluster'] = labels
        except NameError:
            print("Error: Data not provided and not found in global scope (engineered_df and labels required)")
            return None, None
    else:
        # Use the provided data
        df_for_xgboost = df.copy()
        df_for_xgboost['cluster'] = labels
    
    # Process clusters with progress bars
    with progress_bar(total=3, desc="Overall progress") as pbar:
        pbar.set_description("Processing clusters")
        cluster_results = process_clusters(df_for_xgboost, labels)
        pbar.update(1)
        
        # Train global models with progress bars
        pbar.set_description("Training global models")
        global_results = train_global_models(df_for_xgboost)
        pbar.update(1)
        
        # Create visualizations
        pbar.set_description("Creating visualizations")
        all_results = cluster_results + global_results
        with progress_bar(desc="Creating visualizations", total=1) as viz_pbar:
            summary = create_model_visualizations(all_results)
            viz_pbar.update(1)
        pbar.update(1)
    
    # Display results
    print_section("✅ XGBoost Models Complete")
    print("✅ Cluster-specific XGBoost models have been trained")
    print("✅ Global XGBoost models have been trained")
    print("✅ Model files saved to 'xgb_models/' directory")
    print("✅ Performance visualizations saved to 'xgb_plots/' directory")
    print(f"✅ Detailed logs saved to '{log_filename}'")
    
    return xgb_income_models, xgb_accum_models

# Run the pipeline if executing this file directly
if __name__ == "__main__":
    # Assumes engineered_df and labels variables are already defined in the global scope
    xgb_income_models, xgb_accum_models = main()
