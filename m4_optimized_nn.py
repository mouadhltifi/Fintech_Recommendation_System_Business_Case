# =====================================================================================
# M4 MAX GPU-OPTIMIZED NEURAL NETWORK FOR FINANCIAL PREDICTION (GRAPH MODE)
# =====================================================================================

# Disable eager execution for graph mode
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import os
import random
import warnings
import numpy as np
import pandas as pd

# Set environment variables for reproducibility
os.environ['PYTHONHASHSEED'] = '42'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TensorFlow logging
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# Verify graph mode is active
print("CRITICAL - Using graph mode:", not tf.executing_eagerly())

# Other imports
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
import time
import gc
import joblib
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import (
    roc_auc_score, precision_score, accuracy_score, 
    recall_score, f1_score, confusion_matrix, classification_report
)

# Configure GPU/device settings
print("Configuring TensorFlow for M4 Max GPU...")

# Configure GPU memory growth in TF2.x style
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    print(f"Found {len(physical_devices)} GPU(s)")
    for device in physical_devices:
        try:
            tf.config.experimental.set_memory_growth(device, True)
            print(f"Memory growth enabled for {device}")
        except Exception as e:
            print(f"Error setting memory growth: {e}")
else:
    print("No GPU devices found, using CPU")

# Create default TF1.x graph and session
graph = tf.compat.v1.Graph()
with graph.as_default():
    sess = tf.compat.v1.Session(graph=graph, config=tf.compat.v1.ConfigProto(
        allow_soft_placement=True,
        gpu_options=tf.compat.v1.GPUOptions(allow_growth=True)
    ))
    tf.compat.v1.global_variables_initializer().run(session=sess)

# Try mixed precision
try:
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)
    print(f"Mixed precision policy: {policy.name}")
except:
    print("Mixed precision not available, using default precision")

# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.compat.v1.set_random_seed(seed)  # TF 1.x style
    os.environ['PYTHONHASHSEED'] = str(seed)

# Set seed
set_seed(42)

# Ensure output directories exist
os.makedirs('m4_optimized_models', exist_ok=True)

# Simple test to verify TensorFlow is working with the GPU
print("Running simple TensorFlow operation to test GPU...")
with graph.as_default():
    a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    b = tf.constant([[5.0, 6.0], [7.0, 8.0]])
    c = tf.matmul(a, b)
    result = sess.run(c)
    print(f"Matrix multiplication result: {result}")
    device = sess.run(tf.compat.v1.report_uninitialized_variables())
    print(f"Using device: {tf.compat.v1.Session().list_devices()}")

# =====================================================================================
# SIMPLIFIED MODEL ARCHITECTURE FOR M4 MAX (GRAPH MODE)
# =====================================================================================

class M4NeuralNetwork:
    def __init__(self, cluster_system, target_variable='IncomeInvestment',
                 distance_metric='Cosine', k=3, n_trials=20):
        """
        Neural Network optimizer specifically designed for Apple M4 Max GPUs
        using TensorFlow graph mode
        """
        self.cluster_system = cluster_system
        self.target_variable = target_variable
        self.distance_metric = distance_metric
        self.k = k
        self.n_trials = n_trials
        self.models = {}
        self.best_params = {}
        self.feature_importance = {}
        self.metrics = {}
        self.scalers = {}
        
        # Get data with clusters
        self.data_with_clusters = cluster_system.get_data_with_clusters(
            distance_metric, k, include_targets=True)
        
        # Define metric names
        self.metric_names = ['accuracy', 'precision', 'recall', 'f1', 'auc']
        
        print(f"Initialized M4-optimized Neural Network (Graph Mode) for {target_variable}")
        print(f"Using {distance_metric} clustering with k={k}")
        print(f"Will perform {n_trials} optimization trials")
    
    def _preprocess_data(self, X, y, cluster_id):
        """
        Preprocess data with multiple scaling options
        """
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Try different scalers and select best
        scalers = {
            'standard': StandardScaler(),
            'minmax': MinMaxScaler(),
            'robust': RobustScaler()
        }
        
        best_skew = float('inf')
        best_scaler_name = 'standard'
        best_X_train_scaled = None
        best_X_test_scaled = None
        
        for name, scaler in scalers.items():
            # Apply scaling
            X_train_scaled = scaler.fit_transform(X_train)
            
            # Calculate skewness for each feature and take mean
            mean_skew = np.mean([abs(np.mean(np.power(col - np.mean(col), 3))) for col in X_train_scaled.T])
            
            # Keep scaler with lowest skewness
            if mean_skew < best_skew:
                best_skew = mean_skew
                best_scaler_name = name
                best_X_train_scaled = X_train_scaled
                best_X_test_scaled = scaler.transform(X_test)
                best_scaler = scaler
        
        print(f"Selected {best_scaler_name} scaler for Cluster {cluster_id} (skew: {best_skew:.4f})")
        
        # Store scaler for later use
        self.scalers[cluster_id] = best_scaler
        
        return best_X_train_scaled, best_X_test_scaled, y_train, y_test
    
    def _build_m4_model(self, input_dim, params):
        """
        Build a M4 Max compatible model with graph mode
        """
        # Extract parameters
        hidden_layers = params['hidden_layers']
        dropout_rates = params['dropout_rates']
        l2_reg = params['l2_reg']
        learning_rate = params['learning_rate']
        
        # Reset graph for clean model building
        tf.compat.v1.keras.backend.clear_session()
        
        # Create a sequential model
        model = tf.keras.Sequential()
        
        # Input layer with noise option
        if params.get('use_noise', False):
            model.add(tf.keras.layers.GaussianNoise(
                params.get('noise_stddev', 0.1),
                input_shape=(input_dim,)
            ))
        
        # First dense layer
        model.add(tf.keras.layers.Dense(
            hidden_layers[0], 
            input_shape=(input_dim,),
            kernel_regularizer=tf.keras.regularizers.l2(l2_reg),
            activation=None
        ))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Activation('relu'))
        model.add(tf.keras.layers.Dropout(dropout_rates[0]))
        
        # Hidden layers
        for i in range(1, len(hidden_layers)):
            model.add(tf.keras.layers.Dense(
                hidden_layers[i],
                kernel_regularizer=tf.keras.regularizers.l2(l2_reg),
                activation=None
            ))
            model.add(tf.keras.layers.BatchNormalization())
            model.add(tf.keras.layers.Activation('relu'))
            model.add(tf.keras.layers.Dropout(dropout_rates[i]))
        
        # Output layer
        model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
        
        # Compile with optimizer
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                tf.keras.metrics.AUC(name='auc'),
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall')
            ]
        )
        
        return model
    
    def _optimize_hyperparams(self, X_train, X_test, y_train, y_test, cluster_id):
        """
        Optimize hyperparameters using Optuna - graph mode compatible
        """
        print(f"\nOptimizing hyperparameters for Cluster {cluster_id}...")
        input_dim = X_train.shape[1]
        
        def objective(trial):
            # Layer sizes with diminishing width
            n_layers = trial.suggest_int('n_layers', 2, 4)  # Reduced complexity for faster trials
            
            # Generate hidden layer sizes
            first_layer = trial.suggest_int('first_layer', 64, 256)
            hidden_layers = [first_layer]
            
            for i in range(1, n_layers):
                # Decrease layer size as we go deeper
                layer_size = trial.suggest_int(f'layer_{i}', 
                                             int(hidden_layers[-1] * 0.5),
                                             hidden_layers[-1])
                hidden_layers.append(layer_size)
            
            # Generate dropout rates
            dropout_rates = [trial.suggest_float(f'dropout_{i}', 0.1, 0.5) 
                            for i in range(n_layers)]
            
            # Other hyperparameters
            params = {
                'hidden_layers': hidden_layers,
                'dropout_rates': dropout_rates,
                'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
                'l2_reg': trial.suggest_float('l2_reg', 1e-6, 1e-3, log=True),
                'use_noise': trial.suggest_categorical('use_noise', [True, False]),
                'noise_stddev': trial.suggest_float('noise_stddev', 0.01, 0.3),
                'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128])
            }
            
            # Class weights for imbalanced data
            neg_class = len(y_train) - np.sum(y_train)
            pos_class = np.sum(y_train)
            
            # Avoid division by zero
            if pos_class == 0: pos_class = 1
            if neg_class == 0: neg_class = 1
                
            class_weight = {0: 1.0, 1: neg_class / pos_class}
            
            # Early stopping
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_auc',
                patience=15,  # Reduced patience for faster trials
                restore_best_weights=True,
                mode='max'
            )
            
            # Create and train model
            model = self._build_m4_model(input_dim, params)
            
            try:
                # Train model
                history = model.fit(
                    X_train, y_train,
                    validation_data=(X_test, y_test),
                    epochs=100,  # Reduced epochs for faster trials
                    batch_size=params['batch_size'],
                    class_weight=class_weight,
                    callbacks=[early_stopping],
                    verbose=0
                )
                
                # Get metrics
                val_auc = max(history.history['val_auc'])
                
                # Final predictions
                y_pred_proba = model.predict(X_test, verbose=0).flatten()
                y_pred = (y_pred_proba >= 0.5).astype(int)
                
                val_precision = precision_score(y_test, y_pred, zero_division=0)
                val_accuracy = accuracy_score(y_test, y_pred)
                
                # Clean up
                del model
                tf.keras.backend.clear_session()
                gc.collect()
                
                # Optimize for multiple metrics
                combined_score = 0.6 * val_auc + 0.25 * val_precision + 0.15 * val_accuracy
                
                return combined_score
            except Exception as e:
                print(f"Error during model training: {e}")
                # Clean up on error
                try:
                    del model
                except:
                    pass
                tf.keras.backend.clear_session()
                gc.collect()
                return 0.0
        
        # Run optimization
        sampler = optuna.samplers.TPESampler(seed=42)
        study = optuna.create_study(direction='maximize', sampler=sampler)
        study.optimize(objective, n_trials=self.n_trials)
        
        # Get best params
        best_trial = study.best_trial
        
        # Construct final params
        n_layers = best_trial.params['n_layers']
        
        # Extract hidden layers
        hidden_layers = [best_trial.params['first_layer']]
        for i in range(1, n_layers):
            hidden_layers.append(best_trial.params[f'layer_{i}'])
            
        # Extract dropout rates
        dropout_rates = [best_trial.params[f'dropout_{i}'] for i in range(n_layers)]
            
        final_params = {
            'hidden_layers': hidden_layers,
            'dropout_rates': dropout_rates,
            'learning_rate': best_trial.params['learning_rate'],
            'l2_reg': best_trial.params['l2_reg'],
            'use_noise': best_trial.params['use_noise'],
            'noise_stddev': best_trial.params['noise_stddev'],
            'batch_size': best_trial.params['batch_size']
        }
        
        print(f"Best hyperparameters for Cluster {cluster_id}:")
        print(f"  Layers: {hidden_layers}")
        print(f"  Learning rate: {final_params['learning_rate']:.6f}")
        print(f"  Batch size: {final_params['batch_size']}")
        
        return final_params
    
    def _train_model(self, X_train, X_test, y_train, y_test, params, cluster_id):
        """
        Train final model with best hyperparameters
        """
        print(f"\nTraining final model for Cluster {cluster_id}...")
        input_dim = X_train.shape[1]
        
        # Clean session for new model
        tf.keras.backend.clear_session()
        
        # Class weights for imbalanced data
        neg_class = len(y_train) - np.sum(y_train)
        pos_class = np.sum(y_train)
        
        # Avoid division by zero
        if pos_class == 0: pos_class = 1
        if neg_class == 0: neg_class = 1
            
        class_weight = {0: 1.0, 1: neg_class / pos_class}
        
        # Callbacks
        # 1. Early stopping
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_auc',
            patience=30,
            restore_best_weights=True,
            mode='max',
            verbose=1
        )
        
        # 2. Learning rate reduction
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-6,
            verbose=1
        )
        
        # 3. Model checkpoint
        model_dir = f"m4_optimized_models/cluster_{cluster_id}_{self.target_variable}"
        os.makedirs(model_dir, exist_ok=True)
        
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            f"{model_dir}/model.h5",
            monitor='val_auc',
            save_best_only=True,
            mode='max',
            verbose=1
        )
        
        # Build model
        try:
            # Build model
            model = self._build_m4_model(input_dim, params)
            
            # Summary
            model.summary()
            
            # Train model
            history = model.fit(
                X_train, y_train,
                validation_data=(X_test, y_test),
                epochs=200,  # Moderate epoch count with early stopping
                batch_size=params['batch_size'],
                class_weight=class_weight,
                callbacks=[early_stopping, reduce_lr, checkpoint],
                verbose=2
            )
            
            # Load best model
            best_model = tf.keras.models.load_model(f"{model_dir}/model.h5")
            
            # Make predictions
            y_pred_proba = best_model.predict(X_test, verbose=0).flatten()
            
            # Threshold optimization for precision/recall balance
            thresholds = np.linspace(0.1, 0.9, 30)  # Fewer threshold points for faster processing
            
            best_threshold = 0.5  # Default
            best_score = 0
            metrics_by_threshold = []
            
            for threshold in thresholds:
                y_pred = (y_pred_proba >= threshold).astype(int)
                
                try:
                    precision = precision_score(y_test, y_pred, zero_division=0)
                    recall = recall_score(y_test, y_pred, zero_division=0)
                    accuracy = accuracy_score(y_test, y_pred)
                    
                    # Simple f1 calculation with safety
                    if precision + recall > 0:
                        f1 = 2 * precision * recall / (precision + recall)
                    else:
                        f1 = 0
                    
                    # Calculate balanced score
                    score = 0.4 * precision + 0.3 * recall + 0.1 * accuracy + 0.2 * f1
                    
                    if score > best_score:
                        best_score = score
                        best_threshold = threshold
                        
                    metrics_by_threshold.append({
                        'threshold': threshold,
                        'precision': precision,
                        'recall': recall,
                        'accuracy': accuracy,
                        'f1': f1,
                        'score': score
                    })
                except Exception as e:
                    print(f"Error with threshold {threshold}: {e}")
                    continue
            
            # Final predictions with optimized threshold
            y_pred = (y_pred_proba >= best_threshold).astype(int)
            
            # Final metrics
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'recall': recall_score(y_test, y_pred, zero_division=0),
                'f1': f1_score(y_test, y_pred, zero_division=0),
                'auc': roc_auc_score(y_test, y_pred_proba),
                'threshold': best_threshold,
                'test_size': len(y_test),
                'positive_rate': y_test.mean()
            }
            
            # Print metrics
            print(f"\nCluster {cluster_id} Model Performance:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value:.4f}")
            
            # Print detailed classification report
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred))
            
            # Store test data for visualization
            test_data = {
                'X_test': X_test,
                'y_test': y_test.values if hasattr(y_test, 'values') else y_test,
                'y_pred_proba': y_pred_proba,
                'y_pred': y_pred,
                'history': history.history
            }
            
            return best_model, metrics, test_data
            
        except Exception as e:
            print(f"Error during model training: {e}")
            # Create empty results in case of errors
            metrics = {metric: 0.0 for metric in self.metric_names}
            metrics.update({
                'threshold': 0.5,
                'test_size': len(y_test),
                'positive_rate': y_test.mean()
            })
            
            # Create dummy test data
            test_data = {
                'X_test': X_test,
                'y_test': y_test.values if hasattr(y_test, 'values') else y_test,
                'y_pred_proba': np.zeros(len(y_test)),
                'y_pred': np.zeros(len(y_test)),
                'history': {'loss': [], 'val_loss': [], 'auc': [], 'val_auc': [], 
                           'precision': [], 'val_precision': []}
            }
            
            # Try to create a simple fallback model
            try:
                simple_model = tf.keras.Sequential([
                    tf.keras.layers.Dense(32, activation='relu', input_shape=(input_dim,)),
                    tf.keras.layers.Dense(1, activation='sigmoid')
                ])
                simple_model.compile(optimizer='adam', loss='binary_crossentropy')
                return simple_model, metrics, test_data
            except:
                return None, metrics, test_data
    
    def _calculate_feature_importance(self, model, X, feature_names, cluster_id):
        """
        Calculate feature importance using permutation method (graph mode compatible)
        """
        print(f"\nCalculating feature importance for Cluster {cluster_id}...")
        
        # Get test data
        y_test = getattr(self, 'y_test', None)
        if y_test is None:
            print("Warning: y_test not available, skipping feature importance")
            return pd.DataFrame({'Feature': feature_names, 'Importance': 0.0})
        
        # For graph mode, use model.predict with batch_size parameter to avoid memory issues
        baseline_predictions = model.predict(X, batch_size=64, verbose=0).flatten()
        baseline_auc = roc_auc_score(y_test, baseline_predictions)
        
        # Calculate importance
        importance = []
        
        for i, feature_name in enumerate(feature_names):
            # Copy the data
            X_permuted = X.copy()
            
            # Shuffle the feature
            X_permuted[:, i] = np.random.permutation(X_permuted[:, i])
            
            # Predict with shuffled feature
            permuted_predictions = model.predict(X_permuted, batch_size=64, verbose=0).flatten()
            
            # Calculate AUC decrease
            permuted_auc = roc_auc_score(y_test, permuted_predictions)
            importance_value = baseline_auc - permuted_auc
            
            # Store feature importance
            importance.append((feature_name, importance_value))
            
            # Print progress for long-running operations
            if (i+1) % 10 == 0:
                print(f"  Processed {i+1}/{len(feature_names)} features")
        
        # Sort by importance
        importance.sort(key=lambda x: x[1], reverse=True)
        
        # Convert to DataFrame
        importance_df = pd.DataFrame(importance, columns=['Feature', 'Importance'])
        
        return importance_df
    
    def _plot_results(self, history, X_test, y_test, y_pred_proba, y_pred, 
                     importance_df, metrics, cluster_id):
        """
        Visualize model results
        """
        # Create directory
        cluster_dir = f"m4_optimized_models/cluster_{cluster_id}_{self.target_variable}"
        os.makedirs(cluster_dir, exist_ok=True)
        
        # 1. Training history
        plt.figure(figsize=(18, 6))
        
        # AUC
        plt.subplot(1, 3, 1)
        plt.plot(history['auc'], label='Train AUC')
        plt.plot(history['val_auc'], label='Validation AUC')
        plt.title('Model AUC')
        plt.xlabel('Epoch')
        plt.ylabel('AUC')
        plt.legend()
        
        # Loss
        plt.subplot(1, 3, 2)
        plt.plot(history['loss'], label='Train Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Precision
        plt.subplot(1, 3, 3)
        plt.plot(history['precision'], label='Train Precision')
        plt.plot(history['val_precision'], label='Validation Precision')
        plt.title('Model Precision')
        plt.xlabel('Epoch')
        plt.ylabel('Precision')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f"{cluster_dir}/training_history.png", dpi=300)
        plt.close()
        
        # 2. Performance visualizations
        plt.figure(figsize=(16, 14))
        
        # ROC Curve
        plt.subplot(2, 2, 1)
        from sklearn import metrics as sklearn_metrics
        fpr, tpr, _ = sklearn_metrics.roc_curve(y_test, y_pred_proba)
        plt.plot(fpr, tpr, label=f'AUC = {metrics["auc"]:.4f}')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - Cluster {cluster_id}')
        plt.legend()
        
        # Precision-Recall Curve
        plt.subplot(2, 2, 2)
        precision, recall, _ = sklearn_metrics.precision_recall_curve(y_test, y_pred_proba)
        plt.plot(recall, precision)
        plt.scatter([metrics['recall']], [metrics['precision']], 
                   c='red', label=f'Threshold = {metrics["threshold"]:.2f}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve - Cluster {cluster_id}')
        plt.legend()
        
        # Confusion Matrix
        plt.subplot(2, 2, 3)
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'Confusion Matrix - Cluster {cluster_id}')
        
        # Metrics Summary
        plt.subplot(2, 2, 4)
        metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1', 'auc']
        values = [metrics[m] for m in metrics_to_plot]
        colors = sns.color_palette("viridis", len(metrics_to_plot))
        bars = plt.bar(metrics_to_plot, values, color=colors)
        plt.ylim(0, 1)
        plt.title(f'Performance Metrics - Cluster {cluster_id}')
        
        # Add values on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(f"{cluster_dir}/performance_summary.png", dpi=300)
        plt.close()
        
        # 3. Feature Importance
        if importance_df is not None and len(importance_df) > 0:
            plt.figure(figsize=(12, max(6, min(30, len(importance_df)) * 0.3)))
            top_n = min(30, len(importance_df))
            sns.barplot(x='Importance', y='Feature', 
                       data=importance_df.head(top_n))
            plt.title(f'Feature Importance - Cluster {cluster_id}')
            plt.tight_layout()
            plt.savefig(f"{cluster_dir}/feature_importance.png", dpi=300)
            plt.close()
    
    def _save_model(self, model, params, metrics, importance_df, feature_names, cluster_id):
        """
        Save model and metadata
        """
        # Create directory
        model_dir = f"m4_optimized_models/cluster_{cluster_id}_{self.target_variable}"
        os.makedirs(model_dir, exist_ok=True)
        
        # Save model
        model.save(f"{model_dir}/model.h5")
        
        # Save scaler
        joblib.dump(self.scalers[cluster_id], f"{model_dir}/scaler.joblib")
        
        # Save hyperparameters
        with open(f"{model_dir}/hyperparameters.json", 'w') as f:
            json.dump(params, f, indent=4)
        
        # Save metrics
        pd.Series(metrics).to_csv(f"{model_dir}/metrics.csv")
        
        # Save feature importance
        if importance_df is not None:
            importance_df.to_csv(f"{model_dir}/feature_importance.csv", index=False)
        
        # Save feature names
        pd.Series(feature_names).to_csv(f"{model_dir}/feature_names.csv", index=False)
        
        print(f"Model and metadata saved to {model_dir}")
    
    def _summarize_results(self):
        """
        Summarize results across all clusters
        """
        print("\nSUMMARY OF RESULTS ACROSS CLUSTERS")
        print("="*80)
        
        # Collect metrics
        all_metrics = {}
        cluster_sizes = {}
        
        for cluster_id, metrics in self.metrics.items():
            all_metrics[cluster_id] = metrics
            cluster_sizes[cluster_id] = metrics['test_size']
        
        if not all_metrics:
            print("No metrics available to summarize.")
            return None
        
        # Create summary DataFrame
        summary = pd.DataFrame(all_metrics).T
        
        # Add cluster sizes
        summary['cluster_size'] = pd.Series(cluster_sizes)
        
        # Calculate weighted average metrics
        total_size = summary['cluster_size'].sum()
        weighted_metrics = {}
        
        for metric in self.metric_names:
            if metric in summary.columns:
                weighted_metrics[metric] = np.sum(summary[metric] * summary['cluster_size']) / total_size
        
        # Print summary
        print("\nMetrics by Cluster:")
        metric_cols = [col for col in self.metric_names if col in summary.columns]
        print(summary[metric_cols + ['threshold', 'cluster_size', 'positive_rate']].round(4))
        
        print("\nWeighted Average Metrics (by cluster size):")
        for metric, value in weighted_metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        # Create summary visualization
        plt.figure(figsize=(12, 10))
        
        # Plot metrics by cluster
        metrics_df = summary[metric_cols].copy()
        metrics_df['Cluster'] = metrics_df.index
        metrics_df = metrics_df.melt(id_vars=['Cluster'], var_name='Metric', value_name='Value')
        
        # Create grouped bar chart
        chart = sns.catplot(x='Metric', y='Value', hue='Cluster', data=metrics_df, kind='bar', height=6, aspect=1.5)
        chart.set_xticklabels(rotation=45)
        plt.title(f'{self.target_variable} - Performance Metrics Across Clusters')
        plt.tight_layout()
        plt.savefig(f"m4_optimized_models/{self.target_variable}_metrics_by_cluster.png", dpi=300)
        plt.close()
        
        return summary
        
    def train_all_clusters(self):
        """
        Train models for all clusters
        """
        print(f"\n{'='*80}")
        print(f"TRAINING {self.target_variable.upper()} M4-OPTIMIZED MODELS FOR ALL CLUSTERS")
        print(f"{'='*80}")
        
        start_time = time.time()
        
        # Get number of clusters
        num_clusters = self.k
        
        # Train for each cluster
        for cluster_id in range(num_clusters):
            cluster_start_time = time.time()
            print(f"\n{'-'*80}")
            print(f"PROCESSING CLUSTER {cluster_id}")
            print(f"{'-'*80}")
            
            # Get data
            X, y_income, y_accum = self.cluster_system.get_training_data_by_cluster(
                self.distance_metric, self.k, cluster_id)
            
            # Select target
            y = y_income if self.target_variable == 'IncomeInvestment' else y_accum
            
            # Skip empty or single-class clusters
            if len(X) < 20 or len(np.unique(y)) < 2:
                print(f"Skipping Cluster {cluster_id}: insufficient data ({len(X)} samples) or only one class")
                continue
                
            # Print info
            print(f"Cluster {cluster_id} size: {len(X)} samples")
            print(f"Positive rate: {y.mean():.2%}")
            
            # 1. Preprocess
            X_train_scaled, X_test_scaled, y_train, y_test = self._preprocess_data(X, y, cluster_id)
            
            # Save for feature importance
            self.y_test = y_test
            
            # 2. Optimize hyperparameters
            best_params = self._optimize_hyperparams(X_train_scaled, X_test_scaled, y_train, y_test, cluster_id)
            self.best_params[cluster_id] = best_params
            
            # 3. Train with best parameters
            model, metrics, test_data = self._train_model(
                X_train_scaled, X_test_scaled, y_train, y_test, best_params, cluster_id)
            self.models[cluster_id] = model
            self.metrics[cluster_id] = metrics
            
            # 4. Calculate feature importance
            feature_names = X.columns
            importance_df = self._calculate_feature_importance(model, X_test_scaled, feature_names, cluster_id)
            self.feature_importance[cluster_id] = importance_df
            
            # 5. Plot results
            self._plot_results(
                test_data['history'],
                test_data['X_test'], 
                test_data['y_test'], 
                test_data['y_pred_proba'], 
                test_data['y_pred'],
                importance_df,
                metrics,
                cluster_id
            )
            
            # 6. Save model
            self._save_model(
                model, best_params, metrics, importance_df, 
                feature_names, cluster_id
            )
            
            # Report time
            cluster_time = time.time() - cluster_start_time
            print(f"\nCluster {cluster_id} completed in {cluster_time:.2f} seconds ({cluster_time/60:.2f} minutes)")
            
            # Clear session between clusters
            tf.keras.backend.clear_session()
            gc.collect()
        
        # Summarize results
        self._summarize_results()
        
        # Total time
        total_time = time.time() - start_time
        print(f"\nTotal training completed in {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
        
        return self.models, self.metrics, self.feature_importance

# For use in Jupyter notebooks
def prepare_for_notebook():
    """Prepare TensorFlow environment for running in notebooks"""
    import tensorflow as tf
    
    # Use graph mode
    tf.compat.v1.disable_eager_execution()
    
    # Configure GPU memory growth
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        print(f"Found {len(physical_devices)} GPU(s)")
        for device in physical_devices:
            try:
                tf.config.experimental.set_memory_growth(device, True)
                print(f"Memory growth enabled for {device}")
            except Exception as e:
                print(f"Error setting memory growth: {e}")
    else:
        print("No GPU devices found, using CPU")
    
    # Create session
    graph = tf.compat.v1.Graph()
    with graph.as_default():
        sess = tf.compat.v1.Session(graph=graph, config=tf.compat.v1.ConfigProto(
            allow_soft_placement=True,
            gpu_options=tf.compat.v1.GPUOptions(allow_growth=True)
        ))
        tf.compat.v1.global_variables_initializer().run(session=sess)
    
    print("TensorFlow configured for graph mode execution")
    return sess, graph

# Run for both target variables if executed as script
if __name__ == "__main__":
    targets = ['IncomeInvestment', 'AccumulationInvestment']
    model_results = {}

    for target in targets:
        print(f"\n{'#'*100}")
        print(f"# TRAINING M4-OPTIMIZED NEURAL NETWORK FOR {target.upper()}")
        print(f"{'#'*100}")
        
        # Create optimizer
        nn_optimizer = M4NeuralNetwork(
            cluster_system=cluster_system,
            target_variable=target,
            distance_metric='Cosine',
            k=3,
            n_trials=20  # Reduced for faster execution
        )
        
        # Train models
        nn_models, nn_metrics, nn_importance = nn_optimizer.train_all_clusters()
        
        # Store results
        model_results[target] = {
            'optimizer': nn_optimizer,
            'models': nn_models,
            'metrics': nn_metrics,
            'importance': nn_importance
        }

    print("\nAll models have been trained and saved!") 