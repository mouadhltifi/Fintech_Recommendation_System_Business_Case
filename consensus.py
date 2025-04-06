# =====================================================================================
# CONSENSUS MODEL: COMBINING NEURAL NETWORK AND RANDOM FOREST MODELS
# =====================================================================================
import numpy as np
import pandas as pd
import torch
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score, precision_score, accuracy_score, confusion_matrix,
    classification_report, recall_score, f1_score
)
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, ClassifierMixin
import warnings
warnings.filterwarnings('ignore')

# Create directory for model outputs
os.makedirs('consensus_model', exist_ok=True)

class NeuralNetwork(torch.nn.Module):
    """Neural network model architecture matching the one from training"""
    def __init__(self, input_dim, hidden_dims, dropout_rates, activation_fn='relu'):
        super(NeuralNetwork, self).__init__()
        
        # Initialize activation function
        if activation_fn == 'relu':
            self.activation = torch.nn.ReLU()
        elif activation_fn == 'leaky_relu':
            self.activation = torch.nn.LeakyReLU(0.1)
        elif activation_fn == 'elu':
            self.activation = torch.nn.ELU()
        elif activation_fn == 'selu':
            self.activation = torch.nn.SELU()
        else:
            self.activation = torch.nn.ReLU()
        
        # Build layers
        layers = []
        
        # Input layer to first hidden layer
        layers.append(torch.nn.Linear(input_dim, hidden_dims[0]))
        layers.append(self.activation)
        layers.append(torch.nn.Dropout(dropout_rates[0]))
        
        # Hidden layers
        for i in range(len(hidden_dims) - 1):
            layers.append(torch.nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            layers.append(self.activation)
            layers.append(torch.nn.Dropout(dropout_rates[i + 1]))
        
        # Output layer
        layers.append(torch.nn.Linear(hidden_dims[-1], 1))
        layers.append(torch.nn.Sigmoid())
        
        # Create sequential model
        self.model = torch.nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)


class ModelWrapper(BaseEstimator, ClassifierMixin):
    """Wrapper for different model types to provide a consistent interface"""
    def __init__(self, model_type, model, scaler=None, threshold=0.5, feature_names=None, model_path=None):
        self.model_type = model_type
        self.model = model
        self.scaler = scaler
        self.threshold = threshold
        self.feature_names = feature_names  # Store expected feature names
        self.model_path = model_path        # Store path to model for reference
        
        # Set device for neural network
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    def fit(self, X, y):
        # This is just a wrapper, so no actual fitting happens here
        return self
    
    def prepare_data(self, X):
        """Prepare data for prediction"""
        # Make a copy to avoid modifying the original
        X_prepared = X.copy()
        
        # Add missing features
        if self.feature_names is not None:
            missing_features = [f for f in self.feature_names if f not in X_prepared.columns]
            for feature in missing_features:
                X_prepared[feature] = 0
                
            # Remove extra features and reorder
            try:
                X_prepared = X_prepared[self.feature_names]
            except KeyError:
                # Some expected features are still missing
                for feature in self.feature_names:
                    if feature not in X_prepared.columns:
                        X_prepared[feature] = 0
                X_prepared = X_prepared[self.feature_names]
        
        return X_prepared
    
    def predict_proba(self, X):
        try:
            # Different prediction methods based on model type
            if self.model_type == 'neural_network':
                # Prepare data for neural network
                X_prepared = self.prepare_data(X)
                
                # Convert to numpy array if it's a DataFrame
                if isinstance(X_prepared, pd.DataFrame):
                    X_prepared = X_prepared.values
                    
                # Apply scaler if available
                if self.scaler is not None:
                    try:
                        X_prepared = self.scaler.transform(X_prepared)
                    except Exception as e:
                        print(f"Scaling failed, using unscaled data: {e}")
                
                # Make prediction
                X_tensor = torch.FloatTensor(X_prepared).to(self.device)
                self.model.eval()
                with torch.no_grad():
                    y_proba = self.model(X_tensor).cpu().numpy()
                # Reshape to match sklearn format [prob_neg, prob_pos]
                return np.hstack([1-y_proba, y_proba])
                
            elif self.model_type == 'random_forest':
                # Prepare data for random forest
                X_prepared = self.prepare_data(X)
                
                # Convert to numpy array if it's a DataFrame
                if isinstance(X_prepared, pd.DataFrame):
                    X_prepared = X_prepared.values
                    
                # Apply scaler if available
                if self.scaler is not None:
                    try:
                        X_prepared = self.scaler.transform(X_prepared)
                    except Exception as e:
                        print(f"Scaling failed, using unscaled data: {e}")
                
                # Make prediction
                return self.model.predict_proba(X_prepared)
            else:
                raise ValueError(f"Unknown model type: {self.model_type}")
                
        except Exception as e:
            print(f"Error predicting with {self.model_type} model: {e}")
            # Return a default prediction (0.5 probability)
            return np.full((X.shape[0], 2), 0.5)
    
    def predict(self, X):
        probas = self.predict_proba(X)
        return (probas[:, 1] >= self.threshold).astype(int)


class ConsensusModel:
    def __init__(self, target_variable, cluster_system=None):
        """
        Initialize the consensus model.
        
        Parameters:
        -----------
        target_variable : str
            Target variable to predict ('IncomeInvestment' or 'AccumulationInvestment')
        cluster_system : object, optional
            Cluster system to use for getting cluster assignments
        """
        self.target_variable = target_variable
        self.cluster_system = cluster_system
        
        # Dictionary to store wrapped models
        self.model_wrappers = []
        
        # Meta-model for learning weights
        self.meta_model = None
        
        # Directory paths for models
        self.model_dirs = {
            'neural_network': 'direct_nn_models',
            'random_forest': 'cluster_rf_models'
        }
        
        # For cluster-based models
        self.num_clusters = 3  # Default
        self.distance_metric = 'Cosine'  # Default
        
        # For evaluation
        self.metrics = {}
        self.feature_names = None
    
    def load_models(self):
        """Load all available trained models"""
        print(f"Loading models for {self.target_variable} prediction...")
        
        # Load direct neural network model
        nn_dir = f"{self.model_dirs['neural_network']}/{self.target_variable}"
        if os.path.exists(nn_dir):
            print(f"Loading Neural Network model from {nn_dir}")
            try:
                # Load model info and scaler
                model_info = joblib.load(f"{nn_dir}/model_info.pkl")
                scaler = joblib.load(f"{nn_dir}/scaler.pkl")
                
                # Get the feature names from the scaler if available
                feature_names = None
                if hasattr(scaler, 'feature_names_in_'):
                    feature_names = scaler.feature_names_in_
                
                # Create model with the same architecture
                model = NeuralNetwork(
                    input_dim=model_info['input_dim'],
                    hidden_dims=model_info['hidden_dims'],
                    dropout_rates=model_info['dropout_rates'],
                    activation_fn=model_info['activation_fn']
                )
                
                # Set device
                device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
                model.to(device)
                
                # Load weights
                model.load_state_dict(torch.load(f"{nn_dir}/model.pt", map_location=device))
                model.eval()
                
                # Get metrics to extract threshold
                metrics = pd.read_csv(f"{nn_dir}/metrics.csv", index_col=0).squeeze()
                threshold = metrics.get('threshold', 0.5)
                
                # Wrap the model
                self.model_wrappers.append(
                    ModelWrapper('neural_network', model, scaler, threshold, feature_names)
                )
                
                print(f"Neural Network model loaded successfully")
            except Exception as e:
                print(f"Error loading Neural Network model: {e}")
        
        # Load cluster-specific Random Forest models
        base_dir = self.model_dirs['random_forest']
        
        # Check for each cluster
        for cluster_id in range(self.num_clusters):
            cluster_dir = f"{base_dir}/cluster_{cluster_id}_{self.target_variable}"
            
            if os.path.exists(cluster_dir):
                print(f"Loading random_forest model for Cluster {cluster_id}")
                try:
                    # Load model, scaler, and metrics
                    model = joblib.load(f"{cluster_dir}/model.pkl")
                    scaler = joblib.load(f"{cluster_dir}/scaler.pkl")
                    
                    # Get feature names
                    feature_names = None
                    if hasattr(scaler, 'feature_names_in_'):
                        feature_names = scaler.feature_names_in_
                    elif hasattr(model, 'feature_names_in_'):
                        feature_names = model.feature_names_in_
                        
                    # Get metrics to extract threshold
                    metrics = pd.read_csv(f"{cluster_dir}/metrics.csv", index_col=0).squeeze()
                    threshold = metrics.get('threshold', 0.5)
                    
                    # Wrap the model
                    self.model_wrappers.append(
                        ModelWrapper('random_forest', model, scaler, threshold, feature_names)
                    )
                        
                    print(f"Random_forest model for Cluster {cluster_id} loaded successfully")
                except Exception as e:
                    print(f"Error loading random_forest model for Cluster {cluster_id}: {e}")
        
        print(f"Total models loaded: {len(self.model_wrappers)}")
        return self
    
    def get_cluster_assignments(self, X):
        """
        Get cluster assignments for data points
        
        Parameters:
        -----------
        X : DataFrame
            Features
            
        Returns:
        --------
        np.array
            Cluster IDs for each data point
        """
        if self.cluster_system is None:
            raise ValueError("Cluster system not provided")
        
        return self.cluster_system.assign_clusters(X, self.distance_metric, self.num_clusters)
    
    def get_base_predictions(self, X):
        """
        Get predictions from all base models
        
        Parameters:
        -----------
        X : DataFrame
            Features
            
        Returns:
        --------
        np.array
            Base model predictions (n_samples, n_models)
        """
        if not self.model_wrappers:
            raise ValueError("No models loaded. Call load_models() first.")
        
        # Store feature names for later use if not already stored
        if self.feature_names is None and isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
        
        # Get predictions from each model
        all_preds = np.zeros((X.shape[0], len(self.model_wrappers)))
        valid_models = []
        
        for i, model_wrapper in enumerate(self.model_wrappers):
            try:
                # Get probability predictions
                preds = model_wrapper.predict_proba(X)[:, 1]
                all_preds[:, i] = preds
                valid_models.append(i)
            except Exception as e:
                print(f"Error getting predictions from model {i}: {e}")
                # Fill with default predictions (0.5)
                all_preds[:, i] = 0.5
        
        # If some models failed, only return predictions from valid models
        if len(valid_models) < len(self.model_wrappers):
            print(f"Using {len(valid_models)} out of {len(self.model_wrappers)} models")
            all_preds = all_preds[:, valid_models]
            self.model_wrappers = [self.model_wrappers[i] for i in valid_models]
        
        return all_preds
    
    def train_meta_model(self, X, y, cv=5, meta_model_type='advanced', max_iterations=1000, patience=50):
        """
        Train a meta-model to find optimal weights for the ensemble
        
        Parameters:
        -----------
        X : DataFrame
            Features
        y : Series
            Target variable
        cv : int
            Number of cross-validation folds
        meta_model_type : str
            Type of meta-model ('advanced', 'logistic', 'voting', 'ensemble')
        max_iterations : int
            Maximum number of training iterations
        patience : int
            Early stopping patience - how many iterations without improvement before stopping
            
        Returns:
        --------
        self
        """
        print(f"Training meta-model to find optimal weights (max_iterations={max_iterations}, patience={patience})...")
        
        # Store training data statistics
        self.metrics['training_size'] = len(X)
        self.metrics['positive_rate'] = y.mean()
        
        # Split data for final evaluation
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Choose meta-model type
        if meta_model_type == 'advanced':
            # Use a more powerful Gradient Boosting model with early stopping
            print("Using advanced GradientBoostingClassifier with early stopping")
            
            # Use stratified cross-validation to get out-of-fold predictions
            # This ensures class balance in each fold
            skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
            
            # Get total number of models
            n_models = len(self.model_wrappers)
            
            # Store out-of-fold predictions
            oof_preds = np.zeros((X_train.shape[0], n_models))
            
            # For each fold
            for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
                print(f"Processing fold {fold+1}/{cv}")
                
                # Split data
                X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
                
                # Get predictions from each base model for this fold
                try:
                    fold_preds = np.zeros((len(val_idx), n_models))
                    for i, model_wrapper in enumerate(self.model_wrappers):
                        # For this fold, we predict on the validation set
                        fold_preds[:, i] = model_wrapper.predict_proba(X_fold_val)[:, 1]
                    
                    # Store the predictions
                    oof_preds[val_idx] = fold_preds
                except Exception as e:
                    print(f"Error in fold {fold+1}: {e}")
                    # Continue with the next fold
                    continue
            
            # Set up validation data from one fold for early stopping
            # We'll use 20% of the data for validation
            X_meta_train, X_meta_val, y_meta_train, y_meta_val = train_test_split(
                oof_preds, y_train, test_size=0.2, random_state=42, stratify=y_train
            )
            
            # Train the meta-model with early stopping
            self.meta_model = GradientBoostingClassifier(
                n_estimators=max_iterations,  # Maximum number of boosting stages
                learning_rate=0.05,          # Slower learning rate for better convergence
                max_depth=5,                 # Deeper trees for more complex interactions
                min_samples_split=20,        # Minimum samples to split an internal node
                min_samples_leaf=10,         # Minimum samples in a leaf
                subsample=0.8,               # Use 80% of samples for each tree (reduces overfitting)
                max_features='sqrt',         # Use sqrt(n_features) for each tree
                random_state=42,
                verbose=0,
                n_iter_no_change=patience,   # Early stopping patience
                validation_fraction=0.2,     # Fraction of training data to use for early stopping
                tol=1e-4                     # Tolerance for early stopping
            )
            
            # Train meta-model on out-of-fold predictions
            self.meta_model.fit(oof_preds, y_train)
            
            # Get actual iterations used (due to early stopping)
            actual_iters = self.meta_model.n_estimators_
            print(f"Meta-model training completed after {actual_iters} iterations (max was {max_iterations})")
            
            # Calculate feature importances (model weights)
            importances = self.meta_model.feature_importances_
            total_importance = np.sum(importances)
            normalized_weights = importances / total_importance if total_importance > 0 else importances
            
            # Print model weights
            print("Meta-model normalized weights:")
            for i, weight in enumerate(normalized_weights):
                print(f"  Model {i+1}: {weight:.4f}")
        
        elif meta_model_type == 'ensemble':
            # Use an ensemble of meta-models
            print("Using ensemble of meta-models")
            
            # Use cross-validation to get out-of-fold predictions
            kf = KFold(n_splits=cv, shuffle=True, random_state=42)
            
            # Get total number of models
            n_models = len(self.model_wrappers)
            
            # Store out-of-fold predictions
            oof_preds = np.zeros((X_train.shape[0], n_models))
            
            # For each fold
            for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
                print(f"Processing fold {fold+1}/{cv}")
                
                # Split data
                X_fold_val = X_train.iloc[val_idx]
                
                # Get predictions from each base model for this fold
                try:
                    fold_preds = np.zeros((len(val_idx), n_models))
                    for i, model_wrapper in enumerate(self.model_wrappers):
                        # For this fold, we predict on the validation set
                        fold_preds[:, i] = model_wrapper.predict_proba(X_fold_val)[:, 1]
                    
                    # Store the predictions
                    oof_preds[val_idx] = fold_preds
                except Exception as e:
                    print(f"Error in fold {fold+1}: {e}")
                    # Continue with the next fold
                    continue
            
            # Create ensemble meta-model
            meta_estimators = [
                ('lr', LogisticRegression(C=1.0, max_iter=max_iterations, class_weight='balanced')),
                ('gb', GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)),
                ('rf', RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, class_weight='balanced'))
            ]
            
            self.meta_model = VotingClassifier(
                estimators=meta_estimators,
                voting='soft',
                weights=[1, 2, 1]  # Give more weight to gradient boosting
            )
            
            # Train meta-model on out-of-fold predictions
            self.meta_model.fit(oof_preds, y_train)
            
            # Print model info
            print("Meta-model ensemble created with following models:")
            for name, _ in meta_estimators:
                print(f"  - {name}")
        
        elif meta_model_type == 'logistic':
            # Original implementation with logistic regression
            print("Using logistic regression meta-model")
            
            # Use cross-validation to get out-of-fold predictions
            kf = KFold(n_splits=cv, shuffle=True, random_state=42)
            
            # Get total number of models
            n_models = len(self.model_wrappers)
            
            # Store out-of-fold predictions
            oof_preds = np.zeros((X_train.shape[0], n_models))
            
            # For each fold
            for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
                print(f"Processing fold {fold+1}/{cv}")
                
                # Split data
                X_fold_val = X_train.iloc[val_idx]
                
                # Get predictions from each base model for this fold
                try:
                    fold_preds = np.zeros((len(val_idx), n_models))
                    for i, model_wrapper in enumerate(self.model_wrappers):
                        # For this fold, we predict on the validation set
                        fold_preds[:, i] = model_wrapper.predict_proba(X_fold_val)[:, 1]
                    
                    # Store the predictions
                    oof_preds[val_idx] = fold_preds
                except Exception as e:
                    print(f"Error in fold {fold+1}: {e}")
                    # Continue with the next fold
                    continue
            
            # Train meta-model on out-of-fold predictions with increased max_iter
            self.meta_model = LogisticRegression(C=1.0, class_weight='balanced', max_iter=max_iterations)
            self.meta_model.fit(oof_preds, y_train)
            
            # Print model weights
            print("Meta-model weights:")
            for i, coef in enumerate(self.meta_model.coef_[0]):
                print(f"  Model {i+1}: {coef:.4f}")
            
        elif meta_model_type == 'voting':
            # Simple equal weights averaging approach
            print("Using equal weights for all models")
            
            # Get base predictions from all models
            base_preds = self.get_base_predictions(X_train)
            
            # Create a simple meta-model that averages predictions
            class AveragingModel:
                def predict_proba(self, X):
                    # X is already the base predictions
                    avg_proba = np.mean(X, axis=1)
                    return np.vstack([1 - avg_proba, avg_proba]).T
            
            self.meta_model = AveragingModel()
        
        # Evaluate on test set
        self.evaluate(X_test, y_test)
        
        return self
    
    def predict_proba(self, X):
        """
        Make probability predictions using the consensus model
        
        Parameters:
        -----------
        X : DataFrame
            Features
            
        Returns:
        --------
        np.array
            Probability predictions
        """
        if self.meta_model is None:
            raise ValueError("Meta-model not trained. Call train_meta_model() first.")
        
        # Get base model predictions
        base_preds = self.get_base_predictions(X)
        
        # Apply meta-model
        if isinstance(self.meta_model, LogisticRegression):
            return self.meta_model.predict_proba(base_preds)
        elif hasattr(self.meta_model, 'predict_proba'):
            # For custom averaging model or any other model with predict_proba
            return self.meta_model.predict_proba(base_preds)
        else:
            # Fallback to simple averaging
            avg_proba = np.mean(base_preds, axis=1)
            return np.vstack([1 - avg_proba, avg_proba]).T
    
    def predict(self, X, threshold=0.5):
        """
        Make predictions using the consensus model
        
        Parameters:
        -----------
        X : DataFrame
            Features
        threshold : float
            Probability threshold for binary classification
            
        Returns:
        --------
        np.array
            Binary predictions
        """
        # Get probability predictions
        probas = self.predict_proba(X)[:, 1]
        
        # Apply threshold
        return (probas >= threshold).astype(int)
    
    def evaluate(self, X, y, threshold=0.5):
        """
        Evaluate the consensus model
        
        Parameters:
        -----------
        X : DataFrame
            Features
        y : Series
            Target variable
        threshold : float
            Probability threshold for binary classification
            
        Returns:
        --------
        dict
            Performance metrics
        """
        # Get predictions
        y_pred_proba = self.predict_proba(X)[:, 1]
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, zero_division=0),
            'recall': recall_score(y, y_pred, zero_division=0),
            'f1': f1_score(y, y_pred, zero_division=0),
            'auc': roc_auc_score(y, y_pred_proba),
            'threshold': threshold,
            'test_size': len(y),
            'positive_rate': y.mean()
        }
        
        # Store metrics
        self.metrics.update(metrics)
        
        # Print metrics
        print(f"\nConsensus Model Performance:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        # Print classification report
        print("\nClassification Report:")
        print(classification_report(y, y_pred))
        
        # Get individual model predictions for comparison
        base_model_metrics = {}
        for i, model_wrapper in enumerate(self.model_wrappers):
            try:
                y_pred_proba_base = model_wrapper.predict_proba(X)[:, 1]
                y_pred_base = model_wrapper.predict(X)
                
                base_model_metrics[f"Model {i+1}"] = {
                    'accuracy': accuracy_score(y, y_pred_base),
                    'precision': precision_score(y, y_pred_base, zero_division=0),
                    'recall': recall_score(y, y_pred_base, zero_division=0),
                    'f1': f1_score(y, y_pred_base, zero_division=0),
                    'auc': roc_auc_score(y, y_pred_proba_base)
                }
            except Exception as e:
                print(f"Error evaluating model {i+1}: {e}")
                # Skip this model
                continue
        
        # Store base model metrics
        self.metrics['base_models'] = base_model_metrics
        
        # Print comparison
        if base_model_metrics:
            print("\nConsensus model vs. Base models:")
            comparison_df = pd.DataFrame({k: v for k, v in base_model_metrics.items()})
            comparison_df['Consensus'] = pd.Series({
                'accuracy': metrics['accuracy'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1': metrics['f1'],
                'auc': metrics['auc']
            })
            print(comparison_df.round(4))
        
        return metrics
    
    def plot_results(self, X, y, threshold=0.5):
        """
        Plot performance metrics and model contributions
        
        Parameters:
        -----------
        X : DataFrame
            Features
        y : Series
            Target variable
        threshold : float
            Probability threshold for binary classification
            
        Returns:
        --------
        None
        """
        # Create output directory
        os.makedirs('consensus_model', exist_ok=True)
        output_dir = f"consensus_model/{self.target_variable}"
        os.makedirs(output_dir, exist_ok=True)
        
        # Get predictions
        y_pred_proba = self.predict_proba(X)[:, 1]
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        # Get base model predictions
        base_preds = self.get_base_predictions(X)
        
        # Create figure
        plt.figure(figsize=(20, 15))
        
        # 1. ROC Curve
        plt.subplot(2, 3, 1)
        from sklearn.metrics import roc_curve
        
        # Plot ROC for consensus model
        fpr, tpr, _ = roc_curve(y, y_pred_proba)
        plt.plot(fpr, tpr, label=f'Consensus (AUC = {self.metrics["auc"]:.4f})', linewidth=2)
        
        # Plot ROC for each base model
        for i, model_wrapper in enumerate(self.model_wrappers):
            try:
                y_pred_proba_base = model_wrapper.predict_proba(X)[:, 1]
                fpr, tpr, _ = roc_curve(y, y_pred_proba_base)
                base_auc = roc_auc_score(y, y_pred_proba_base)
                plt.plot(fpr, tpr, alpha=0.7, label=f'Model {i+1} (AUC = {base_auc:.4f})')
            except Exception as e:
                print(f"Error plotting ROC for model {i+1}: {e}")
                continue
        
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves')
        plt.legend(loc='lower right')
        
        # 2. Model Weights
        plt.subplot(2, 3, 2)
        if isinstance(self.meta_model, LogisticRegression):
            weights = np.abs(self.meta_model.coef_[0])
            # Normalize weights
            weights = weights / weights.sum()
        else:  # VotingClassifier or custom model
            weights = np.ones(len(self.model_wrappers)) / len(self.model_wrappers)
            if hasattr(self.meta_model, 'weights') and self.meta_model.weights is not None:
                weights = np.array(self.meta_model.weights)
        
        plt.bar(range(len(weights)), weights)
        plt.xticks(range(len(weights)), [f'Model {i+1}' for i in range(len(weights))])
        plt.ylabel('Weight')
        plt.title('Model Weights in Consensus')
        
        # 3. Confusion Matrix
        plt.subplot(2, 3, 3)
        cm = confusion_matrix(y, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        
        # 4. Performance Comparison
        plt.subplot(2, 3, 4)
        metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1', 'auc']
        
        # Prepare data
        model_names = [f'Model {i+1}' for i in range(len(self.model_wrappers))] + ['Consensus']
        metric_values = []
        
        for i, model_wrapper in enumerate(self.model_wrappers):
            try:
                y_pred_proba_base = model_wrapper.predict_proba(X)[:, 1]
                y_pred_base = model_wrapper.predict(X)
                
                values = [
                    accuracy_score(y, y_pred_base),
                    precision_score(y, y_pred_base, zero_division=0),
                    recall_score(y, y_pred_base, zero_division=0),
                    f1_score(y, y_pred_base, zero_division=0),
                    roc_auc_score(y, y_pred_proba_base)
                ]
                metric_values.append(values)
            except Exception as e:
                print(f"Error calculating metrics for model {i+1}: {e}")
                # Use placeholder values
                metric_values.append([0.5, 0.5, 0.5, 0.5, 0.5])
        
        # Add consensus model
        metric_values.append([
            self.metrics['accuracy'],
            self.metrics['precision'],
            self.metrics['recall'],
            self.metrics['f1'],
            self.metrics['auc']
        ])
        
        # Convert to DataFrame for easier plotting
        plot_df = pd.DataFrame(metric_values, index=model_names, columns=metrics_to_plot)
        plot_df.plot(kind='bar', ax=plt.gca())
        plt.ylabel('Score')
        plt.title('Performance Metrics Comparison')
        plt.legend(loc='lower right')
        plt.xticks(rotation=45)
        
        # 5. Model Correlation Heatmap
        plt.subplot(2, 3, 5)
        try:
            corr_matrix = np.corrcoef(base_preds, rowvar=False)
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
            sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', vmin=-1, vmax=1,
                        square=True, fmt='.2f', cbar_kws={'shrink': .8})
            plt.title('Model Correlation Matrix')
        except Exception as e:
            print(f"Error creating correlation heatmap: {e}")
            plt.text(0.5, 0.5, "Correlation matrix unavailable", 
                     ha='center', va='center', fontsize=12)
            plt.title('Model Correlation Matrix (Error)')
        
        # Save figure
        plt.tight_layout()
        plt.savefig(f"{output_dir}/consensus_results.png", dpi=300)
        plt.close()
        
        # Save metrics
        pd.Series(self.metrics).to_csv(f"{output_dir}/metrics.csv")
        
        # Save model predictions for further analysis
        pred_df = pd.DataFrame({
            'true': y,
            'consensus': y_pred_proba
        })
        
        # Add base model predictions
        for i in range(base_preds.shape[1]):
            pred_df[f'model_{i+1}'] = base_preds[:, i]
        
        pred_df.to_csv(f"{output_dir}/predictions.csv", index=False)
        
        print(f"Results saved to {output_dir}")
        
        return self
    
    def save(self):
        """Save the consensus model"""
        # Create output directory
        os.makedirs('consensus_model', exist_ok=True)
        output_dir = f"consensus_model/{self.target_variable}"
        os.makedirs(output_dir, exist_ok=True)
        
        # Save meta-model
        joblib.dump(self.meta_model, f"{output_dir}/meta_model.pkl")
        
        # Save metrics
        pd.Series(self.metrics).to_csv(f"{output_dir}/metrics.csv")
        
        # Save feature names
        if self.feature_names is not None:
            pd.Series(self.feature_names).to_csv(f"{output_dir}/feature_names.csv", index=False)
        
        print(f"Consensus model saved to {output_dir}")
        return self
    
    def load(self):
        """Load a saved consensus model"""
        output_dir = f"consensus_model/{self.target_variable}"
        
        if not os.path.exists(output_dir):
            raise ValueError(f"Model directory {output_dir} does not exist")
        
        # Load meta-model
        self.meta_model = joblib.load(f"{output_dir}/meta_model.pkl")
        
        # Load metrics
        self.metrics = pd.read_csv(f"{output_dir}/metrics.csv", index_col=0).squeeze().to_dict()
        
        # Load feature names if available
        if os.path.exists(f"{output_dir}/feature_names.csv"):
            self.feature_names = pd.read_csv(f"{output_dir}/feature_names.csv").squeeze().tolist()
        
        print(f"Consensus model loaded from {output_dir}")
        return self

# Train consensus models for both target variables
print("=" * 80)
print("CONSENSUS MODEL TRAINING (WITHOUT XGBOOST)")
print("=" * 80)

# Try to use existing cluster system if available
try:
    # Assume cluster_system is already defined in the notebook
    print("Using existing cluster system")
except NameError:
    print("Cluster system not available, using direct models only")
    cluster_system = None

# Define target variables
targets = ['IncomeInvestment', 'AccumulationInvestment']

# Store consensus model results
consensus_results = {}

# Train consensus models for each target
for target in targets:
    print("\n" + "=" * 80)
    print(f"TRAINING CONSENSUS MODEL FOR {target.upper()}")
    print("=" * 80)
    
    if target not in engineered_optimised_df.columns:
        print(f"Target {target} not found in dataframe")
        continue
    
    # Initialize consensus model
    consensus = ConsensusModel(target_variable=target, cluster_system=cluster_system if 'cluster_system' in locals() else None)
    
    # Load all available models
    consensus.load_models()
    
    # Skip if no models were loaded
    if not consensus.model_wrappers:
        print(f"No models loaded for {target}, skipping...")
        continue
    
    # Split data for meta-model training
    X = engineered_optimised_df.drop(columns=['IncomeInvestment', 'AccumulationInvestment'])
    y = engineered_optimised_df[target]
    
    # Train meta-model
    consensus.train_meta_model(X, y, cv=5, meta_model_type='advanced')
    
    # Plot and save results
    consensus.plot_results(X, y)
    consensus.save()
    
    # Store results
    consensus_results[target] = consensus

print("\nConsensus models training completed (without XGBoost)!")