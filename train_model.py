import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, fbeta_score, classification_report
import pickle
import os

from sklearn.ensemble import HistGradientBoostingClassifier
MODEL_TYPE = 'hgb'

def train_model(X_train, y_train):
    print(f"Training Model ({MODEL_TYPE})...")
    
    if MODEL_TYPE == 'xgb':
        model = XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )
    elif MODEL_TYPE == 'hgb':
        model = HistGradientBoostingClassifier(
            max_iter=300,
            learning_rate=0.05,
            max_depth=6,
            random_state=42
        )
    else:
        model = RandomForestClassifier(n_estimators=150, max_depth=15, random_state=42, n_jobs=-1)
        
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_val, y_val):
    print("Evaluating Model...")
    
    # We predict probabilities
    y_probs = model.predict_proba(X_val)[:, 1]
    
    # Entity Resolution Challenge objective is F0.5
    # Let's find the optimal threshold for F0.5
    best_threshold = 0.5
    best_f05 = 0.0
    
    # Scan through potential thresholds
    for thresh in np.arange(0.3, 0.9, 0.05):
        y_pred = (y_probs >= thresh).astype(int)
        # beta=0.5 means Precision is weighted twice as heavily as Recall
        f05 = fbeta_score(y_val, y_pred, beta=0.5)
        if f05 > best_f05:
            best_f05 = f05
            best_threshold = thresh
            
    print(f"Optimal Threshold for F0.5: {best_threshold:.2f}")
    y_pred_optimal = (y_probs >= best_threshold).astype(int)
    
    p = precision_score(y_val, y_pred_optimal)
    r = recall_score(y_val, y_pred_optimal)
    f05 = fbeta_score(y_val, y_pred_optimal, beta=0.5)
    
    print("-" * 30)
    print(f"Validation Precision : {p:.4f}")
    print(f"Validation Recall    : {r:.4f}")
    print(f"Validation F0.5      : {f05:.4f}")
    print("-" * 30)
    
    return best_threshold, f05

def save_model(model, threshold, filepath="output/entity_resolution_model.pkl"):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump({'model': model, 'threshold': threshold}, f)
    print(f"Model saved to {filepath}")

def main():
    print("--- Phase 9: Model Training Pipeline ---")
    print("This module provides train_model(), evaluate_model(), and save_model().")
    print("To execute, pass the engineered feature matrices (X_train, X_val) to these functions.")
    
    # Feature Importances will be accessible via model.feature_importances_

if __name__ == "__main__":
    main()
