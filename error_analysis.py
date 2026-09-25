import pandas as pd
import pickle

def analyze_errors(model_path, X_val, y_val, threshold=0.5):
    print("--- Phase 15: Error Analysis ---")
    with open(model_path, 'rb') as f:
        data = pickle.load(f)
        model = data['model']
        # Use the tuned threshold if available
        threshold = data.get('threshold', threshold)
        
    y_probs = model.predict_proba(X_val)[:, 1]
    y_pred = (y_probs >= threshold).astype(int)
    
    X_val = X_val.copy()
    X_val['true_label'] = y_val
    X_val['predicted_label'] = y_pred
    X_val['prediction_confidence'] = y_probs
    
    false_positives = X_val[(X_val['true_label'] == 0) & (X_val['predicted_label'] == 1)]
    false_negatives = X_val[(X_val['true_label'] == 1) & (X_val['predicted_label'] == 0)]
    
    print(f"Total False Positives: {len(false_positives)}")
    print(f"Total False Negatives: {len(false_negatives)}")
    
    if len(false_positives) > 0:
        print("\nTop 5 Highest Confidence False Positives (Model thought they matched but they didn't):")
        print(false_positives.sort_values('prediction_confidence', ascending=False).head(5)[
            ['prediction_confidence', 'name_sim_norm', 'addr_sim_norm', 'name_semantic_sim', 'addr_semantic_sim']
        ])
        
    if len(false_negatives) > 0:
        print("\nTop 5 Highest Confidence False Negatives (Model thought they didn't match but they did):")
        print(false_negatives.sort_values('prediction_confidence', ascending=True).head(5)[
            ['prediction_confidence', 'name_sim_norm', 'addr_sim_norm', 'name_semantic_sim', 'addr_semantic_sim']
        ])
        
    print("\nError Analysis Complete.")
    return false_positives, false_negatives

if __name__ == "__main__":
    print("Error Analysis module loaded.")
