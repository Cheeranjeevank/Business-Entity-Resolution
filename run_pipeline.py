import pandas as pd
import numpy as np
import os
import gc
from sklearn.model_selection import train_test_split

# Import our custom modules
from normalization import normalize_dataframe
from candidate_generation import tfidf_blocking
from feature_engineering import process_candidate_pairs
from semantic_embeddings import add_semantic_features
from train_model import train_model, evaluate_model, save_model

def load_data(n_rows=None):
    print(f"Loading data (n_rows={n_rows if n_rows else 'ALL'})...")
    s1 = pd.read_csv("student_resource/dataset/train/train_source1.tsv", sep="\t", dtype=str, nrows=n_rows)
    s2 = pd.read_csv("student_resource/dataset/train/train_source2.tsv", sep="\t", dtype=str, nrows=n_rows)
    s3 = pd.read_csv("student_resource/dataset/train/train_source3.tsv", sep="\t", dtype=str, nrows=n_rows)
    gt = pd.read_csv("student_resource/dataset/train/train_ground_truth.tsv", sep="\t", dtype=str, nrows=n_rows)
    return s1, s2, s3, gt

def create_labels(candidate_dict, gt_df):
    """
    Creates a dataframe of (source1_id, candidate_id, target).
    Target is 1 if candidate_id is in gt_df for source1_id, else 0.
    """
    print("Creating positive and negative labels from Ground Truth...")
    
    gt_dict = {}
    for _, row in gt_df.iterrows():
        if pd.notna(row['matched_entity_ids']) and str(row['matched_entity_ids']).strip():
            gt_dict[row['source1_entity_id']] = set(row['matched_entity_ids'].split(','))
        else:
            gt_dict[row['source1_entity_id']] = set()
            
    rows = []
    for s1_id, candidates in candidate_dict.items():
        true_matches = gt_dict.get(s1_id, set())
        for cand_id in candidates:
            target = 1 if cand_id in true_matches else 0
            rows.append({
                'source1_entity_id': s1_id,
                'candidate_entity_id': cand_id,
                'target': target
            })
            
    # Inject missed true positives into training set so the model learns from them
    for s1_id, true_matches in gt_dict.items():
        cands = candidate_dict.get(s1_id, set())
        missed = true_matches - cands
        for cand_id in missed:
            rows.append({
                'source1_entity_id': s1_id,
                'candidate_entity_id': cand_id,
                'target': 1
            })
            
    return pd.DataFrame(rows)

def prepare_corpus(s2, s3):
    print("Combining S2 and S3 into single corpus...")
    corpus = pd.concat([s2, s3], ignore_index=True)
    return corpus

def main():
    # -------------------------------------------------------------------------
    # IMPORTANT: We use 50,000 rows here so the script completes in a few minutes. 
    # Processing the entire 2GB dataset (N_ROWS = None) locally without a GPU 
    # for Sentence-Transformers will take a very long time!
    # -------------------------------------------------------------------------
    N_ROWS = 50000 
    
    s1, s2, s3, gt = load_data(N_ROWS)
    
    # 1. Normalization
    print("\n--- Running Phase 3: Normalization ---")
    s1 = normalize_dataframe(s1)
    s2 = normalize_dataframe(s2)
    s3 = normalize_dataframe(s3)
    corpus = prepare_corpus(s2, s3)
    
    s1['blocking_key'] = s1['norm_name'].fillna("") + " " + s1['norm_address'].fillna("")
    corpus['blocking_key'] = corpus['norm_name'].fillna("") + " " + corpus['norm_address'].fillna("")
    
    # 2. Candidate Generation
    print("\n--- Running Phase 4: Candidate Generation ---")
    candidates_dict = tfidf_blocking(s1, corpus, k=15)
    
    # 3. Label Creation
    print("\n--- Preparing Training Dataset ---")
    pairs_df = create_labels(candidates_dict, gt)
    print(f"Generated {len(pairs_df)} candidate pairs for training.")
    print(f"Positives: {pairs_df['target'].sum()} | Negatives: {len(pairs_df) - pairs_df['target'].sum()}")
    
    del candidates_dict
    gc.collect()
    
    # 4. Feature Engineering
    print("\n--- Running Phase 5&6: Feature Engineering ---")
    X_df = process_candidate_pairs(s1, corpus, pairs_df)
    
    # 5. Semantic Embeddings
    print("\n--- Running Phase 8: Semantic Embeddings ---")
    X_df = add_semantic_features(X_df, s1, corpus)
    
    # 6. Train/Val Split (80/20)
    print("\n--- Splitting Data ---")
    features = [c for c in X_df.columns if c not in ['source1_entity_id', 'candidate_entity_id', 'target']]
    X_train, X_val, y_train, y_val = train_test_split(
        X_df[features], X_df['target'], test_size=0.2, random_state=42
    )
    
    # 7. Model Training
    print("\n--- Running Phase 9: Model Training ---")
    model = train_model(X_train, y_train)
    
    # 8. Evaluation
    print("\n--- Evaluating Model ---")
    threshold, f05 = evaluate_model(model, X_val, y_val)
    
    # 9. Save Output
    save_model(model, threshold, "output/entity_resolution_model.pkl")
    
    if hasattr(model, 'feature_importances_'):
        print("\nFeature Importances:")
        importances = pd.DataFrame({'Feature': features, 'Importance': model.feature_importances_})
        print(importances.sort_values(by='Importance', ascending=False).to_string(index=False))

if __name__ == "__main__":
    main()
