import pandas as pd
import numpy as np
import pickle
from normalization import normalize_dataframe
from feature_engineering import process_candidate_pairs
from semantic_embeddings import add_semantic_features
from candidate_generation import tfidf_blocking

def generate_predictions(test_s1_path, test_s2_path, test_s3_path, model_path, output_path):
    print("--- Phase 19: Final Predictions & Submission ---")
    
    # 1. Load Data
    s1 = pd.read_csv(test_s1_path, sep="\t", dtype=str)
    s2 = pd.read_csv(test_s2_path, sep="\t", dtype=str)
    s3 = pd.read_csv(test_s3_path, sep="\t", dtype=str)
    
    # 2. Normalize
    print("Normalizing test datasets...")
    s1 = normalize_dataframe(s1)
    s2 = normalize_dataframe(s2)
    s3 = normalize_dataframe(s3)
    corpus = pd.concat([s2, s3], ignore_index=True)
    
    s1['blocking_key'] = s1['norm_name'].fillna("") + " " + s1['norm_address'].fillna("")
    corpus['blocking_key'] = corpus['norm_name'].fillna("") + " " + corpus['norm_address'].fillna("")
    
    # 3. Generate Candidates
    print("Generating Candidates...")
    candidates_dict = tfidf_blocking(s1, corpus, k=15)
    
    pairs = []
    for s1_id, cands in candidates_dict.items():
        for cand_id in cands:
            pairs.append({'source1_entity_id': s1_id, 'candidate_entity_id': cand_id})
    pairs_df = pd.DataFrame(pairs)
    
    # 4. Feature Engineering
    print("Generating Features...")
    X_test = process_candidate_pairs(s1, corpus, pairs_df)
    X_test = add_semantic_features(X_test, s1, corpus)
    
    features = [c for c in X_test.columns if c not in ['source1_entity_id', 'candidate_entity_id']]
    
    # 5. Predict
    print("Running Model Inference...")
    with open(model_path, 'rb') as f:
        data = pickle.load(f)
        model = data['model']
        threshold = data.get('threshold', 0.5)
        
    y_probs = model.predict_proba(X_test[features])[:, 1]
    y_pred = (y_probs >= threshold).astype(int)
    
    # 6. Format Submission
    print("Formatting Submission...")
    X_test['prediction'] = y_pred
    positive_predictions = X_test[X_test['prediction'] == 1]
    
    submission_dict = {}
    for idx, row in positive_predictions.iterrows():
        s1_id = row['source1_entity_id']
        cand_id = row['candidate_entity_id']
        if s1_id not in submission_dict:
            submission_dict[s1_id] = []
        submission_dict[s1_id].append(cand_id)
        
    # Convert to required format
    sub_rows = []
    for s1_id, matches in submission_dict.items():
        sub_rows.append({
            'source1_entity_id': s1_id,
            'matched_entity_ids': ",".join(matches)
        })
        
    sub_df = pd.DataFrame(sub_rows)
    sub_df.to_csv(output_path, index=False, sep="\t")
    print(f"Submission saved to {output_path}")

if __name__ == "__main__":
    generate_predictions(
        test_s1_path='student_resource/dataset/test/test_source1.tsv',
        test_s2_path='student_resource/dataset/test/test_source2.tsv',
        test_s3_path='student_resource/dataset/test/test_source3.tsv',
        model_path='output/entity_resolution_model.pkl',
        output_path='output/submission.tsv'
    )
