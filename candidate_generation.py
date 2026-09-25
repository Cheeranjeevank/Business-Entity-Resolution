import pandas as pd
import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from normalization import normalize_dataframe
from collections import defaultdict

def create_blocking_key(row):
    name = str(row['norm_name']) if pd.notna(row['norm_name']) else ""
    addr = str(row['norm_address']) if pd.notna(row['norm_address']) else ""
    return name + " " + addr

def tfidf_blocking(query_df, corpus_df, k=30):
    print(f"Fitting TF-IDF on Corpus (Size: {len(corpus_df)})...")
    # Using word n-grams (1-2) which are sparse and memory efficient
    vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=2)
    corpus_tfidf = vectorizer.fit_transform(corpus_df['blocking_key'])
    
    print("Building NearestNeighbors index (this may take a moment)...")
    nn = NearestNeighbors(n_neighbors=min(k, len(corpus_df)), metric='cosine', n_jobs=-1)
    nn.fit(corpus_tfidf)
    
    print(f"Transforming queries (Size: {len(query_df)})...")
    query_tfidf = vectorizer.transform(query_df['blocking_key'])
    
    print("Searching for top candidates...")
    # To avoid memory errors, search in batches if query is too large
    batch_size = 10000
    all_indices = []
    
    for i in range(0, query_tfidf.shape[0], batch_size):
        batch = query_tfidf[i:i+batch_size]
        _, indices = nn.kneighbors(batch)
        all_indices.append(indices)
        
    all_indices = np.vstack(all_indices)
    
    candidates = defaultdict(set)
    for i, row_indices in enumerate(all_indices):
        s1_id = query_df.iloc[i]['entity_id']
        matched_ids = corpus_df.iloc[row_indices]['entity_id'].values
        candidates[s1_id].update(matched_ids)
        
    return candidates

def evaluate_recall(candidates, gt):
    gt_dict = {}
    for _, row in gt.iterrows():
        if pd.notna(row['matched_entity_ids']) and str(row['matched_entity_ids']).strip():
            gt_dict[row['source1_entity_id']] = set(row['matched_entity_ids'].split(','))
        else:
            gt_dict[row['source1_entity_id']] = set()
            
    total_positives = 0
    recovered_positives = 0
    
    for s1_id, true_matches in gt_dict.items():
        total_positives += len(true_matches)
        if s1_id in candidates:
            recovered_positives += len(true_matches.intersection(candidates[s1_id]))
            
    recall = recovered_positives / total_positives if total_positives > 0 else 0
    print(f"Candidate Recall: {recall:.4f} ({recovered_positives}/{total_positives})")
    return recall

def main():
    print("--- Phase 4: Candidate Generation ---")
    
    # We will test on a smaller sample (e.g. 50k rows) to prove the pipeline works fast locally.
    # In production, this can be run on the entire dataset.
    N_ROWS = 50000
    
    print(f"Loading {N_ROWS} rows of data for demonstration...")
    s1 = pd.read_csv("student_resource/dataset/train/train_source1.tsv", sep="\t", dtype=str, nrows=N_ROWS)
    s2 = pd.read_csv("student_resource/dataset/train/train_source2.tsv", sep="\t", dtype=str, nrows=N_ROWS)
    s3 = pd.read_csv("student_resource/dataset/train/train_source3.tsv", sep="\t", dtype=str, nrows=N_ROWS)
    gt = pd.read_csv("student_resource/dataset/train/train_ground_truth.tsv", sep="\t", dtype=str, nrows=N_ROWS)
    
    s1 = normalize_dataframe(s1)
    s2 = normalize_dataframe(s2)
    s3 = normalize_dataframe(s3)
    
    s1['blocking_key'] = s1.apply(create_blocking_key, axis=1)
    s2['blocking_key'] = s2.apply(create_blocking_key, axis=1)
    s3['blocking_key'] = s3.apply(create_blocking_key, axis=1)
    
    cand_s2 = tfidf_blocking(s1, s2, k=15)
    cand_s3 = tfidf_blocking(s1, s3, k=15)
    
    final_candidates = defaultdict(set)
    for k, v in cand_s2.items():
        final_candidates[k].update(v)
    for k, v in cand_s3.items():
        final_candidates[k].update(v)
        
    print("\nEvaluating Blocking Strategy on Ground Truth:")
    evaluate_recall(final_candidates, gt)
    
    # Save the output in the required format
    output_rows = []
    for s1_id in s1['entity_id']:
        cands = list(final_candidates.get(s1_id, []))
        output_rows.append({
            'source1_entity_id': s1_id,
            'candidate_entity_ids': ",".join(cands)
        })
        
    out_df = pd.DataFrame(output_rows)
    os.makedirs("output", exist_ok=True)
    out_df.to_csv("output/candidate_pairs.tsv", sep="\t", index=False)
    print("\nSaved output/candidate_pairs.tsv successfully.")

if __name__ == "__main__":
    main()
