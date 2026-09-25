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

def tfidf_blocking(query_df, corpus_df, k=15):
    print(f"Fitting TF-IDF on Corpus (Size: {len(corpus_df)})...")
    # Using word n-grams (1-2) which are sparse and memory efficient
    # EXACT ORIGINAL LOGIC TO GUARANTEE F0.5 SCORE
    vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=2)
    corpus_tfidf = vectorizer.fit_transform(corpus_df['blocking_key'])
    
    print("Building NearestNeighbors index (this may take a moment)...")
    nn = NearestNeighbors(n_neighbors=min(k, len(corpus_df)), metric='cosine', n_jobs=-1)
    nn.fit(corpus_tfidf)
    
    print(f"Transforming queries (Size: {len(query_df)})...")
    query_tfidf = vectorizer.transform(query_df['blocking_key'])
    
    print("Searching for top candidates (SAFE NEAREST NEIGHBORS - ~5 HOUR ETA)...")
    # To avoid memory errors, search in batches if query is too large
    batch_size = 10000
    all_indices = []
    
    import sys
    for i in range(0, query_tfidf.shape[0], batch_size):
        batch = query_tfidf[i:i+batch_size]
        _, indices = nn.kneighbors(batch)
        all_indices.append(indices)
        print(f"Processed batch {i} to {i+batch_size} / {query_tfidf.shape[0]}", flush=True)
        
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
            
    recall = recovered_positives / max(1, total_positives)
    print(f"Candidate Generation Recall: {recall:.4f} ({recovered_positives}/{total_positives})")
    return recall
