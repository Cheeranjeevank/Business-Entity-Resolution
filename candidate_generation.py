import pandas as pd
import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from normalization import normalize_dataframe
from collections import defaultdict
from scipy.sparse import csr_matrix

def create_blocking_key(row):
    name = str(row['norm_name']) if pd.notna(row['norm_name']) else ""
    addr = str(row['norm_address']) if pd.notna(row['norm_address']) else ""
    return name + " " + addr

def get_top_k_csr(mat: csr_matrix, k: int):
    n_queries = mat.shape[0]
    top_indices = np.zeros((n_queries, k), dtype=np.int32)
    
    for i in range(n_queries):
        start = mat.indptr[i]
        end = mat.indptr[i+1]
        data = mat.data[start:end]
        indices = mat.indices[start:end]
        
        if len(data) == 0:
            top_indices[i] = -1
            continue
            
        if len(data) <= k:
            idx_sorted = np.argsort(-data)
            valid_k = len(data)
            top_indices[i, :valid_k] = indices[idx_sorted]
            top_indices[i, valid_k:] = -1
        else:
            idx_top_k = np.argpartition(-data, k)[:k]
            idx_sorted = idx_top_k[np.argsort(-data[idx_top_k])]
            top_indices[i] = indices[idx_sorted]
            
    return top_indices

def tfidf_blocking(query_df, corpus_df, k=15):
    print(f"Fitting TF-IDF on Corpus (Size: {len(corpus_df)})...")
    vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=2, max_df=0.05)
    corpus_tfidf = vectorizer.fit_transform(corpus_df['blocking_key'])
    
    print(f"Transforming queries (Size: {len(query_df)})...")
    query_tfidf = vectorizer.transform(query_df['blocking_key'])
    
    print("Searching for top candidates (NUCLEAR FAST DOT-PRODUCT)...")
    batch_size = 50000
    all_indices = []
    
    corpus_tfidf_T = corpus_tfidf.T
    
    import sys
    for i in range(0, query_tfidf.shape[0], batch_size):
        batch = query_tfidf[i:i+batch_size]
        sim_matrix = batch.dot(corpus_tfidf_T)
        indices = get_top_k_csr(sim_matrix.tocsr(), k)
        all_indices.append(indices)
        print(f"Processed batch {i} to {i+batch_size}", flush=True)
        
    all_indices = np.vstack(all_indices)
    
    candidates = defaultdict(set)
    for i, row_indices in enumerate(all_indices):
        s1_id = query_df.iloc[i]['entity_id']
        # filter out -1
        valid_idx = row_indices[row_indices != -1]
        matched_ids = corpus_df.iloc[valid_idx]['entity_id'].values
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
