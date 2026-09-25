import pandas as pd
import numpy as np
import difflib

def string_similarity(s1, s2):
    if pd.isna(s1) or pd.isna(s2) or not s1 or not s2:
        return 0.0
    # difflib Ratio computes similarity in [0, 1]
    return difflib.SequenceMatcher(None, str(s1), str(s2)).ratio()

def token_jaccard(s1, s2):
    if pd.isna(s1) or pd.isna(s2) or not s1 or not s2:
        return 0.0
    set1 = set(str(s1).split())
    set2 = set(str(s2).split())
    if not set1 or not set2:
        return 0.0
    return len(set1.intersection(set2)) / len(set1.union(set2))

def length_diff(s1, s2):
    if pd.isna(s1) or pd.isna(s2):
        return -1
    return abs(len(str(s1)) - len(str(s2)))

def number_overlap(n1, n2):
    if pd.isna(n1) or pd.isna(n2) or not n1 or not n2:
        return 0
    set1 = set(str(n1).split())
    set2 = set(str(n2).split())
    if not set1 or not set2:
        return 0
    return len(set1.intersection(set2))

def build_features(query_row, candidate_row):
    """
    Computes pairwise features between a single Source 1 query and a Source 2/3 candidate.
    """
    features = {}
    
    # Raw Name Features
    features['name_sim_raw'] = string_similarity(query_row.get('business_name', ''), candidate_row.get('business_name', ''))
    features['name_jaccard_raw'] = token_jaccard(query_row.get('business_name', ''), candidate_row.get('business_name', ''))
    features['name_len_diff_raw'] = length_diff(query_row.get('business_name', ''), candidate_row.get('business_name', ''))
    
    # Normalized Name Features
    features['name_sim_norm'] = string_similarity(query_row.get('norm_name', ''), candidate_row.get('norm_name', ''))
    features['name_jaccard_norm'] = token_jaccard(query_row.get('norm_name', ''), candidate_row.get('norm_name', ''))
    features['name_len_diff_norm'] = length_diff(query_row.get('norm_name', ''), candidate_row.get('norm_name', ''))
    
    # Raw Address Features
    features['addr_sim_raw'] = string_similarity(query_row.get('business_address', ''), candidate_row.get('business_address', ''))
    features['addr_jaccard_raw'] = token_jaccard(query_row.get('business_address', ''), candidate_row.get('business_address', ''))
    
    # Normalized Address Features
    features['addr_sim_norm'] = string_similarity(query_row.get('norm_address', ''), candidate_row.get('norm_address', ''))
    features['addr_jaccard_norm'] = token_jaccard(query_row.get('norm_address', ''), candidate_row.get('norm_address', ''))
    
    # Specific Sub-Component Features
    q_postal = query_row.get('postal_code', '')
    c_postal = candidate_row.get('postal_code', '')
    features['postal_match'] = 1 if (q_postal and q_postal == c_postal) else 0
    
    q_nums = query_row.get('address_numbers', '')
    c_nums = candidate_row.get('address_numbers', '')
    features['address_number_overlap'] = number_overlap(q_nums, c_nums)
    
    # Country Match
    features['country_match'] = 1 if (query_row.get('country', '') == candidate_row.get('country', '')) else 0
    
    # Missing Value Flags (Phase 7 Boost)
    q_name_miss = 1 if not query_row.get('business_name', '') or pd.isna(query_row.get('business_name')) else 0
    c_name_miss = 1 if not candidate_row.get('business_name', '') or pd.isna(candidate_row.get('business_name')) else 0
    features['name_missing_query'] = q_name_miss
    features['name_missing_cand'] = c_name_miss
    
    q_addr_miss = 1 if not query_row.get('business_address', '') or pd.isna(query_row.get('business_address')) else 0
    c_addr_miss = 1 if not candidate_row.get('business_address', '') or pd.isna(candidate_row.get('business_address')) else 0
    features['addr_missing_query'] = q_addr_miss
    features['addr_missing_cand'] = c_addr_miss
    features['both_addr_missing'] = 1 if q_addr_miss and c_addr_miss else 0
    
    return features

def process_candidate_pairs(s1_df, corpus_df, candidate_pairs_df):
    """
    Iterates over the generated candidate pairs and builds the feature matrix X.
    candidate_pairs_df should have 'source1_entity_id', 'candidate_entity_id', and optionally 'target'
    """
    from joblib import Parallel, delayed
    import multiprocessing as mp
    
    # Index dataframes for O(1) fast lookup
    s1_df = s1_df.set_index('entity_id')
    corpus_df = corpus_df.set_index('entity_id')
    
    print(f"Extracting features for {len(candidate_pairs_df)} candidate pairs (Nuclear Parallel Processing)...")
    
    def process_chunk(chunk_df):
        X = []
        for idx, row in chunk_df.iterrows():
            s1_id = row['source1_entity_id']
            cand_id = row['candidate_entity_id']
            
            if s1_id not in s1_df.index or cand_id not in corpus_df.index:
                continue
                
            features = build_features(s1_df.loc[s1_id], corpus_df.loc[cand_id])
            features['source1_entity_id'] = s1_id
            features['candidate_entity_id'] = cand_id
            
            if 'target' in row:
                features['target'] = row['target']
                
            X.append(features)
        return pd.DataFrame(X)

    n_cores = max(1, mp.cpu_count() - 1)
    chunks = np.array_split(candidate_pairs_df, n_cores * 4) # Split into more chunks than cores for better distribution
    results = Parallel(n_jobs=n_cores, require='sharedmem')(delayed(process_chunk)(chunk) for chunk in chunks)
    
    return pd.concat(results, ignore_index=True)

if __name__ == "__main__":
    print("Feature Engineering module loaded successfully.")
    print("Use process_candidate_pairs() to transform candidate TSVs into ML-ready datasets.")
