import pandas as pd
import numpy as np

try:
    from sentence_transformers import SentenceTransformer # type: ignore
except ImportError:
    print("Warning: sentence_transformers not installed.")
    print("Please run: pip install sentence-transformers")

_model = None

def get_model():
    global _model
    if 'SentenceTransformer' not in globals():
        print("SentenceTransformer not installed. Skipping semantic embeddings.")
        return None
    if _model is None:
        print("Loading SentenceTransformer model (all-MiniLM-L6-v2)...")
        # all-MiniLM-L6-v2 is small (80MB), extremely fast, and highly effective for semantic matching
        _model = SentenceTransformer('all-MiniLM-L6-v2')
    return _model

def add_semantic_features(features_df, s1_df, corpus_df):
    """
    Given a dataframe of extracted features (which includes source1_entity_id and candidate_entity_id),
    computes deep semantic cosine similarity and appends them as new columns.
    Optimized to only encode unique strings.
    """
    model = get_model()
    if model is None:
        return features_df
    
    # Fast O(1) lookup
    if s1_df.index.name != 'entity_id':
        s1_df = s1_df.set_index('entity_id')
    if corpus_df.index.name != 'entity_id':
        corpus_df = corpus_df.set_index('entity_id')
        
    print("Extracting unique text strings for 10x faster encoding...")
    unique_names = list(set(s1_df['norm_name'].dropna().astype(str).tolist() + corpus_df['norm_name'].dropna().astype(str).tolist()))
    unique_addrs = list(set(s1_df['norm_address'].dropna().astype(str).tolist() + corpus_df['norm_address'].dropna().astype(str).tolist()))
    
    print(f"Encoding {len(unique_names)} unique names...")
    # normalize_embeddings=True converts the vectors to L2 length = 1, meaning dot product is exactly cosine similarity
    names_emb = model.encode(unique_names, batch_size=256, show_progress_bar=True, normalize_embeddings=True)
    name_emb_dict = {name: emb for name, emb in zip(unique_names, names_emb)}
    name_emb_dict[""] = np.zeros(names_emb.shape[1])
    
    print(f"Encoding {len(unique_addrs)} unique addresses...")
    addrs_emb = model.encode(unique_addrs, batch_size=256, show_progress_bar=True, normalize_embeddings=True)
    addr_emb_dict = {addr: emb for addr, emb in zip(unique_addrs, addrs_emb)}
    addr_emb_dict[""] = np.zeros(addrs_emb.shape[1])
    
    print(f"Mapping embeddings to {len(features_df)} pairs and computing fast dot-products...")
    
    s1_names = s1_df['norm_name'].fillna("").astype(str).to_dict()
    s1_addrs = s1_df['norm_address'].fillna("").astype(str).to_dict()
    corpus_names = corpus_df['norm_name'].fillna("").astype(str).to_dict()
    corpus_addrs = corpus_df['norm_address'].fillna("").astype(str).to_dict()
    
    name_sims = np.zeros(len(features_df))
    addr_sims = np.zeros(len(features_df))
    
    for i, row in enumerate(features_df.itertuples()):
        s1_id = row.source1_entity_id
        cand_id = row.candidate_entity_id
        
        q_n = s1_names.get(s1_id, "")
        c_n = corpus_names.get(cand_id, "")
        q_a = s1_addrs.get(s1_id, "")
        c_a = corpus_addrs.get(cand_id, "")
        
        q_n_emb = name_emb_dict.get(q_n)
        c_n_emb = name_emb_dict.get(c_n)
        if q_n_emb is not None and c_n_emb is not None:
            name_sims[i] = np.dot(q_n_emb, c_n_emb)
            
        q_a_emb = addr_emb_dict.get(q_a)
        c_a_emb = addr_emb_dict.get(c_a)
        if q_a_emb is not None and c_a_emb is not None:
            addr_sims[i] = np.dot(q_a_emb, c_a_emb)
            
    features_df['name_semantic_sim'] = name_sims
    features_df['addr_semantic_sim'] = addr_sims
    
    return features_df

if __name__ == "__main__":
    print("Phase 8: Semantic Embeddings module ready.")
    print("Function add_semantic_features() can be integrated into the main pipeline.")
