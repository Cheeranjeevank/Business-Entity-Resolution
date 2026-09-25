import pandas as pd
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("Warning: sentence_transformers not installed.")
    print("Please run: pip install sentence-transformers")

_model = None

def get_model():
    global _model
    if _model is None:
        print("Loading SentenceTransformer model (all-MiniLM-L6-v2)...")
        # all-MiniLM-L6-v2 is small (80MB), extremely fast, and highly effective for semantic matching
        _model = SentenceTransformer('all-MiniLM-L6-v2')
    return _model

def add_semantic_features(features_df, s1_df, corpus_df):
    """
    Given a dataframe of extracted features (which includes source1_entity_id and candidate_entity_id),
    computes deep semantic cosine similarity and appends them as new columns.
    """
    model = get_model()
    
    # Fast O(1) lookup
    if s1_df.index.name != 'entity_id':
        s1_df = s1_df.set_index('entity_id')
    if corpus_df.index.name != 'entity_id':
        corpus_df = corpus_df.set_index('entity_id')
        
    q_names, c_names = [], []
    q_addrs, c_addrs = [], []
    
    print(f"Extracting strings for semantic encoding of {len(features_df)} pairs...")
    for idx, row in features_df.iterrows():
        s1_id = row['source1_entity_id']
        cand_id = row['candidate_entity_id']
        
        q_names.append(str(s1_df.loc[s1_id, 'norm_name']) if pd.notna(s1_df.loc[s1_id, 'norm_name']) else "")
        c_names.append(str(corpus_df.loc[cand_id, 'norm_name']) if pd.notna(corpus_df.loc[cand_id, 'norm_name']) else "")
        
        q_addrs.append(str(s1_df.loc[s1_id, 'norm_address']) if pd.notna(s1_df.loc[s1_id, 'norm_address']) else "")
        c_addrs.append(str(corpus_df.loc[cand_id, 'norm_address']) if pd.notna(corpus_df.loc[cand_id, 'norm_address']) else "")
        
    print("Encoding Source 1 Names...")
    # normalize_embeddings=True converts the vectors to L2 length = 1, meaning dot product is exactly cosine similarity
    q_names_emb = model.encode(q_names, batch_size=256, show_progress_bar=True, normalize_embeddings=True)
    
    print("Encoding Candidate Names...")
    c_names_emb = model.encode(c_names, batch_size=256, show_progress_bar=True, normalize_embeddings=True)
    
    print("Encoding Source 1 Addresses...")
    q_addrs_emb = model.encode(q_addrs, batch_size=256, show_progress_bar=True, normalize_embeddings=True)
    
    print("Encoding Candidate Addresses...")
    c_addrs_emb = model.encode(c_addrs, batch_size=256, show_progress_bar=True, normalize_embeddings=True)
    
    print("Computing Cosine Similarities via Matrix Dot Product...")
    features_df['name_semantic_sim'] = np.sum(q_names_emb * c_names_emb, axis=1)
    features_df['addr_semantic_sim'] = np.sum(q_addrs_emb * c_addrs_emb, axis=1)
    
    return features_df

if __name__ == "__main__":
    print("Phase 8: Semantic Embeddings module ready.")
    print("Function add_semantic_features() can be integrated into the main pipeline.")
