import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
import os
import json
import re

def normalize_name(name):
    if pd.isna(name):
        return ""
    name = str(name).lower()
    name = re.sub(r'[^a-z0-9]', '', name)
    return name

def main():
    print("Loading Source 1 and Ground Truth...")
    s1 = pd.read_csv("student_resource/dataset/train/train_source1.tsv", sep="\t", dtype=str)
    gt = pd.read_csv("student_resource/dataset/train/train_ground_truth.tsv", sep="\t", dtype=str)
    
    print("Normalizing business names to create groups...")
    s1['norm_name'] = s1['business_name'].apply(normalize_name)
    
    # We want to ensure businesses with the same normalized name end up in the same split.
    groups = s1['norm_name'].values
    
    print("Performing GroupShuffleSplit (80/20)...")
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(gss.split(s1, groups=groups))
    
    train_s1 = s1.iloc[train_idx].copy()
    val_s1 = s1.iloc[val_idx].copy()
    
    # Filter ground truth based on the split
    print("Filtering ground truth...")
    train_gt = gt[gt['source1_entity_id'].isin(train_s1['entity_id'])].copy()
    val_gt = gt[gt['source1_entity_id'].isin(val_s1['entity_id'])].copy()
    
    # Calculate stats
    def get_stats(df_s1, df_gt, name):
        df_gt['matches_list'] = df_gt['matched_entity_ids'].fillna('').apply(lambda x: [i for i in x.split(',') if i])
        num_matches = df_gt['matches_list'].apply(len)
        total_positives = int(num_matches.sum())
        total_entities = len(df_s1)
        match_rate = total_positives / total_entities if total_entities > 0 else 0
        
        # Source distribution in positives
        s2_matches = df_gt['matches_list'].apply(lambda x: sum(1 for i in x if i.startswith('S2-'))).sum()
        s3_matches = df_gt['matches_list'].apply(lambda x: sum(1 for i in x if i.startswith('S3-'))).sum()
        
        return {
            "split_name": name,
            "entities": total_entities,
            "positive_pairs": total_positives,
            "match_rate_per_entity": match_rate,
            "s2_positives": int(s2_matches),
            "s3_positives": int(s3_matches)
        }
    
    print("Calculating statistics...")
    train_stats = get_stats(train_s1, train_gt, "Train")
    val_stats = get_stats(val_s1, val_gt, "Validation")
    
    # Save splits
    os.makedirs("student_resource/dataset/splits", exist_ok=True)
    
    print("Saving splits...")
    train_s1.drop(columns=['norm_name']).to_csv("student_resource/dataset/splits/train_s1.tsv", sep="\t", index=False)
    val_s1.drop(columns=['norm_name']).to_csv("student_resource/dataset/splits/val_s1.tsv", sep="\t", index=False)
    train_gt.drop(columns=['matches_list'], errors='ignore').to_csv("student_resource/dataset/splits/train_gt.tsv", sep="\t", index=False)
    val_gt.drop(columns=['matches_list'], errors='ignore').to_csv("student_resource/dataset/splits/val_gt.tsv", sep="\t", index=False)
    
    stats = {"train": train_stats, "val": val_stats}
    with open("phase2_stats.json", "w") as f:
        json.dump(stats, f, indent=4)
        
    print("Phase 2 complete.")

if __name__ == "__main__":
    main()
