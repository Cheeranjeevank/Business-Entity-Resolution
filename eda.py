import pandas as pd
import os
import json

def analyze_file(filepath):
    print(f"Analyzing {filepath}...")
    size = os.path.getsize(filepath)
    # Read in chunks to avoid memory issues if files are huge, but let's try reading the whole thing first since they might fit in memory.
    # The whole dataset is ~2GB, so individual files are maybe 500MB, pandas can handle 500MB.
    try:
        df = pd.read_csv(filepath, sep='\t', dtype=str)
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None
    
    rows = len(df)
    cols = len(df.columns)
    dtypes = df.dtypes.astype(str).to_dict()
    missing = (df.isnull().sum() / rows * 100).to_dict()
    unique = df.nunique().to_dict()
    duplicate_rows = df.duplicated().sum()
    
    dup_names = df.duplicated(subset=['business_name']).sum() if 'business_name' in df.columns else 0
    dup_addresses = df.duplicated(subset=['business_address']).sum() if 'business_address' in df.columns else 0
    
    return {
        "size_bytes": size,
        "rows": rows,
        "cols": cols,
        "columns": list(df.columns),
        "missing_percentage": missing,
        "unique_counts": unique,
        "duplicate_rows": int(duplicate_rows),
        "duplicate_names": int(dup_names),
        "duplicate_addresses": int(dup_addresses)
    }

def main():
    base_dir = "student_resource/dataset"
    train_dir = os.path.join(base_dir, "train")
    test_dir = os.path.join(base_dir, "test")
    
    files_to_analyze = {
        "train_s1": os.path.join(train_dir, "train_source1.tsv"),
        "train_s2": os.path.join(train_dir, "train_source2.tsv"),
        "train_s3": os.path.join(train_dir, "train_source3.tsv"),
        "train_gt": os.path.join(train_dir, "train_ground_truth.tsv"),
        "test_s1": os.path.join(test_dir, "test_source1.tsv"),
        "test_s2": os.path.join(test_dir, "test_source2.tsv"),
        "test_s3": os.path.join(test_dir, "test_source3.tsv"),
    }
    
    results = {}
    for key, path in files_to_analyze.items():
        if os.path.exists(path):
            results[key] = analyze_file(path)
        else:
            print(f"File not found: {path}")
            
    # Ground truth analysis
    print("Analyzing ground truth mappings...")
    gt_path = files_to_analyze["train_gt"]
    if os.path.exists(gt_path):
        gt = pd.read_csv(gt_path, sep='\t', dtype=str)
        gt['matched_entity_ids'] = gt['matched_entity_ids'].fillna('')
        
        # Split by comma
        gt['matches_list'] = gt['matched_entity_ids'].apply(lambda x: x.split(',') if x else [])
        gt['num_matches'] = gt['matches_list'].apply(len)
        
        results['ground_truth_stats'] = {
            "total_entities_s1": len(gt),
            "entities_with_matches": int((gt['num_matches'] > 0).sum()),
            "entities_with_no_matches": int((gt['num_matches'] == 0).sum()),
            "entities_with_multiple_matches": int((gt['num_matches'] > 1).sum()),
            "max_matches_for_one_entity": int(gt['num_matches'].max()),
            "total_positive_pairs": int(gt['num_matches'].sum())
        }
    
    with open("eda_report.json", "w") as f:
        json.dump(results, f, indent=4)
        
    print("EDA completed. Results saved to eda_report.json.")

if __name__ == "__main__":
    main()
