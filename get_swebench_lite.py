#!/usr/bin/env python3
"""
Script to download SWE-bench Lite dataset and export to JSON.

Usage:
    pip install datasets
    python get_swebench_lite.py

Output:
    swebench_lite.json - Full dataset with all fields
"""

import json
from datasets import load_dataset

DATASET_NAME = "princeton-nlp/SWE-bench_Lite"
SPLIT = "test"
OUTPUT_FILE = "swebench_lite.json"

def main():
    print(f"Loading dataset: {DATASET_NAME} (split: {SPLIT})")
    dataset = load_dataset(DATASET_NAME, split=SPLIT)
    
    print(f"Loaded {len(dataset)} instances")
    print(f"Columns: {dataset.column_names}")
    
    # Convert to list of dicts
    data = []
    for item in dataset:
        data.append({
            'instance_id': item['instance_id'],
            'repo': item['repo'],
            'problem_statement': item['problem_statement'],
            'hints_text': item.get('hints_text', ''),
            'patch': item.get('patch', ''),
            'test_patch': item.get('test_patch', ''),
            'version': item.get('version', ''),
            'base_commit': item.get('base_commit', ''),
            'created_at': item.get('created_at', ''),
        })
    
    # Save to JSON
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"\nSaved {len(data)} instances to {OUTPUT_FILE}")
    
    # Print summary by repo
    repos = {}
    for item in data:
        repo = item['repo']
        repos[repo] = repos.get(repo, 0) + 1
    
    print(f"\nInstances by repository:")
    for repo in sorted(repos.keys(), key=lambda x: -repos[x]):
        print(f"  {repo}: {repos[repo]}")

if __name__ == "__main__":
    main()