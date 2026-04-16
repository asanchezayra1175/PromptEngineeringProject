"""
eval_localization_accuracy.py

Evaluates how accurately the Localization Agent identified the correct
files and symbols by comparing its output against the gold patch.

For each instance with a localization.json:
  - Extracts which files were actually changed in the gold patch
  - Extracts which functions/classes were actually changed in the gold patch
  - Compares against relevant_files and relevant_symbols in localization.json
  - Reports file-level and symbol-level hit rates

Usage:
    python eval_localization_accuracy.py
    python eval_localization_accuracy.py --states-dir ./states --split dev
    python eval_localization_accuracy.py --states-dir ./states/localization/my-run --split test

Environment variables (or .env):
    DATASET_NAME : dataset to load (default: princeton-nlp/SWE-bench_Lite)
    SPLIT        : dev or test (default: dev)
    LOCALIZATION_STATE_DIR : where localization.json files live (default: ./states)
"""

from dotenv import load_dotenv
load_dotenv()

import argparse
import json
import os
import re
from pathlib import Path
from collections import defaultdict
from datasets import load_dataset
from unidiff import PatchSet

DATASET_NAME = os.environ.get("DATASET_NAME", "princeton-nlp/SWE-bench_Lite")
SPLIT        = os.environ.get("SPLIT", "dev")
STATE_DIR    = Path(os.environ.get("LOCALIZATION_STATE_DIR", "./states"))


# ---------------------------------------------------------------------------
# Gold patch parsing
# ---------------------------------------------------------------------------

def parse_gold_files(patch_str: str) -> set[str]:
    """
    Extract the set of files modified by the gold patch.
    Returns paths relative to repo root (no leading a/ or b/).
    """
    if not patch_str or not patch_str.strip():
        return set()
    try:
        ps = PatchSet(patch_str)
        files = set()
        for pf in ps:
            # unidiff gives paths like 'a/src/file.py' — strip leading a/ or b/
            path = pf.path
            path = re.sub(r'^[ab]/', '', path)
            files.add(path)
        return files
    except Exception:
        # Fallback: regex parse diff --git a/... b/...
        files = set()
        for m in re.finditer(r'^diff --git a/(.+?) b/', patch_str, re.MULTILINE):
            files.add(m.group(1))
        return files


def parse_gold_symbols(patch_str: str) -> set[str]:
    """
    Extract function/method/class names that were modified in the gold patch
    by scanning hunk headers (@@ ... @@ def foo) and changed lines.
    Returns a set of symbol names (not qualified — just the bare name).
    """
    if not patch_str or not patch_str.strip():
        return set()

    symbols = set()

    # Pattern 1: hunk headers often contain the enclosing function name
    # e.g. "@@ -59,4 +59,4 @@ class Rule_L060(BaseRule):"
    for m in re.finditer(r'^@@[^@]*@@\s*(.*)', patch_str, re.MULTILINE):
        context = m.group(1).strip()
        # Extract def/class names from hunk context
        for sym in re.finditer(r'\b(?:def|class)\s+(\w+)', context):
            symbols.add(sym.group(1))

    # Pattern 2: changed lines (+ or -) that define functions/classes
    for m in re.finditer(r'^[+-]\s*(?:def|class)\s+(\w+)', patch_str, re.MULTILINE):
        symbols.add(m.group(1))

    return symbols


# ---------------------------------------------------------------------------
# Localization result parsing
# ---------------------------------------------------------------------------

def load_localization(state_dir: Path, instance_id: str) -> dict | None:
    path = state_dir / instance_id / "localization.json"
    if not path.exists():
        return None
    try:
        text = path.read_text().strip()
        if not text:
            return None
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None


def normalize_file(path: str) -> str:
    """
    Normalize a file path so localization output and gold patch can be compared.
    Strips /testbed/ prefix and leading slashes.
    """
    path = path.strip()
    path = re.sub(r'^/testbed/', '', path)
    path = re.sub(r'^/', '', path)
    return path


def normalize_symbol(sym: str) -> str:
    """
    Normalize a symbol name: take only the last component of dotted names.
    e.g. 'Schema._invoke_field_validators' -> '_invoke_field_validators'
    """
    return sym.strip().split('.')[-1]


# ---------------------------------------------------------------------------
# Per-instance evaluation
# ---------------------------------------------------------------------------

def evaluate_instance(instance: dict, state_dir: Path) -> dict | None:
    instance_id = instance["instance_id"]
    loc = load_localization(state_dir, instance_id)

    if loc is None or loc.get("status") != "success":
        return None

    gold_patch = instance.get("patch", "")

    # Parse gold patch
    gold_files   = parse_gold_files(gold_patch)
    gold_symbols = parse_gold_symbols(gold_patch)

    # Parse localization output
    loc_files_raw   = loc.get("relevant_files", [])
    loc_symbols_raw = loc.get("relevant_symbols", [])

    loc_files   = {normalize_file(f) for f in loc_files_raw}
    loc_symbols = {normalize_symbol(s) for s in loc_symbols_raw}
    gold_files_norm   = {normalize_file(f) for f in gold_files}
    gold_symbols_norm = {normalize_symbol(s) for s in gold_symbols}

    # File-level: did localization include ALL files in the gold patch?
    file_hits    = loc_files & gold_files_norm
    file_misses  = gold_files_norm - loc_files
    file_extra   = loc_files - gold_files_norm

    file_recall    = len(file_hits) / len(gold_files_norm) if gold_files_norm else None
    file_precision = len(file_hits) / len(loc_files) if loc_files else None

    # Symbol-level: did localization include any symbol from the gold patch?
    symbol_hits   = loc_symbols & gold_symbols_norm
    symbol_misses = gold_symbols_norm - loc_symbols

    symbol_recall = len(symbol_hits) / len(gold_symbols_norm) if gold_symbols_norm else None

    # Perfect file match = all gold files identified, no extra files
    perfect_file = (file_misses == set()) and bool(gold_files_norm)

    return {
        "instance_id":       instance_id,
        "repo":              instance["repo"],
        "gold_files":        sorted(gold_files_norm),
        "loc_files":         sorted(loc_files),
        "file_hits":         sorted(file_hits),
        "file_misses":       sorted(file_misses),
        "file_extra":        sorted(file_extra),
        "file_recall":       round(file_recall, 3) if file_recall is not None else None,
        "file_precision":    round(file_precision, 3) if file_precision is not None else None,
        "perfect_file":      perfect_file,
        "gold_symbols":      sorted(gold_symbols_norm),
        "loc_symbols":       sorted(loc_symbols),
        "symbol_hits":       sorted(symbol_hits),
        "symbol_misses":     sorted(symbol_misses),
        "symbol_recall":     round(symbol_recall, 3) if symbol_recall is not None else None,
        "perfect_symbol":    (symbol_misses == set()) and bool(gold_symbols_norm),
    }


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(results: list[dict], split: str):
    total = len(results)
    if total == 0:
        print("No results to summarize.")
        return

    # File-level metrics
    perfect_file     = sum(1 for r in results if r["perfect_file"])
    has_all_files    = sum(1 for r in results if r["file_recall"] == 1.0)
    file_recalls     = [r["file_recall"] for r in results if r["file_recall"] is not None]
    file_precisions  = [r["file_precision"] for r in results if r["file_precision"] is not None]
    avg_file_recall  = sum(file_recalls) / len(file_recalls) if file_recalls else 0
    avg_file_prec    = sum(file_precisions) / len(file_precisions) if file_precisions else 0

    # Symbol-level metrics
    perfect_symbol   = sum(1 for r in results if r["perfect_symbol"])
    symbol_recalls   = [r["symbol_recall"] for r in results if r["symbol_recall"] is not None]
    avg_sym_recall   = sum(symbol_recalls) / len(symbol_recalls) if symbol_recalls else 0

    # Per-repo breakdown
    by_repo = defaultdict(list)
    for r in results:
        by_repo[r["repo"].split("/")[-1]].append(r)

    print(f"\n{'=' * 70}")
    print(f"LOCALIZATION ACCURACY — {split.upper()} SET")
    print(f"{'=' * 70}")
    print(f"Instances evaluated  : {total}")
    print()
    print(f"FILE-LEVEL:")
    print(f"  Perfect file match : {perfect_file}/{total} ({100*perfect_file/total:.1f}%)")
    print(f"  All gold files hit : {has_all_files}/{total} ({100*has_all_files/total:.1f}%)")
    print(f"  Avg file recall    : {avg_file_recall:.3f}")
    print(f"  Avg file precision : {avg_file_prec:.3f}")
    print()
    print(f"SYMBOL-LEVEL:")
    print(f"  Perfect symbol hit : {perfect_symbol}/{total} ({100*perfect_symbol/total:.1f}%)")
    print(f"  Avg symbol recall  : {avg_sym_recall:.3f}")
    print()

    # Per-instance detail
    print(f"{'Instance ID':<45} {'File R':<8} {'File P':<8} {'Sym R':<8} {'Perfect'}")
    print("-" * 80)
    for r in sorted(results, key=lambda x: x["instance_id"]):
        fr = f"{r['file_recall']:.2f}" if r['file_recall'] is not None else "N/A"
        fp = f"{r['file_precision']:.2f}" if r['file_precision'] is not None else "N/A"
        sr = f"{r['symbol_recall']:.2f}" if r['symbol_recall'] is not None else "N/A"
        pf = "✓" if r['perfect_file'] else "✗"
        print(f"{r['instance_id']:<45} {fr:<8} {fp:<8} {sr:<8} {pf}")

        if r["file_misses"]:
            print(f"  MISSED FILES: {r['file_misses']}")
        if r["file_extra"]:
            print(f"  EXTRA FILES:  {r['file_extra']}")
        if r["symbol_misses"]:
            print(f"  MISSED SYMS:  {r['symbol_misses']}")

    # Per-repo summary
    print(f"\n{'=' * 70}")
    print("PER-REPO BREAKDOWN")
    print(f"{'=' * 70}")
    print(f"{'Repo':<30} {'N':<5} {'Perfect File %':<17} {'Avg File R':<13} {'Avg Sym R'}")
    print("-" * 70)
    for repo, repo_results in sorted(by_repo.items()):
        n     = len(repo_results)
        pf    = sum(1 for r in repo_results if r["perfect_file"])
        afr   = [r["file_recall"] for r in repo_results if r["file_recall"] is not None]
        asr   = [r["symbol_recall"] for r in repo_results if r["symbol_recall"] is not None]
        afr_m = sum(afr)/len(afr) if afr else 0
        asr_m = sum(asr)/len(asr) if asr else 0
        print(f"{repo:<30} {n:<5} {f'{pf}/{n} ({100*pf/n:.0f}%)':<17} {afr_m:<13.3f} {asr_m:.3f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate localization accuracy against gold patches."
    )
    parser.add_argument(
        "--states-dir", type=Path, default=STATE_DIR,
        help="Directory containing {instance_id}/localization.json files"
    )
    parser.add_argument(
        "--split", type=str, default=SPLIT,
        help="Dataset split: dev or test (default: from SPLIT env var)"
    )
    parser.add_argument(
        "--dataset", type=str, default=DATASET_NAME,
        help="HuggingFace dataset name"
    )
    parser.add_argument(
        "--instance-ids", type=str, default="",
        help="Comma-separated instance IDs to evaluate (default: all with localization)"
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Save full results to a JSON file"
    )
    args = parser.parse_args()

    print(f"Loading dataset {args.dataset} split={args.split}...")
    dataset   = load_dataset(args.dataset, split=args.split)
    instances = list(dataset)

    # Filter
    filter_ids = [i.strip() for i in args.instance_ids.split(",") if i.strip()]
    if filter_ids:
        instances = [i for i in instances if i["instance_id"] in filter_ids]
    else:
        # Only evaluate instances that have a localization file
        instances = [
            i for i in instances
            if (args.states_dir / i["instance_id"] / "localization.json").exists()
        ]

    if not instances:
        print("No instances found. Check --states-dir and --instance-ids.")
        return

    print(f"Evaluating {len(instances)} instance(s) from {args.states_dir}...")

    results = []
    skipped = 0
    for instance in instances:
        result = evaluate_instance(instance, args.states_dir)
        if result is None:
            skipped += 1
            continue
        results.append(result)

    if skipped:
        print(f"Skipped {skipped} instances (no localization state or failed status)")

    print_summary(results, args.split)

    if args.output:
        args.output.write_text(json.dumps(results, indent=2))
        print(f"\nFull results saved to: {args.output}")
    else:
        # Auto-save alongside script
        out = Path("localization_accuracy.json")
        out.write_text(json.dumps(results, indent=2))
        print(f"\nFull results saved to: {out}")


if __name__ == "__main__":
    main()