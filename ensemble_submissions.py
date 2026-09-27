#!/usr/bin/env python3
"""Consensus & Weighted Blending Ensemble for Amazon ML Challenge 2026.

Merges predictions from Teammate 1 (e5-small + e5-base), Teammate 2 (DeBERTa),
and Teammate 3 (BGE-M3) into a unified master submission package.

Usage:
    python ensemble_submissions.py \
        --inputs matching_team1.tsv matching_team2.tsv matching_team3.tsv \
        --candidates candidate_pairs.tsv \
        --code-dir code/business_entity_resolution \
        --doc Documentation_template.md \
        --out final_submission.zip \
        --min-votes 2
"""
import argparse
import collections
import os
import subprocess
import sys


def parse_matches(tsv_path):
    """Read matching_results.tsv into dict: {source1_id: set(matched_query_ids)}"""
    matches = {}
    with open(tsv_path, "r", encoding="utf-8") as f:
        header = f.readline().rstrip("\r\n")
        assert "source1_entity_id" in header and "matched_entity_ids" in header, (
            f"Invalid header in {tsv_path}: {header}"
        )
        for line in f:
            line = line.rstrip("\r\n")
            if not line:
                continue
            parts = line.split("\t")
            s1_id = parts[0]
            m_str = parts[1] if len(parts) > 1 else ""
            if m_str.strip():
                matches[s1_id] = [m.strip() for m in m_str.split(",") if m.strip()]
            else:
                matches[s1_id] = []
    return matches


def ensemble_matches(input_files, min_votes=2):
    """Perform consensus voting across multiple prediction files.
    
    If min_votes=2, a query-S1 match is accepted if at least 2 models agreed.
    If only 1 file is provided, it passes through directly.
    """
    print(f"Reading {len(input_files)} prediction files for consensus voting...")
    all_predictions = []
    all_s1_keys = []
    
    for fpath in input_files:
        print(f"  Loading: {fpath}")
        preds = parse_matches(fpath)
        all_predictions.append(preds)
        if not all_s1_keys:
            all_s1_keys = list(preds.keys())

    consensus_matches = {}
    total_matches_kept = 0
    single_model_rejected = 0

    threshold_votes = min(min_votes, len(input_files))
    print(f"\nApplying Consensus Voting Rule (Threshold: >= {threshold_votes} model agreement)...")

    for s1_id in all_s1_keys:
        vote_counts = collections.Counter()
        for preds in all_predictions:
            for q_id in preds.get(s1_id, []):
                vote_counts[q_id] += 1
                
        kept = []
        for q_id, votes in vote_counts.items():
            if votes >= threshold_votes:
                kept.append(q_id)
                total_matches_kept += 1
            else:
                single_model_rejected += 1
                
        consensus_matches[s1_id] = kept

    print(f"Consensus Statistics:")
    print(f"  Total Verified Matches Kept: {total_matches_kept:,}")
    print(f"  Lone False-Positive Decoys Pruned: {single_model_rejected:,}")
    return all_s1_keys, consensus_matches


def main():
    parser = argparse.ArgumentParser(description="Multi-Model Consensus Ensemble")
    parser.add_argument("--inputs", nargs="+", required=True, help="List of matching_results.tsv files")
    parser.add_argument("--candidates", required=True, help="Path to valid candidate_pairs.tsv")
    parser.add_argument("--code-dir", default="code/business_entity_resolution", help="Path to code folder")
    parser.add_argument("--doc", default="Documentation_template.md", help="Path to Documentation_template.md")
    parser.add_argument("--out", default="final_submission.zip", help="Output submission zip path")
    parser.add_argument("--min-votes", type=int, default=2, help="Minimum model votes to accept a match")
    args = parser.parse_args()

    s1_keys, consensus = ensemble_matches(args.inputs, min_votes=args.min_votes)

    out_dir = "ensemble_output"
    os.makedirs(out_dir, exist_ok=True)
    out_matching = os.path.join(out_dir, "matching_results.tsv")
    out_candidates = os.path.join(out_dir, "candidate_pairs.tsv")

    print(f"\nWriting unified matching_results.tsv to {out_matching}...")
    with open(out_matching, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for s1_id in s1_keys:
            matched_str = ",".join(consensus[s1_id])
            f.write(f"{s1_id}\t{matched_str}\n")

    # Copy / ensure candidate_pairs.tsv in out_dir
    if os.path.abspath(args.candidates) != os.path.abspath(out_candidates):
        import shutil
        shutil.copy(args.candidates, out_candidates)

    print(f"Packaging verified submission zip: {args.out}...")
    build_script = "build_submission_zip.py"
    if not os.path.exists(build_script):
        build_script = os.path.join(os.path.dirname(__file__), "build_submission_zip.py")

    cmd = [
        sys.executable, build_script,
        "--team", "final",
        "--matching", out_matching,
        "--candidates", out_candidates,
        "--code-dir", args.code_dir,
        "--doc", args.doc,
        "--out-dir", os.path.dirname(os.path.abspath(args.out)) or ".",
        "--force",
        "--check-subset"
    ]
    subprocess.check_call(cmd)
    print(f"\n>>> [SUCCESS] Master Ensemble Package built and verified successfully: {args.out}")


if __name__ == "__main__":
    main()
