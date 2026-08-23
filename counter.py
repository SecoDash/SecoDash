#!/usr/bin/env python3
"""checkpoint_counter.py - Count and summarize repositories in SecoDash checkpoint files.

Scans checkpoint_*.jsonl files under the pipeline's output directory (or
a directory you point it at) and reports record counts, duplicate rates,
platform/language breakdowns, and a CSV summary.

Usage:
    python counter.py                                  # summarize every checkpoint under output/
    python counter.py --dir some/other/folder          # summarize checkpoints under a different root
    python counter.py --file path/to/checkpoint.jsonl  # summarize a single file
    python counter.py --keyword machine_learning       # filter by keyword/base name
    python counter.py --all                            # include sample record listings
"""
import argparse
import json
import re
from pathlib import Path
from typing import Dict, Optional
from collections import defaultdict
from datetime import datetime

from output_paths import OUTPUT_DIR, reports_dir


def count_jsonl_records(filepath: Path) -> Dict:
    """Count records in a JSONL checkpoint file with basic statistics.

    Returns a dict with: total, unique_urls, duplicate_count, platforms,
    languages, errors, sample (first 5 records).
    """
    stats = {
        "total": 0,
        "unique_urls": set(),
        "duplicate_count": 0,
        "platforms": defaultdict(int),
        "languages": defaultdict(int),
        "errors": 0,
        "sample": [],
    }

    if not filepath.exists():
        print(f"File not found: {filepath}")
        return stats

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if not line.strip():
                    continue

                try:
                    record = json.loads(line)
                    stats["total"] += 1

                    url = record.get("html_url") or record.get("web_url")
                    if url:
                        if url in stats["unique_urls"]:
                            stats["duplicate_count"] += 1
                        else:
                            stats["unique_urls"].add(url)

                    platform = record.get("source") or record.get("platform")
                    if platform:
                        stats["platforms"][platform] += 1

                    lang = record.get("primary_language") or record.get("language")
                    if lang:
                        stats["languages"][lang] += 1

                    if len(stats["sample"]) < 5:
                        stats["sample"].append({
                            "line": line_num,
                            "title": record.get("repo_title") or record.get("name", "Unknown"),
                            "url": url,
                            "lang": lang,
                            "platform": platform,
                        })

                except json.JSONDecodeError as e:
                    stats["errors"] += 1
                    print(f"  JSON decode error at line {line_num}: {e}")

    except Exception as e:
        print(f"Error reading file: {e}")

    return stats


def find_checkpoint_files(root: Path, keyword: Optional[str] = None) -> list:
    """Recursively find checkpoint_(raw|derived)_<base_name>.jsonl files under `root`."""
    pattern = r"checkpoint_(raw|derived)_([^.]*)\.jsonl"
    checkpoint_files = []

    for f in root.rglob("checkpoint_*.jsonl"):
        match = re.match(pattern, f.name)
        if not match:
            continue
        checkpoint_type, base_name = match.group(1), match.group(2)
        if keyword and keyword not in base_name:
            continue
        checkpoint_files.append({
            "path": f,
            "type": checkpoint_type,
            "base_name": base_name,
            "size_bytes": f.stat().st_size,
        })

    return checkpoint_files


def count_checkpoints(root: Path, keyword: Optional[str] = None, show_all: bool = False) -> Dict:
    """Find and summarize every checkpoint file under `root`."""
    results = {}
    checkpoint_files = find_checkpoint_files(root, keyword)

    if not checkpoint_files:
        print(f"No checkpoint files found under {root}")
        print("   Looking for files matching: checkpoint_*.jsonl")
        return results

    print(f"\n{'=' * 70}")
    print("SECODASH CHECKPOINT COUNTER")
    print(f"{'=' * 70}")
    print(f"Found {len(checkpoint_files)} checkpoint files under {root}")
    print(f"{'=' * 70}\n")

    total_raw = 0
    total_derived = 0
    unique_repos_raw = set()
    unique_repos_derived = set()

    for file_info in sorted(checkpoint_files, key=lambda x: x["base_name"]):
        path = file_info["path"]
        stats = count_jsonl_records(path)

        key = f"{file_info['base_name']}_{file_info['type']}"
        results[key] = {
            "path": str(path),
            "type": file_info["type"],
            "base_name": file_info["base_name"],
            "total": stats["total"],
            "unique_urls": len(stats["unique_urls"]),
            "duplicates": stats["duplicate_count"],
            "platforms": dict(stats["platforms"]),
            "languages": dict(sorted(stats["languages"].items(), key=lambda x: -x[1])[:10]),
            "sample": stats["sample"],
            "errors": stats["errors"],
        }

        if file_info["type"] == "raw":
            total_raw += stats["total"]
            unique_repos_raw.update(stats["unique_urls"])
        else:
            total_derived += stats["total"]
            unique_repos_derived.update(stats["unique_urls"])

        print(f"{path.relative_to(root) if root in path.parents else path.name}")
        print(f"   |-- Type: {file_info['type'].upper()}")
        print(f"   |-- Strategy: {file_info['base_name']}")
        print(f"   |-- Records: {stats['total']:,}")
        print(f"   |-- Unique URLs: {len(stats['unique_urls']):,}")
        print(f"   |-- Duplicates: {stats['duplicate_count']:,}")
        print(f"   |-- File size: {file_info['size_bytes'] / 1024:.1f} KB")
        print(f"   |-- Platforms: {dict(stats['platforms'])}")

        if stats["languages"]:
            top_langs = dict(list(dict(sorted(stats["languages"].items(), key=lambda x: -x[1])).items())[:5])
            print(f"   `-- Top languages: {top_langs}")
        else:
            print("   `-- Languages: No language data")
        print()

        if show_all and stats["sample"]:
            print("   Sample records:")
            for s in stats["sample"][:3]:
                print(f"      - {s['title']} ({s.get('lang', 'unknown')}) - {s['url']}")
            print()

    print(f"{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"Total RAW records:      {total_raw:,}")
    print(f"Unique RAW repos:       {len(unique_repos_raw):,}")
    print(f"Total DERIVED records:  {total_derived:,}")
    print(f"Unique DERIVED repos:   {len(unique_repos_derived):,}")

    if total_raw > 0:
        raw_dup_rate = (total_raw - len(unique_repos_raw)) / total_raw * 100
        derived_dup_rate = (total_derived - len(unique_repos_derived)) / total_derived * 100 if total_derived > 0 else 0
        print(f"RAW duplicate rate:     {raw_dup_rate:.1f}%")
        print(f"DERIVED duplicate rate: {derived_dup_rate:.1f}%")

    print(f"{'=' * 70}\n")

    return results


def main():
    parser = argparse.ArgumentParser(description="Count repositories in SecoDash checkpoint files")
    parser.add_argument("--dir", "-d", type=str, default=None,
                         help="Root directory to search for checkpoints (default: the pipeline's output/ dir)")
    parser.add_argument("--file", "-f", type=str, help="Count a specific checkpoint file")
    parser.add_argument("--keyword", "-k", type=str, help="Filter by keyword/base name")
    parser.add_argument("--all", "-a", action="store_true", help="Show detailed breakdown including samples")
    parser.add_argument("--summary", "-s", action="store_true", help="Show only summary")
    parser.add_argument("--list", "-l", action="store_true", help="List all checkpoint files found")

    args = parser.parse_args()
    root = Path(args.dir) if args.dir else OUTPUT_DIR

    if args.list:
        files = sorted(root.rglob("checkpoint_*.jsonl"))
        if files:
            print(f"Checkpoint files found under {root}:")
            for f in files:
                size = f.stat().st_size / 1024
                print(f"   {f.relative_to(root)} ({size:.1f} KB)")
        else:
            print(f"No checkpoint files found under {root}.")
        return

    if args.file:
        path = Path(args.file)
        if not path.exists():
            print(f"File not found: {path}")
            return

        stats = count_jsonl_records(path)
        print(f"\n{path.name}")
        print(f"{'=' * 60}")
        print(f"Total records:      {stats['total']:,}")
        print(f"Unique URLs:        {len(stats['unique_urls']):,}")
        print(f"Duplicates:         {stats['duplicate_count']:,}")
        if stats['total'] > 0:
            dup_rate = stats['duplicate_count'] / stats['total'] * 100
            print(f"Duplicate rate:     {dup_rate:.1f}%")
        print(f"Platforms:          {dict(stats['platforms'])}")
        if stats['languages']:
            print("Top languages:")
            for lang, count in sorted(stats['languages'].items(), key=lambda x: -x[1])[:5]:
                print(f"   {lang}: {count}")
        if stats['sample']:
            print("\nSample records:")
            for s in stats['sample'][:3]:
                print(f"   {s['title']} - {s.get('url', 'No URL')}")
        return

    if args.summary:
        count_checkpoints(root, args.keyword, show_all=False)
        return

    results = count_checkpoints(root, args.keyword, show_all=args.all)

    if results:
        import csv
        csv_path = reports_dir() / f"checkpoint_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        try:
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Base Name", "Type", "Total", "Unique", "Duplicates",
                                  "Platforms", "Top Language", "Top Language Count"])
                for key, data in results.items():
                    if data['languages']:
                        top_lang, top_count = list(data['languages'].items())[0]
                    else:
                        top_lang, top_count = "None", 0
                    writer.writerow([
                        data['base_name'], data['type'], data['total'], data['unique_urls'], data['duplicates'],
                        str(data['platforms']), top_lang, top_count,
                    ])
            print(f"Summary exported to: {csv_path}")
        except Exception as e:
            print(f"Could not export CSV: {e}")


if __name__ == "__main__":
    main()
