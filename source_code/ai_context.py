"""Builds a single text payload summarizing an ecosystem, for pasting into an LLM.

Combines aggregate statistics, a minified JSON snapshot of a stratified 
repository sample, and bounded knowledge-graph triples into one plain-text 
file suitable for pasting directly into a chat-based LLM as context.
"""
import json
import os
import csv
from collections import Counter, defaultdict
from pathlib import Path

from state import logger

# Context bounds to prevent blowing the LLM token budget
MAX_REPOS_IN_CONTEXT = 80
MAX_KG_TRIPLES = 500
MAX_PAYLOAD_CHARS = 100000  # ~25k tokens


def generate_ai_context(keyword: str, sampling: str, derived_path: Path, kg_path: Path, output_path: Path, scores_path: Path = None):
    """Build the AI context payload for one run and write it to `output_path`.

    Args:
        keyword: the ecosystem search keyword this run covers.
        sampling: the sampling strategy used ("popularity" / "stratified").
        derived_path: path to checkpoint_derived_<base_name>.jsonl.
        kg_path: path to the pruned knowledge-graph JSON for this run (optional).
        output_path: destination .txt file for the assembled payload.
        scores_path: path to snapshot_<base_name>.csv. If None, it is inferred
                     automatically from the output_path directory structure.
    """
    if not os.path.exists(derived_path):
        logger.error(f"Cannot generate AI context: {derived_path} not found. Run pipeline first.")
        return

    # Infer scores_path automatically to maintain backwards compatibility with main.py
    if scores_path is None:
        base_name = f"{keyword}_{sampling}"
        run_dir = output_path.parent.parent
        scores_path = run_dir / "scores" / f"snapshot_{base_name}.csv"

    # 1. Load derived repos mapped by URL for merging
    repos_by_url = {}
    with open(derived_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                url = rec.get("html_url")
                if url:
                    repos_by_url[url] = rec

    if not repos_by_url:
        logger.error("No derived repositories found.")
        return

    # 2. Merge scores and quadrants from the CSV
    if scores_path and os.path.exists(scores_path):
        with open(scores_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                url = row.get("html_url")
                if url in repos_by_url:
                    repos_by_url[url]["quadrant"] = row.get("quadrant", "Unknown")
                    for score_key in ["popularity_score", "maturity_score", "sustainability_score", "fair_score", "ai_score", "rse_score"]:
                        try:
                            repos_by_url[url][score_key] = float(row.get(score_key, 0.0))
                        except (ValueError, TypeError):
                            repos_by_url[url][score_key] = None
    else:
        logger.warning(f"Scores file {scores_path} not found. AI context will lack scoring and quadrant data.")

    repos = list(repos_by_url.values())
    logger.info(f"Generating AI context prompt for '{keyword}'...")

    # 3. Aggregate stats
    stars_sum = sum(int(r.get("stargaze_count") or 0) for r in repos)
    forks_sum = sum(int(r.get("forks_count") or 0) for r in repos)

    lang_counter = Counter(r.get("primary_language") for r in repos if r.get("primary_language"))
    topic_counter = Counter(t for r in repos for t in r.get("topics", []))
    pkg_counter = Counter(p for r in repos for p in (r.get("packages") if isinstance(r.get("packages"), list) else []))
    tax_counter = Counter(t for r in repos for t in r.get("taxonomy_tags", []))
    quadrant_counter = Counter(r.get("quadrant", "Unknown") for r in repos)

    # 4. Stratified repository selection (avoids purely star-biased context)
    repos_by_quad = defaultdict(list)
    for r in repos:
        repos_by_quad[r.get("quadrant", "Unknown")].append(r)

    # Target an even split across the 4 main quadrants
    target_per_quad = MAX_REPOS_IN_CONTEXT // 4
    selected_repos = []

    for quad in ["AI4RSE", "RSE", "Vibe", "Exploratory"]:
        q_repos = repos_by_quad.get(quad, [])
        if not q_repos:
            continue
        
        # Optimize the selection based on what makes a repo a strong example of its quadrant
        if quad == "AI4RSE":
            q_repos.sort(key=lambda x: (x.get("rse_score") or 0) + (x.get("ai_score") or 0), reverse=True)
        elif quad == "RSE":
            q_repos.sort(key=lambda x: x.get("rse_score") or 0, reverse=True)
        elif quad == "Vibe":
            q_repos.sort(key=lambda x: x.get("ai_score") or 0, reverse=True)
        else: # Exploratory
            q_repos.sort(key=lambda x: int(x.get("stargaze_count") or 0), reverse=True)
        
        selected_repos.extend(q_repos[:target_per_quad])

    # If some quadrants were underrepresented, backfill to MAX_REPOS_IN_CONTEXT using top stars overall
    if len(selected_repos) < MAX_REPOS_IN_CONTEXT:
        remaining = [r for r in repos if r not in selected_repos]
        remaining.sort(key=lambda x: int(x.get("stargaze_count") or 0), reverse=True)
        needed = MAX_REPOS_IN_CONTEXT - len(selected_repos)
        selected_repos.extend(remaining[:needed])

    # Minify the JSON with the scores included
    brief_repos = []
    for r in selected_repos:
        brief_repos.append({
            "repo_title": r.get("repo_title"),
            "quadrant": r.get("quadrant", "Unknown"),
            "scores": {
                "rse": round(r.get("rse_score"), 2) if r.get("rse_score") is not None else None,
                "ai": round(r.get("ai_score"), 2) if r.get("ai_score") is not None else None,
                "popularity": round(r.get("popularity_score"), 2) if r.get("popularity_score") is not None else None,
                "maturity": round(r.get("maturity_score"), 2) if r.get("maturity_score") is not None else None,
                "sustainability": round(r.get("sustainability_score"), 2) if r.get("sustainability_score") is not None else None,
                "fair": round(r.get("fair_score"), 2) if r.get("fair_score") is not None else None,
            },
            "stars": r.get("stargaze_count"),
            "primary_language": r.get("primary_language"),
            "topics": r.get("topics", [])[:10],
            "packages": (r.get("packages") if isinstance(r.get("packages"), list) else [])[:15],
            "taxonomy": r.get("taxonomy_tags", [])[:5],
        })

    # 5. Load and boundedly prioritize knowledge-graph triples
    kg_lines = []
    if kg_path and os.path.exists(kg_path):
        try:
            with open(kg_path, 'r', encoding='utf-8') as f:
                kg_data = json.load(f)
            
            triples = kg_data.get("triples", [])
            selected_repo_names = {r.get("repo_title") for r in selected_repos if r.get("repo_title")}
            
            # Prioritize triples connected to the repositories we actually gave the LLM context about
            high_pri_triples = [t for t in triples if t.get('subject') in selected_repo_names]
            low_pri_triples = [t for t in triples if t.get('subject') not in selected_repo_names]
            
            ordered_triples = high_pri_triples + low_pri_triples
            
            for triple in ordered_triples[:MAX_KG_TRIPLES]:
                kg_lines.append(f"({triple['subject']}) -[{triple['predicate']}]-> ({triple['object']})")
            
            if len(triples) > MAX_KG_TRIPLES:
                kg_lines.append(f"... (Truncated {len(triples) - MAX_KG_TRIPLES} additional triples for context budget)")
        except Exception as e:
            logger.warning(f"Failed to load KG data: {e}")

    # 6. Assemble the payload safely
    def build_payload():
        lines = [
            "=== SECODASH AI CONTEXT PAYLOAD ===",
            f"Ecosystem Keyword: {keyword}",
            f"Sampling Strategy: {sampling}",
            f"Total Repositories Analyzed: {len(repos)}",
            f"Total Stars: {stars_sum}",
            f"Total Forks: {forks_sum}",
            "",
            "--- AGGREGATE FREQUENCIES ---",
            f"Quadrant Distribution: {', '.join(f'{k}({v})' for k, v in quadrant_counter.most_common())}",
            f"Top 10 Languages: {', '.join(f'{k}({v})' for k, v in lang_counter.most_common(10))}",
            f"Top 15 Topics: {', '.join(f'{k}({v})' for k, v in topic_counter.most_common(15))}",
            f"Top 20 Packages: {', '.join(f'{k}({v})' for k, v in pkg_counter.most_common(20))}",
            f"Top IEEE Taxonomy Tags: {', '.join(f'{k}({v})' for k, v in tax_counter.most_common(10))}",
            "",
            "--- STRATIFIED REPOSITORY SAMPLE (JSON) ---",
            "A balanced sample representing all quadrants (AI4RSE, RSE, Vibe, Exploratory) including scores.",
            json.dumps(brief_repos, indent=2),
            "",
        ]

        if kg_lines:
            lines.extend([
                "--- KNOWLEDGE GRAPH (S-P-O TRIPLES) ---",
                "The following relationships map Repositories to their Languages, Topics, Packages, and Contributors:",
            ])
            lines.extend(kg_lines)

        lines.extend([
            "",
            "=== END OF PAYLOAD ===",
            "SYSTEM PROMPT: You are SecoDash, an AI assistant for analyzing software ecosystems.",
            "Use the statistical summaries, the repository JSON, and the Knowledge Graph triples above to answer questions about this ecosystem.",
            "Base your answers strictly on the provided data. Explain trends using the IEEE taxonomy where applicable.",
        ])
        
        return "\n".join(lines)

    payload_text = build_payload()

    # Enforce strict budget limit
    if len(payload_text) > MAX_PAYLOAD_CHARS:
        logger.warning(f"Context payload exceeds limit ({len(payload_text)} > {MAX_PAYLOAD_CHARS}). Truncating safely.")
        sys_prompt_idx = payload_text.rfind("=== END OF PAYLOAD ===")
        sys_prompt = payload_text[sys_prompt_idx:] if sys_prompt_idx != -1 else ""
        
        available_chars = MAX_PAYLOAD_CHARS - len(sys_prompt) - 100
        payload_text = payload_text[:available_chars] + "\n... [CONTENT TRUNCATED DUE TO LENGTH LIMITS]\n\n" + sys_prompt

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(payload_text)

    logger.info(f"AI Context Payload generated successfully: {output_path}")