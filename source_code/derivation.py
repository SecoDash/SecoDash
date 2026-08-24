"""Turns a raw enrichment record into derived boolean/scalar/categorical features.

This is the feature-engineering step between extraction.py (raw API data)
and scoring.py (weighted composite scores): every field scoring.py reads
via `val(...)` is computed here.
"""
from __future__ import annotations
import re
from datetime import datetime, timezone
from typing import List, Optional

from ai_assistant import llm_assistant
from config import AIOPS_KW, MLOPS_KW, LLM_MODEL, README_MIN_CHARS, RECENCY_MONTHS, TAXONOMY_FILE
from extraction import compute_file_hints
from taxonomy import TaxonomyMatcher

taxonomy_matcher = TaxonomyMatcher(TAXONOMY_FILE)


def detect_primary_language(languages: Optional[dict]) -> Optional[str]:
    """Pick the language with the highest byte count from GitHub/GitLab language stats."""
    if not languages:
        return None
    try:
        return max(languages.items(), key=lambda x: x[1])[0]
    except (ValueError, AttributeError):
        return None


def compute_derived(rec: dict) -> dict:
    d = rec.copy()
    rmd = (d.get("raw_readme") or "").lower()
    fh = compute_file_hints(d.get("raw_tree_paths"))
    repo_url = d.get("html_url", "Unknown Repository")

    def _recent(date_str: Optional[str]) -> Optional[int]:
        if not date_str:
            return None
        try:
            dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
            return int((datetime.now(timezone.utc) - dt).days <= RECENCY_MONTHS * 30.44)
        except Exception:
            return None

    def _has_kw(text: str, kw_list: List[str]) -> bool:
        return bool([k for k in kw_list if re.search(rf"\b({re.escape(k)})\b", text)])

    d["primary_language"] = detect_primary_language(d.get("raw_languages"))
    d["has_readme"] = int(len(rmd) >= README_MIN_CHARS) if d.get("raw_readme") is not None else None
    d["has_languages"] = 1 if d.get("raw_languages") else 0
    d["has_recognized_license"] = 0 if d.get("license_key") is None else int(d.get("license_key") != "other")

    # Prefer the explicit indicator from extraction; fall back to the
    # truthiness of raw_packages for records produced before that field existed.
    d["has_packages"] = d.get("raw_packages_indicator", 1 if d.get("raw_packages") else 0)

    d["ci"] = fh.get("has_workflows")
    d["tests"] = fh.get("has_tests")
    d["has_docs_dir"] = fh.get("has_docs_dir")
    d["has_citation_artifact"] = int("doi.org" in rmd or fh.get("has_citation_cff") == 1) if d.get("raw_readme") is not None else None
    d["data_link"] = int(_has_kw(rmd, ["dataset", "zenodo", "figshare"])) if d.get("raw_readme") is not None else None
    d["repro_tools"] = int(fh.get("has_dockerfile") == 1 or _has_kw(rmd, ["docker", "singularity"])) if d.get("raw_readme") is not None else None

    # Older checkpoints stored an integer fallback (1/0) in this field instead
    # of a real list; ignore anything that isn't a list so downstream code can
    # always assume `packages` is either a list or absent.
    raw_pkgs = d.get("raw_packages", [])
    d["packages"] = raw_pkgs if isinstance(raw_pkgs, list) else []

    # Continuous dependency-count measure consumed by scoring.py's "packages"
    # / "fair_packages" composite-score terms (see extraction.py for the
    # manifest parsers - Python, JavaScript, R, Julia, C++, C#/F#, PHP, Perl,
    # Rust, Go, Java, Conda, and Erlang - that populate `d["packages"]` above).
    d["declared_dependency_count"] = len(d["packages"]) if isinstance(d["packages"], list) else None

    d["contributors"] = d.get("raw_contributors", [])

    title_desc = f"{d.get('repo_title', '')} {d.get('description', '')} {' '.join(d.get('topics', []))}"
    d["taxonomy_tags"] = taxonomy_matcher.match(title_desc)
    d["has_taxonomy_match"] = 1 if d["taxonomy_tags"] else 0

    base_aiops = int(_has_kw(rmd, AIOPS_KW)) if d.get("raw_readme") is not None else 0
    base_mlops = int(_has_kw(rmd, MLOPS_KW) or fh.get("has_mlops_tools") == 1) if d.get("raw_readme") is not None else 0

    if base_aiops == 0 and base_mlops == 0 and llm_assistant.is_available:
        # Keyword heuristics found nothing - ask the LLM to classify the
        # repo's domain instead of defaulting straight to "neither".
        gpt_aiops, gpt_mlops = llm_assistant.classify_ops_fallback(
            title_desc + " " + (d.get("raw_readme") or "")[:2000],
            repo_context=repo_url,
        )
        d["aiops_proxy"] = gpt_aiops
        d["mlops_proxy"] = gpt_mlops
        d["used_llm_fallback"] = 1 if (gpt_aiops or gpt_mlops) else 0
    else:
        d["aiops_proxy"] = base_aiops
        d["mlops_proxy"] = base_mlops
        d["used_llm_fallback"] = 0

    d["ai_commit_evidence"] = llm_assistant.classify_commits(d.get("raw_commits_msgs"), repo_context=repo_url)
    d["ai_code_evidence"] = llm_assistant.classify_code(d.get("raw_sample_code"), repo_context=repo_url)
    d["ai_readme_evidence"] = llm_assistant.classify_readme(d.get("raw_readme"), repo_context=repo_url)

    d["genai_detected"] = 1 if (
        d["ai_commit_evidence"] == 1 or
        d["ai_code_evidence"] == 1 or
        d["ai_readme_evidence"] == 1
    ) else 0
    d["genai_model"] = LLM_MODEL if llm_assistant.is_available else "OFFLINE"
    d["pushed_recent"] = _recent(d.get("pushed_at"))

    # Drop the bulky raw fields now that every feature derived from them has
    # been computed - keeps the derived checkpoint compact.
    for k in ["raw_readme", "raw_tree_paths", "raw_sample_code", "raw_commits_msgs", "raw_packages", "raw_contributors"]:
        d.pop(k, None)
    return d
