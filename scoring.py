"""Scoring and quadrant-classification engine.

Every raw/derived signal is normalized to [0, 1] and combined into five
weighted composite scores (popularity, maturity, sustainability, fair,
ai), plus an overall "rse_score" (0.6 * maturity + 0.4 * sustainability).
Repositories are then split into four quadrants by comparing rse_score
and ai_score against their sample medians.
"""
from __future__ import annotations
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config import BASELINE_WEIGHTS, LANG_MAX_DEPS


def weighted_score(components: List[Tuple[float, float]]) -> float:
    """Weighted sum over (weight, value) pairs, renormalized to ignore NaN values.

    Renormalizing (rather than treating a NaN as 0) means a repository
    missing one signal isn't unfairly penalized relative to one where that
    signal is genuinely absent (0).
    """
    total_weight = sum(w for w, _ in components)
    avail = [(w, v) for w, v in components if not (isinstance(v, float) and math.isnan(v))]
    if not avail:
        return np.nan
    avail_weight = sum(w for w, _ in avail)
    if avail_weight == 0:
        return np.nan
    raw_sum = sum(w * v for w, v in avail)
    return raw_sum * (total_weight / avail_weight)


def compute_scores(rec: dict, w_dict: dict = BASELINE_WEIGHTS, overrides: Optional[dict] = None) -> Dict[str, float]:
    """Compute all five composite scores plus rse_score for one derived record.

    `overrides` lets a specific normalized term be forced to a fixed value
    (used by analysis.run_ablation to zero out one term at a time and
    measure its effect on the resulting ranking).
    """
    ov = overrides or {}

    def val(key: str) -> float:
        v = rec.get(key)
        return float(v) if v is not None else np.nan

    def extract(term_name: str, calc_val: float) -> float:
        return float(ov[term_name]) if term_name in ov else calc_val

    s_raw = val("stargaze_count")
    f_raw = val("forks_count")
    w_raw = val("watchers_count")

    # Log-scaled normalizations, each capped at 1.0 once a term reaches its
    # reference ceiling (5000 stars / 1000 forks / 500 watchers).
    v_stars = extract("stars", np.nan if np.isnan(s_raw) else max(0.0, min(1.0, math.log1p(s_raw) / math.log1p(5000))))
    v_forks = extract("forks", np.nan if np.isnan(f_raw) else max(0.0, min(1.0, math.log1p(f_raw) / math.log1p(1000))))
    v_watch = extract("watchers", np.nan if np.isnan(w_raw) else max(0.0, min(1.0, math.log1p(w_raw) / math.log1p(500))))

    # Dependency breadth: log-scaled count of declared dependencies
    # (derivation.py populates `declared_dependency_count`), normalized
    # against a language-specific expected maximum (LANG_MAX_DEPS) so a JS
    # repo's typically larger dependency count isn't compared against the
    # same denominator as a Rust repo's typically smaller one.
    lang = str(rec.get("primary_language", "default")).lower()
    lang_max = LANG_MAX_DEPS.get(lang, LANG_MAX_DEPS["default"])
    dep_count = val("declared_dependency_count")
    calc_pkg = np.nan if np.isnan(dep_count) else max(0.0, min(1.0, math.log1p(dep_count) / math.log1p(lang_max)))
    v_pkg = extract("packages", calc_pkg)

    # Binary/derived normalizations.
    v_cit = extract("citation_bonus", val("has_citation_artifact"))
    v_readme = extract("readme", val("has_readme"))
    v_lic = extract("license", val("has_recognized_license"))
    v_branch = extract("branches", np.nan if np.isnan(val("raw_branches_count")) else float(val("raw_branches_count") > 1))
    v_ci = extract("ci", val("ci"))
    v_tests = extract("tests", val("tests"))

    c_raw = val("raw_contributors_count")
    v_contrib = extract("contributors", np.nan if np.isnan(c_raw) else max(0.0, min(1.0, c_raw / 50.0)))

    v_upd = extract("recent_update", val("pushed_recent"))
    cv_raw = val("raw_commits_total")
    v_commit = extract("commit_volume", np.nan if np.isnan(cv_raw) else float(cv_raw >= 20))
    v_repro = extract("fair_repro", val("repro_tools"))
    v_docs = extract("fair_docs", val("has_docs_dir"))

    # AI-specific terms.
    v_mlops = extract("ai_mlops", val("mlops_proxy"))
    v_aiops = extract("ai_aiops", val("aiops_proxy"))
    v_data = extract("ai_data", val("data_link"))
    v_genai = extract("ai_genai", val("genai_detected"))

    wp, wm, ws, wf, wa = w_dict["popularity"], w_dict["maturity"], w_dict["sustainability"], w_dict["fair"], w_dict["ai"]

    score_pop = weighted_score([(wp["stars"], v_stars), (wp["forks"], v_forks),
                                 (wp["watchers"], v_watch), (wp["citation_bonus"], v_cit)])
    score_mat = weighted_score([(wm["readme"], v_readme), (wm["license"], v_lic), (wm["packages"], v_pkg),
                                 (wm["branches"], v_branch), (wm["ci"], v_ci), (wm["tests"], v_tests)])
    score_sus = weighted_score([(ws["contributors"], v_contrib), (ws["recent_update"], v_upd),
                                 (ws["commit_volume"], v_commit)])
    score_fair = weighted_score([(wf["fair_license"], v_lic), (wf["fair_citation"], v_cit), (wf["fair_packages"], v_pkg),
                                  (wf["fair_repro"], v_repro), (wf["fair_docs"], v_docs)])
    score_ai = weighted_score([(wa["ai_mlops"], v_mlops), (wa["ai_aiops"], v_aiops), (wa["ai_data"], v_data),
                                (wa["ai_citation"], v_cit), (wa["ai_genai"], v_genai)])

    score_rse = weighted_score([(0.6, score_mat), (0.4, score_sus)])

    return {
        "popularity_score": score_pop,
        "maturity_score": score_mat,
        "sustainability_score": score_sus,
        "fair_score": score_fair,
        "ai_score": score_ai,
        "rse_score": score_rse,
    }


def classify_quadrant(rse: float, ai: float, rse_median: float, ai_median: float) -> str:
    """Classify a repository into one of four quadrants using strict inequality against the medians."""
    if pd.isna(rse) or pd.isna(ai):
        return "Unknown"

    high_rse = rse > rse_median
    high_ai = ai > ai_median

    if high_rse and high_ai:
        return "AI4RSE"
    elif high_rse and not high_ai:
        return "RSE"
    elif not high_rse and high_ai:
        return "Vibe"
    return "Exploratory"
