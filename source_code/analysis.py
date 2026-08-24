"""Sensitivity and robustness analysis for the scoring model.

Three independent checks on how much the scoring weights matter:
  - run_ablation: zero out one scoring term at a time and measure the
    Spearman correlation (with bootstrap CI) between the original and
    ablated ranking, plus how many repos change quadrant as a result.
  - run_weight_perturbation: randomly jitter every weight by +/-20% across
    many iterations and measure how often each repo's quadrant stays stable.
  - run_equal_weights: recompute scores with every term in a category
    weighted equally and measure agreement with the original quadrant.

compare_sampling_strategies then compares two full runs (e.g. "popularity"
vs. "stratified") statistically: score distributions (Mann-Whitney U),
quadrant distributions (chi-square), and repository overlap.
"""
from __future__ import annotations
from copy import deepcopy
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, mannwhitneyu, chi2_contingency

from config import BASELINE_WEIGHTS
from state import SHUTDOWN, logger
from scoring import compute_scores, classify_quadrant


def bootstrap_spearman(x: np.ndarray, y: np.ndarray, rng: np.random.Generator, n_resamples: int = 500) -> Tuple:
    """Spearman correlation between x and y, with a bootstrap 95% confidence interval."""
    valid_idx = ~np.isnan(x) & ~np.isnan(y)
    x_v, y_v = x[valid_idx], y[valid_idx]
    if len(x_v) < 2 or np.std(x_v) == 0 or np.std(y_v) == 0:
        return np.nan, np.nan, np.nan, 0, n_resamples

    obs_rho, _ = spearmanr(x_v, y_v)
    rhos, invalid = [], 0
    idx = np.arange(len(x_v))
    for _ in range(n_resamples):
        samp = rng.choice(idx, size=len(idx), replace=True)
        xs, ys = x_v[samp], y_v[samp]
        if np.std(xs) == 0 or np.std(ys) == 0:
            invalid += 1
            continue
        r, _ = spearmanr(xs, ys)
        rhos.append(r)

    if not rhos:
        return obs_rho, np.nan, np.nan, 0, invalid
    return obs_rho, np.percentile(rhos, 2.5), np.percentile(rhos, 97.5), len(rhos), invalid


def run_ablation(derived_records: List[dict], baseline_df: pd.DataFrame, base_weights: dict, rng: np.random.Generator) -> pd.DataFrame:
    """For each scoring term, zero its weight and measure the ranking/quadrant impact."""
    results = []
    rse_med_base = baseline_df["rse_score"].dropna().median()
    ai_med_base = baseline_df["ai_score"].dropna().median()
    valid_baseline_len = (baseline_df["quadrant"] != "Unknown").sum()

    for category, terms in base_weights.items():
        score_name = f"{category}_score"
        baseline_values = baseline_df[score_name].values

        for term, weight in terms.items():
            ablated_scores = [compute_scores(rec, w_dict=base_weights, overrides={term: 0}) for rec in derived_records]
            ablated_series = np.array([s[score_name] for s in ablated_scores])
            obs_rho, rho_low, rho_high, val_bs, inv_bs = bootstrap_spearman(baseline_values, ablated_series, rng, 500)

            if score_name in ("maturity_score", "sustainability_score"):
                new_rse = [0.6 * s["maturity_score"] + 0.4 * s["sustainability_score"] for s in ablated_scores]
                new_quads = [classify_quadrant(r, a, rse_med_base, ai_med_base) for r, a in zip(new_rse, baseline_df["ai_score"])]
            elif score_name == "ai_score":
                new_ai = [s["ai_score"] for s in ablated_scores]
                new_quads = [classify_quadrant(r, a, rse_med_base, ai_med_base) for r, a in zip(baseline_df["rse_score"], new_ai)]
            else:
                new_quads = list(baseline_df["quadrant"])

            valid_changes = sum(1 for o, n in zip(baseline_df["quadrant"], new_quads) if o != n and o != "Unknown" and n != "Unknown")
            results.append({
                "composite_score": score_name, "term_removed": term, "weight": weight,
                "obs_spearman_rho": round(obs_rho, 4) if not np.isnan(obs_rho) else None,
                "rho_95ci": f"[{round(rho_low, 3)}, {round(rho_high, 3)}]" if not np.isnan(rho_low) else "[NaN, NaN]",
                "valid_bootstraps": val_bs, "quadrant_shift_pct": round(100.0 * valid_changes / max(1, valid_baseline_len), 2),
            })
            logger.debug(f"  Ablated {term}: rho={obs_rho:.3f}")

    return pd.DataFrame(results)


def generate_perturbed_weights(base: dict, rng: np.random.Generator, noise_pct: float = 0.20) -> dict:
    """Return a copy of `base` with every term jittered by +/-noise_pct and renormalized within its category."""
    w = deepcopy(base)
    for cat, terms in w.items():
        total = sum(v * (1.0 + rng.uniform(-noise_pct, noise_pct)) for v in terms.values())
        for k in terms:
            terms[k] = (terms[k] / total) * 100.0

    # The "ai" category sums to 110 (not 100) by design - a GenAI-usage bonus
    # on top of the other AI signals - so renormalize it to 110 to preserve that.
    ai_total = sum(w["ai"].values())
    for k in w["ai"]:
        w["ai"][k] = (w["ai"][k] / ai_total) * 110.0
    return w


def run_weight_perturbation(derived_records: List[dict], baseline_df: pd.DataFrame, rng: np.random.Generator, n_iters: int = 200) -> dict:
    """Monte Carlo test: how often does each repo's quadrant survive random weight jitter?"""
    valid_idx = [i for i, q in enumerate(baseline_df["quadrant"]) if q != "Unknown"]
    stable_count = np.zeros(len(baseline_df))
    rse_med_base = baseline_df["rse_score"].dropna().median()
    ai_med_base = baseline_df["ai_score"].dropna().median()

    for i in range(n_iters):
        if SHUTDOWN.requested:
            break
        pw = generate_perturbed_weights(BASELINE_WEIGHTS, rng)
        iter_scores = [compute_scores(rec, w_dict=pw) for rec in derived_records]
        for j, s in enumerate(iter_scores):
            if baseline_df["quadrant"].iloc[j] == "Unknown":
                continue
            q = classify_quadrant(s["rse_score"], s["ai_score"], rse_med_base, ai_med_base)
            if q == baseline_df["quadrant"].iloc[j]:
                stable_count[j] += 1

    stabilities = (stable_count[valid_idx] / max(1, n_iters)) * 100.0
    return {
        "median_classification_stability_pct": round(np.median(stabilities), 2),
        "std_classification_stability_pct": round(np.std(stabilities), 2),
    }


def run_equal_weights(derived_records: List[dict], baseline_df: pd.DataFrame) -> dict:
    """Recompute quadrants with every term in each category weighted equally; measure agreement with baseline."""
    ew = deepcopy(BASELINE_WEIGHTS)
    for cat, terms in ew.items():
        val = 100.0 / len(terms) if cat != "ai" else 110.0 / len(terms)
        for k in terms:
            terms[k] = val

    iter_scores = [compute_scores(rec, w_dict=ew) for rec in derived_records]
    rse_med_base = baseline_df["rse_score"].dropna().median()
    ai_med_base = baseline_df["ai_score"].dropna().median()
    ew_quads = [classify_quadrant(s["rse_score"], s["ai_score"], rse_med_base, ai_med_base) for s in iter_scores]

    valid_pairs = [(a, b) for a, b in zip(baseline_df["quadrant"], ew_quads) if a != "Unknown"]
    agreement = 100.0 * sum(1 for a, b in valid_pairs if a == b) / max(1, len(valid_pairs))
    return {"equal_weight_agreement_pct": round(agreement, 2)}


def compare_sampling_strategies(dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Statistically compare two or more sampling-strategy runs for the same keyword."""
    strategies = list(dfs.keys())
    if len(strategies) < 2:
        return pd.DataFrame()
    comparison_rows = []
    score_columns = ["popularity_score", "maturity_score", "sustainability_score", "fair_score", "ai_score", "rse_score"]

    # 1. Score distributions (Mann-Whitney U for the two-strategy case).
    for score_col in score_columns:
        row = {"metric": f"{score_col}_distribution", "row_type": "score_distribution"}
        for strategy in strategies:
            values = dfs[strategy][score_col].dropna()
            if len(values) > 0:
                row[f"{strategy}_mean"], row[f"{strategy}_median"], row[f"{strategy}_n"] = round(values.mean(), 3), round(values.median(), 3), len(values)
        if len(strategies) == 2:
            s1, s2 = strategies
            vals1, vals2 = dfs[s1][score_col].dropna(), dfs[s2][score_col].dropna()
            if len(vals1) > 0 and len(vals2) > 0:
                try:
                    stat, p_value = mannwhitneyu(vals1, vals2, alternative='two-sided')
                    row["mannwhitney_p_value"] = round(p_value, 6)
                    row["significant_at_0.05"] = "Yes" if p_value < 0.05 else "No"
                except Exception as e:
                    logger.warning(f"  Mann-Whitney U test failed for '{score_col}': {e}")
        comparison_rows.append(row)

    # 2. Quadrant distributions per strategy.
    for strategy in strategies:
        quadrants = dfs[strategy]["quadrant"].value_counts()
        row = {"metric": f"quadrant_distribution_{strategy}", "row_type": "quadrant_distribution", "strategy": strategy}
        for quad in ["AI4RSE", "RSE", "Vibe", "Exploratory", "Unknown"]:
            row[f"{quad}"] = f"{int(quadrants.get(quad, 0))} ({round(quadrants.get(quad, 0) / len(dfs[strategy]) * 100, 1)}%)"
        comparison_rows.append(row)

    # 3. Chi-square test and repository overlap (two-strategy case only).
    if len(strategies) == 2:
        s1, s2 = strategies
        all_quads = ["AI4RSE", "RSE", "Vibe", "Exploratory", "Unknown"]
        s1_counts = [int(dfs[s1]["quadrant"].value_counts().get(q, 0)) for q in all_quads]
        s2_counts = [int(dfs[s2]["quadrant"].value_counts().get(q, 0)) for q in all_quads]
        contingency_table = np.array([s1_counts, s2_counts])
        non_zero_cols = contingency_table.sum(axis=0) > 0
        if non_zero_cols.sum() > 1:
            try:
                chi2, p_val, dof, _ = chi2_contingency(contingency_table[:, non_zero_cols])
                comparison_rows.append({
                    "metric": "quadrant_chi_square_test", "row_type": "chi_square",
                    "chi2_statistic": round(chi2, 3), "p_value": round(p_val, 6),
                    "significant_at_0.05": "Yes" if p_val < 0.05 else "No",
                    "excluded_quadrants": [q for q, keep in zip(all_quads, non_zero_cols) if not keep],
                })
            except Exception as e:
                logger.warning(f"  Chi-square test failed: {e}")

        urls1, urls2 = set(dfs[s1]["html_url"].dropna()), set(dfs[s2]["html_url"].dropna())
        overlap = urls1 & urls2
        comparison_rows.append({
            "metric": "repository_overlap", "row_type": "overlap",
            f"{s1}_total": len(urls1), f"{s2}_total": len(urls2),
            "overlap_count": len(overlap), "overlap_pct": round(100.0 * len(overlap) / max(1, len(urls1 | urls2)), 1),
        })

    return pd.DataFrame(comparison_rows)
