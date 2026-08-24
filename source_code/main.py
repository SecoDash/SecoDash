"""SecoDash 2 pipeline entry point.

Orchestrates one full run per requested sampling strategy:
    1. Discovery & extraction  - find repositories, pull raw data.
    2. Derivation              - compute scoring features.
    3. Scoring & outputs       - composite scores, quadrant table, snapshots,
                                  knowledge graph, AI context payload.
    4. Ablation                - per-term sensitivity analysis.
    5. Robustness              - weight-perturbation & equal-weight baselines.

Every artifact is written under `output/<keyword>_<strategy>/...` (see
output_paths.py). Checkpoints are resumable: re-running with the same
keyword and strategy picks up from whatever was already written.

Usage:
    python main.py --keyword "machine learning" --max-repos 500 --sampling popularity
See README.md for the full CLI reference.
"""
import argparse
import concurrent.futures
import json
import re
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from config import RANDOM_SEED, TAXONOMY_FILE, MAX_THREADS, LLM_MAX_WORKERS, BASELINE_WEIGHTS, GITHUB_TOKEN, OUTPUT_DIR
from output_paths import RunPaths, comparison_csv_path, find_checkpoints
from state import SHUTDOWN, configure_logging, install_signal_handlers, logger

from discovery import search_github_repos, search_gitlab_repos
from extraction import enrich_raw_github, enrich_raw_gitlab
from derivation import compute_derived
from scoring import compute_scores, classify_quadrant
from snapshot_manager import SnapshotManager
from ai_assistant import llm_assistant
from taxonomy import TaxonomyMatcher
from knowledge_graph import build_knowledge_graph
from ai_context import generate_ai_context
from analysis import run_ablation, run_weight_perturbation, run_equal_weights, compare_sampling_strategies

taxonomy_matcher = TaxonomyMatcher(TAXONOMY_FILE)

# Monte Carlo iterations for the weight-perturbation robustness check. Kept
# as a named constant (rather than relying on run_weight_perturbation's
# default) so the value written to the robustness-baselines CSV always
# matches the value actually used, even if that default changes later.
WEIGHT_PERTURBATION_ITERS = 200


def validate_keyword(keyword: str) -> bool:
    """Reject disallowed keywords outright, then require either a taxonomy match or an LLM sanity check."""
    block_re = re.compile(r"\b(porn|nsfw|xxx|adult)\b", re.I)
    if block_re.search(keyword):
        logger.error("Keyword blocked by safety policy.")
        return False

    q_tokens = taxonomy_matcher._tokenize(keyword)
    best_sim = 0.0
    if q_tokens and taxonomy_matcher.tokens:
        for t_tokens in taxonomy_matcher.tokens:
            inter = len(q_tokens & t_tokens)
            if inter:
                union = len(q_tokens) + len(t_tokens) - inter
                best_sim = max(best_sim, inter / union)

    if best_sim >= 0.28:
        logger.info(f"Keyword validated via IEEE Taxonomy (score: {best_sim:.3f}).")
        return True

    logger.info("Taxonomy match failed. Falling back to AI classification...")
    return llm_assistant.classify_keyword(keyword)


class PipelineRunner:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.rng = np.random.default_rng(RANDOM_SEED)
        self.global_seen_urls: Set[str] = set()
        self.checkpoint_lock = threading.Lock()

    def _append_jsonl(self, path: Path, data: dict):
        with self.checkpoint_lock:
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(data) + "\n")

    def _load_jsonl(self, path: Path) -> List[dict]:
        records = []
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        try:
                            records.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass
        return records

    def _load_and_deduplicate(self, path: Path, global_exclude: Optional[Set[str]] = None) -> List[dict]:
        """Load a checkpoint, dropping intra-file duplicates and (optionally) URLs seen in other strategies."""
        if not path.exists():
            return []

        records = []
        seen_local = set()
        duplicates_found = False

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                    url = rec.get("html_url")

                    if not url:
                        records.append(rec)
                        continue

                    if url in seen_local:
                        duplicates_found = True
                        continue

                    # Under --no-overlap, also drop URLs already claimed by another strategy.
                    if global_exclude and url in global_exclude:
                        duplicates_found = True
                        continue

                    seen_local.add(url)
                    records.append(rec)
                except json.JSONDecodeError:
                    pass

        if duplicates_found:
            logger.info(f"Sanitizing {path.name}: removing duplicates/overlaps. Rewriting clean checkpoint.")
            with self.checkpoint_lock:
                temp_path = path.with_suffix(".tmp")
                with open(temp_path, "w", encoding="utf-8") as f:
                    for rec in records:
                        f.write(json.dumps(rec) + "\n")
                temp_path.replace(path)

        return records

    def log_error(self, stage: str, url: Optional[str], exc: Exception, paths: RunPaths):
        entry = {
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Stage": stage,
            "URL": url,
            "Error": str(exc),
            "Traceback": traceback.format_exc(),
        }
        logger.error(f"  [{stage}] Failed on {url}: {exc}")
        self._append_jsonl(paths.error_log, entry)

    def run_strategy(self, strategy: str) -> Optional[pd.DataFrame]:
        start_time = time.time()
        logger.info(f"\n{'=' * 60}\nRunning Strategy: {strategy.upper()}\n{'=' * 60}")

        safe_kw = re.sub(r'[^a-zA-Z0-9_]', '_', self.args.keyword.lower())
        base_name = f"{safe_kw}_{strategy}"
        paths = RunPaths(base_name)

        # --- pre-run audit sync -------------------------------------------
        if self.args.generate_audit and llm_assistant.audit_log:
            existing_audit = self._load_jsonl(paths.audit_log)
            existing_prompts = {e.get("Prompt") for e in existing_audit}
            for entry in llm_assistant.audit_log:
                if entry.get("Prompt") not in existing_prompts:
                    self._append_jsonl(paths.audit_log, entry)

        # --- pre-populate exclusions & deduplicate existing checkpoints ---
        other_seen = set()
        if self.args.no_overlap:
            other_seen.update(self.global_seen_urls)
            # Scan every other strategy's raw checkpoint for this keyword under OUTPUT_DIR.
            for p in find_checkpoints(keyword_prefix=safe_kw, checkpoint_type="raw"):
                if strategy not in p.name:
                    with open(p, "r", encoding="utf-8") as f:
                        for line in f:
                            if line.strip():
                                try:
                                    u = json.loads(line).get("html_url")
                                    if u:
                                        other_seen.add(u)
                                except json.JSONDecodeError:
                                    pass

        raw_records = self._load_and_deduplicate(paths.raw_checkpoint, global_exclude=other_seen)
        derived_records = self._load_and_deduplicate(paths.derived_checkpoint, global_exclude=other_seen)

        exclude = set(other_seen) if self.args.no_overlap else set()
        for r in raw_records:
            u = r.get("html_url")
            if u:
                exclude.add(u)
                if self.args.no_overlap:
                    self.global_seen_urls.add(u)

        # --- PHASE 1: discovery & extraction --------------------------------
        logger.info(f"\n{'=' * 60}\nPHASE 1: Raw Data Extraction\n{'=' * 60}")
        stall_rounds = 0

        while len(raw_records) < self.args.max_repos and not SHUTDOWN.requested:
            needed = self.args.max_repos - len(raw_records)
            count_before = len(raw_records)
            logger.info(f"Need {needed} more repos (currently have {len(raw_records)})")

            items = [("github", i) for i in search_github_repos(self.args.keyword, needed, strategy, self.rng, exclude)]
            if self.args.include_gitlab:
                items.extend([("gitlab", i) for i in search_gitlab_repos(self.args.keyword, needed, self.rng, exclude)])

            if not items:
                logger.warning("No more repos found. Stopping search.")
                break

            def process_raw(task: Tuple[str, dict]) -> Optional[dict]:
                platform, item = task
                if SHUTDOWN.requested:
                    return None
                try:
                    return enrich_raw_github(item) if platform == "github" else enrich_raw_gitlab(item)
                except Exception as e:
                    self.log_error("extraction", item.get('html_url'), e, paths)
                    return None

            with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
                futures = [executor.submit(process_raw, t) for t in items]
                for future in concurrent.futures.as_completed(futures):
                    if SHUTDOWN.requested:
                        break
                    try:
                        res = future.result()
                    except Exception as e:
                        self.log_error("extraction_unhandled", None, e, paths)
                        res = None
                    if res:
                        self._append_jsonl(paths.raw_checkpoint, res)
                        raw_records.append(res)
                        if len(raw_records) >= self.args.max_repos:
                            executor.shutdown(wait=False, cancel_futures=True)
                            break

            if len(raw_records) == count_before:
                stall_rounds += 1
                if stall_rounds >= 5:
                    logger.warning("Stopping: 5 consecutive passes with no progress.")
                    break
            else:
                stall_rounds = 0

        if SHUTDOWN.requested:
            return None

        # --- PHASE 2: derivation --------------------------------------------
        logger.info(f"\n{'=' * 60}\nPHASE 2: Feature Derivation\n{'=' * 60}")
        derived_urls = {r.get("html_url") for r in derived_records if r.get("html_url")}
        raw_to_derive = [r for r in raw_records if r.get("html_url") not in derived_urls]

        if self.args.generate_audit:
            all_urls = sorted([r.get("html_url") for r in raw_records if r.get("html_url")])
            sample_size = max(1, round(0.10 * len(all_urls))) if all_urls else 0
            llm_assistant.audit_repos = set(self.rng.choice(all_urls, size=sample_size, replace=False)) if sample_size else set()
            llm_assistant.audit_sink = lambda entry: self._append_jsonl(paths.audit_log, entry)

        def process_derived(raw_rec: dict) -> Optional[dict]:
            if SHUTDOWN.requested:
                return None
            try:
                return compute_derived(raw_rec)
            except Exception as e:
                self.log_error("derivation", raw_rec.get('html_url'), e, paths)
                return None

        with concurrent.futures.ThreadPoolExecutor(max_workers=LLM_MAX_WORKERS) as executor:
            futures = [executor.submit(process_derived, r) for r in raw_to_derive]
            for i, future in enumerate(concurrent.futures.as_completed(futures)):
                if SHUTDOWN.requested:
                    break
                try:
                    res = future.result()
                except Exception as e:
                    self.log_error("derivation_unhandled", None, e, paths)
                    res = None
                if res:
                    self._append_jsonl(paths.derived_checkpoint, res)
                    derived_records.append(res)

        if SHUTDOWN.requested:
            return None

        # --- PHASE 3: scoring & outputs --------------------------------------
        logger.info(f"\n{'=' * 60}\nPHASE 3: Scoring & Classification\n{'=' * 60}")
        df = pd.DataFrame([{**r, **compute_scores(r)} for r in derived_records])

        if not df.empty:
            rse_med, ai_med = df["rse_score"].dropna().median(), df["ai_score"].dropna().median()
            df["quadrant"] = df.apply(lambda r: classify_quadrant(r["rse_score"], r["ai_score"], rse_med, ai_med), axis=1)
            df["sampling_strategy"] = strategy
            df.to_csv(paths.scores_csv, index=False)
            logger.info(f"Saved tabular scores to {paths.scores_csv}")

        if self.args.snapshot:
            for r in derived_records:
                SnapshotManager.save_snapshot(self.args.keyword, r)
            logger.info("SD1 Time-series Snapshots saved.")
        if self.args.generate_kg:
            build_knowledge_graph(str(paths.derived_checkpoint), paths.kg_json, paths.kg_gexf)
        if self.args.generate_context:
            generate_ai_context(self.args.keyword, strategy, paths.derived_checkpoint, paths.kg_json, paths.ai_context_txt)

        # --- PHASE 4 & 5: ablation & robustness --------------------------------
        logger.info(f"\n{'=' * 60}\nPHASE 4: Sensitivity Analysis (Ablation)\n{'=' * 60}")
        ablation_df = run_ablation(derived_records, df, BASELINE_WEIGHTS, self.rng)
        ablation_df["sampling_strategy"] = strategy
        ablation_df.to_csv(paths.ablation_csv, index=False)
        logger.info(f"Ablation saved: {paths.ablation_csv}")

        logger.info(f"\n{'=' * 60}\nPHASE 5: Robustness Baselines\n{'=' * 60}")
        pert_res = run_weight_perturbation(derived_records, df, self.rng, n_iters=WEIGHT_PERTURBATION_ITERS)
        ew_res = run_equal_weights(derived_records, df)
        logger.info(f"Weight Perturbation Stability: {pert_res['median_classification_stability_pct']}% +/- {pert_res['std_classification_stability_pct']}%")
        logger.info(f"Equal-Weight Agreement: {ew_res['equal_weight_agreement_pct']}%")

        robustness_df = pd.DataFrame([{
            "ecosystem_keyword": self.args.keyword,
            "sampling_strategy": strategy,
            "n_repositories": len(derived_records),
            "weight_perturbation_iterations": WEIGHT_PERTURBATION_ITERS,
            "weight_perturbation_noise_pct": 20.0,
            "weight_perturbation_median_classification_stability_pct": pert_res["median_classification_stability_pct"],
            "weight_perturbation_std_classification_stability_pct": pert_res["std_classification_stability_pct"],
            "equal_weight_agreement_pct": ew_res["equal_weight_agreement_pct"],
        }])
        robustness_df.to_csv(paths.robustness_csv, index=False)
        logger.info(f"Robustness baselines saved: {paths.robustness_csv}")

        # --- export AI audit log from checkpoint (V1 parity) --------------------
        if self.args.generate_audit:
            full_audit = self._load_jsonl(paths.audit_log)
            if full_audit:
                llm_assistant.export_audit_log_to_excel(str(paths.audit_xlsx), records=full_audit)
                logger.info(f"AI human-verification audit log saved to: {paths.audit_xlsx} ({len(full_audit)} samples)")

        logger.info(f"Strategy {strategy.upper()} completed in {time.time() - start_time:.1f}s.")
        return df

    def execute(self):
        if not validate_keyword(self.args.keyword):
            logger.error("Keyword classified as unrelated. Aborting Pipeline.")
            return

        results = {}
        for strategy in self.args.sampling:
            if SHUTDOWN.requested:
                break
            df = self.run_strategy(strategy)

            if df is not None and not df.empty:
                results[strategy] = df
            if self.args.no_overlap and df is not None:
                self.global_seen_urls.update(df["html_url"].dropna())

        if self.args.compare_sampling and len(results) > 1:
            logger.info(f"\n{'=' * 60}\nRunning Statistical Sampling Comparison\n{'=' * 60}")
            comp_df = compare_sampling_strategies(results)
            if not comp_df.empty:
                safe_kw = re.sub(r'[^a-zA-Z0-9_]', '_', self.args.keyword.lower())
                comp_file = comparison_csv_path(safe_kw)
                comp_df.to_csv(comp_file, index=False)
                logger.info(f"Comparison metrics saved to {comp_file}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SecoDash 2 - software ecosystem mining & scoring pipeline")
    parser.add_argument("--keyword", required=True, help="Ecosystem search keyword")
    parser.add_argument("--max-repos", type=int, default=500, help="Max repos per sampling strategy")
    parser.add_argument("--include-gitlab", action="store_true", help="Include GitLab repos in discovery")

    parser.add_argument("--sampling", nargs='+', choices=["popularity", "stratified"], default=["popularity"],
                         help="Array of sampling strategies to execute sequentially.")
    parser.add_argument("--no-overlap", action="store_true",
                         help="Guarantee 0%% repository overlap between multiple samplings.")
    parser.add_argument("--compare-sampling", action="store_true",
                         help="Run Chi-Square/Mann-Whitney comparisons if multiple samplings are provided.")
    parser.add_argument("--snapshot", action="store_true",
                         help="Enable SD1 atomic time-series snapshotting.")
    parser.add_argument("--generate-audit", action="store_true",
                         help="Enable human-readable LLM Audit Log exporting (10%% exact sample).")
    parser.add_argument("--generate-kg", action="store_true",
                         help="Generate GraphML (Gephi) and pruned JSON graphs.")
    parser.add_argument("--generate-context", action="store_true",
                         help="Generate optimized SD1-style context for LLM pasting.")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    return parser


def main():
    install_signal_handlers()
    args = build_arg_parser().parse_args()
    configure_logging(args.verbose)

    if not GITHUB_TOKEN:
        logger.warning("No GITHUB_TOKEN set. Rate limits will be restrictive (60 req/hr).")

    logger.info(f"All output will be written under: {OUTPUT_DIR}")

    llm_assistant.generate_audit = args.generate_audit

    runner = PipelineRunner(args)
    runner.execute()


if __name__ == "__main__":
    main()
