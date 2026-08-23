"""Centralized output directory layout for a SecoDash pipeline run.

Every artifact produced by a run (raw/derived checkpoints, error logs,
audit logs, scored tables, ablation studies, robustness baselines,
knowledge graphs, and AI context payloads) is written under a single
`OUTPUT_DIR` root instead of the current working directory. This keeps
repeated runs for different keywords/strategies from overwriting each
other's files, and makes it trivial to archive or share one run's results.

Layout:

    output/
      <base_name>/                  base_name = "<keyword>_<sampling-strategy>"
        checkpoints/
          checkpoint_raw_<base_name>.jsonl
          checkpoint_derived_<base_name>.jsonl
          checkpoint_errors_<base_name>.jsonl
        audit/
          audit_<base_name>.jsonl         (raw LLM audit trail)
          ai_audit_<base_name>.xlsx       (human-review export)
        scores/
          snapshot_<base_name>.csv        (per-repository scored table)
        ablation/
          ablation_<base_name>.csv
        robustness/
          robustness_baselines_<base_name>.csv
        knowledge_graph/
          kg_<base_name>_pruned.json      (AI-readable S-P-O triples)
          kg_<base_name>_full.gexf        (Gephi-ready graph)
        ai_context/
          ai_context_<base_name>.txt

      <keyword>/
        comparison/
          sampling_comparison_<keyword>.csv   (cross-strategy statistics)

      snapshots/
        <term>/latest/*.json                  (SecoDash-1-style temporal snapshots)
        <term>/snapshots/<timestamp>/*.json

      reports/
        checkpoint_summary_<timestamp>.csv    (ad-hoc checkpoint_counter.py exports)
"""
from pathlib import Path

from config import OUTPUT_DIR


def _ensure(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


class RunPaths:
    """Resolves (and lazily creates) every output path for one pipeline run.

    A "run" is uniquely identified by `base_name` = "<keyword>_<strategy>".
    """

    def __init__(self, base_name: str):
        self.base_name = base_name
        self.run_dir = OUTPUT_DIR / base_name

    # --- raw / derived checkpoints & error log ---------------------------
    @property
    def checkpoints_dir(self) -> Path:
        return _ensure(self.run_dir / "checkpoints")

    @property
    def raw_checkpoint(self) -> Path:
        return self.checkpoints_dir / f"checkpoint_raw_{self.base_name}.jsonl"

    @property
    def derived_checkpoint(self) -> Path:
        return self.checkpoints_dir / f"checkpoint_derived_{self.base_name}.jsonl"

    @property
    def error_log(self) -> Path:
        return self.checkpoints_dir / f"checkpoint_errors_{self.base_name}.jsonl"

    # --- LLM audit trail ---------------------------------------------------
    @property
    def audit_dir(self) -> Path:
        return _ensure(self.run_dir / "audit")

    @property
    def audit_log(self) -> Path:
        return self.audit_dir / f"audit_{self.base_name}.jsonl"

    @property
    def audit_xlsx(self) -> Path:
        return self.audit_dir / f"ai_audit_{self.base_name}.xlsx"

    # --- scoring / ablation / robustness (tabular -> CSV) -----------------
    @property
    def scores_dir(self) -> Path:
        return _ensure(self.run_dir / "scores")

    @property
    def scores_csv(self) -> Path:
        return self.scores_dir / f"snapshot_{self.base_name}.csv"

    @property
    def ablation_dir(self) -> Path:
        return _ensure(self.run_dir / "ablation")

    @property
    def ablation_csv(self) -> Path:
        return self.ablation_dir / f"ablation_{self.base_name}.csv"

    @property
    def robustness_dir(self) -> Path:
        return _ensure(self.run_dir / "robustness")

    @property
    def robustness_csv(self) -> Path:
        return self.robustness_dir / f"robustness_baselines_{self.base_name}.csv"

    # --- knowledge graph (structured -> JSON, visual graph -> GEXF) -------
    @property
    def knowledge_graph_dir(self) -> Path:
        return _ensure(self.run_dir / "knowledge_graph")

    @property
    def kg_json(self) -> Path:
        return self.knowledge_graph_dir / f"kg_{self.base_name}_pruned.json"

    @property
    def kg_gexf(self) -> Path:
        return self.knowledge_graph_dir / f"kg_{self.base_name}_full.gexf"

    # --- AI context payload -------------------------------------------------
    @property
    def ai_context_dir(self) -> Path:
        return _ensure(self.run_dir / "ai_context")

    @property
    def ai_context_txt(self) -> Path:
        return self.ai_context_dir / f"ai_context_{self.base_name}.txt"


def comparison_csv_path(safe_keyword: str) -> Path:
    """Path for the cross-strategy statistical comparison report (keyed by keyword only)."""
    comparison_dir = _ensure(OUTPUT_DIR / safe_keyword / "comparison")
    return comparison_dir / f"sampling_comparison_{safe_keyword}.csv"


def snapshots_root() -> Path:
    """Root directory for SnapshotManager's temporal, per-repository snapshots."""
    return _ensure(OUTPUT_DIR / "snapshots")


def reports_dir() -> Path:
    """Root directory for ad-hoc reports (e.g. checkpoint_counter.py exports)."""
    return _ensure(OUTPUT_DIR / "reports")


def find_checkpoints(keyword_prefix: str = "", checkpoint_type: str = "raw") -> list:
    """Find checkpoint_<type>_<...>.jsonl files anywhere under OUTPUT_DIR.

    Used for cross-strategy deduplication (`--no-overlap`) and by
    checkpoint_counter.py, both of which need to scan every run's
    checkpoints directory rather than a single known path.
    """
    pattern = f"*/checkpoints/checkpoint_{checkpoint_type}_{keyword_prefix}*.jsonl"
    return sorted(OUTPUT_DIR.glob(pattern))
