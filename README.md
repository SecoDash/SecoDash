# SecoDash 

[![DOI](https://zenodo.org/badge/1343847882.svg)](https://doi.org/10.5281/zenodo.22086712)

This tool was developed as part of the paper *"From Repository Data to Ecosystem Intelligence: A Provenance-Aware Framework for Software Ecosystem Analytics"*.

A pipeline for mining, scoring, and analyzing software ecosystems on GitHub
(and optionally GitLab). Give it a keyword (e.g. `"machine learning"`), and
it will:

1. **Discover** matching repositories (star-sorted or star-stratified sampling).
2. **Extract** raw data for each one - README, file tree, languages, contributors,
   commit history, declared dependencies (12+ manifest formats).
3. **Derive** scoring features (readme quality, CI/tests presence, FAIR-ness
   signals, AI/ML-usage signals - with an optional local-LLM fallback for the
   ambiguous cases).
4. **Score** each repository on five weighted axes (popularity, maturity,
   sustainability, FAIR, AI) and classify it into a quadrant.
5. **Analyze** the model itself: per-term ablation, weight-perturbation
   robustness, and equal-weight-agreement baselines.
6. Optionally build a **knowledge graph**, an **LLM context payload**, and
   **temporal snapshots** of every repository record.

## Repository layout

```
SecoDash/
  source_code/       All pipeline code (see module overview below), plus
                     requirements.txt and the taxonomy.json consumed by
                     taxonomy.py. Run main.py from inside this folder.
  extra_materials/   Supplementary artifacts shipped with the paper.
                     ai_audit_rust_popularity.xlsx is an example of the
                     --generate-audit human-review export (a 10% sample of
                     every LLM call) produced by a run on the keyword
                     "rust" with popularity sampling. The LLM calls in
                     this sample were reviewed by a human auditor.
```

## Setup

```bash
cd source_code
pip install -r requirements.txt

export GITHUB_TOKEN="ghp_..."      # optional, but strongly recommended (60 req/hr without it)
export GITLAB_TOKEN="..."          # optional, only needed with --include-gitlab
export LLM_URL="http://localhost:1234/v1/chat/completions"   # optional, OpenAI-compatible endpoint
export LLM_MODEL="gemma-4-e4b-it"                             # optional
```

The AI-assisted classifications (keyword validation fallback, AIOps/MLOps
fallback, GenAI-usage detection) are optional. If no LLM endpoint is reachable
at startup, the pipeline runs fine without them and simply skips those checks.

### Taxonomy file

`taxonomy.py` expects a `taxonomy.json` next to it - a JSON tree of terms
(IEEE taxonomy, or your own) used both for keyword validation and for tagging
each repository. Any of these shapes work:

```json
{"software engineering": {"children": ["testing", "ci/cd", "devops"]}}
```
```json
["machine learning", "natural language processing", "computer vision"]
```

If the file is missing, taxonomy matching is silently disabled (keyword
validation then falls back straight to the LLM check, or accepts anything if
no LLM is configured).

## Running the pipeline

```bash
python main.py --keyword "machine learning" --max-repos 500 --sampling popularity
```

Common options:

| Flag | Effect |
|---|---|
| `--max-repos N` | Target repository count per sampling strategy (default 500) |
| `--sampling popularity stratified` | Run one or more strategies sequentially |
| `--include-gitlab` | Also search GitLab, not just GitHub |
| `--no-overlap` | Guarantee 0% repository overlap across strategies |
| `--compare-sampling` | Run statistical comparison across strategies (needs 2+) |
| `--snapshot` | Save a temporal snapshot of every repository record |
| `--generate-audit` | Export a 10% human-review sample of every LLM call to Excel |
| `--generate-kg` | Build a knowledge graph (JSON triples + Gephi GEXF) |
| `--generate-context` | Build a minified LLM context payload for the ecosystem |
| `--verbose` | Debug-level logging |

The pipeline is **resumable**: re-running the same `--keyword`/`--sampling`
picks up from whatever checkpoints already exist under `output/`, rather than
starting over. Interrupting with Ctrl+C triggers a graceful shutdown - the
current checkpoint is flushed before exit.

## Output layout

Every artifact lands under `output/`, organized so each experiment (one
keyword + one sampling strategy) gets its own self-contained folder, and
tabular results are written as CSV while structured/graph results are
written as JSON:

```
output/
  <keyword>_<strategy>/                    e.g. machine_learning_popularity/
    checkpoints/
      checkpoint_raw_<base_name>.jsonl       raw per-repository records (resumable)
      checkpoint_derived_<base_name>.jsonl   scoring-ready derived records (resumable)
      checkpoint_errors_<base_name>.jsonl    per-repository failures, with tracebacks
    audit/
      audit_<base_name>.jsonl                raw LLM prompt/response audit trail
      ai_audit_<base_name>.xlsx              human-review export (--generate-audit)
    scores/
      snapshot_<base_name>.csv               final per-repository scored table
    ablation/
      ablation_<base_name>.csv               per-term sensitivity analysis
    robustness/
      robustness_baselines_<base_name>.csv   weight-perturbation & equal-weight baselines
    knowledge_graph/                         (--generate-kg)
      kg_<base_name>_pruned.json             AI-readable subject-predicate-object triples
      kg_<base_name>_full.gexf               full graph, ready to open in Gephi
    ai_context/                              (--generate-context)
      ai_context_<base_name>.txt             plain-text payload for pasting into an LLM

  <keyword>/
    comparison/
      sampling_comparison_<keyword>.csv      cross-strategy statistics (--compare-sampling)

  snapshots/                                  (--snapshot)
    <term>/latest/<repo>.json                 most recent version of every record
    <term>/snapshots/<timestamp>/<repo>.json  archived versions (kept: last 20)

  reports/
    checkpoint_summary_<timestamp>.csv        ad-hoc exports from counter.py
```

`output_paths.py` is the single source of truth for this layout - every
module that writes a file imports its path helpers from there rather than
constructing paths itself.

## Module overview

| File | Responsibility |
|---|---|
| `main.py` | CLI entry point and pipeline orchestration (5 phases per strategy) |
| `config.py` | All tunables, credentials (env-var only), and scoring weights |
| `output_paths.py` | Central output-directory layout used by every writer |
| `discovery.py` | GitHub/GitLab repository search & sampling |
| `extraction.py` | Raw data + dependency-manifest extraction per repository |
| `derivation.py` | Raw record → scoring features |
| `scoring.py` | Feature normalization, weighted composite scores, quadrant classification |
| `analysis.py` | Ablation, weight-perturbation, equal-weight, and cross-strategy comparisons |
| `ai_assistant.py` | Optional local-LLM classification calls + audit logging |
| `taxonomy.py` | Keyword/topic matching against a taxonomy tree |
| `knowledge_graph.py` | Repository knowledge graph (JSON triples + Gephi GEXF) |
| `ai_context.py` | Minified LLM context payload for an ecosystem |
| `snapshot_manager.py` | Atomic, deduplicated temporal snapshots per repository |
| `http_client.py` | Retrying/rate-limit-aware HTTP client for both platforms |
| `state.py` | Shared logger and graceful-shutdown flag |
