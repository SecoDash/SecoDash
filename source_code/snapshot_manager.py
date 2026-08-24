"""SecoDash 1-style temporal snapshotting: one atomic "latest" file per
repository, plus a dated history of every version that ever changed.

Snapshots are stored under `output/snapshots/<term>/`:
    latest/<repo_key>.json                    always the most recent version
    snapshots/<timestamp>/<repo_key>.json      one copy per version that changed

A new snapshot copy is only written when the record actually differs from
the last one saved, so re-running the pipeline on unchanged repositories
doesn't grow the history directory.
"""
import json
import hashlib
import tempfile
import os
from datetime import datetime
from itertools import islice
from pathlib import Path
from typing import Optional

from config import MAX_SNAPSHOTS
from output_paths import snapshots_root
from state import logger


class SnapshotManager:
    """Implements SecoDash 1's temporal snapshot architecture."""

    @staticmethod
    def _safe_term(term: str) -> str:
        term = (term or "").strip().lower()
        return "".join(c if c.isalnum() or c in "._-" else "-" for c in term)[:40] or "term"

    @staticmethod
    def _repo_key(item: dict) -> str:
        base = item.get("repo_title") or item.get("full_name") or item.get("html_url") or "repo"
        h = hashlib.sha1(base.encode("utf-8")).hexdigest()[:12]
        safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in base).replace("/", "__")
        return f"{safe[:60]}__{h}"

    @staticmethod
    def _atomic_write_text(path: Path, text: str):
        """Write via a temp file + os.replace so a crash mid-write never leaves a truncated file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile("w", delete=False, dir=str(path.parent), encoding="utf-8") as tmp:
            tmp.write(text)
            tmp.flush()
            os.fsync(tmp.fileno())
            tmp_name = tmp.name
        os.replace(tmp_name, path)

    @staticmethod
    def _read_text_or_none(p: Path) -> Optional[str]:
        try:
            return p.read_text(encoding="utf-8")
        except Exception:
            return None

    @staticmethod
    def _prune_snapshots(term_dir: Path):
        """Keep only the MAX_SNAPSHOTS most recent dated snapshot folders."""
        snaps_root = term_dir / "snapshots"
        if not snaps_root.exists():
            return
        snaps = sorted([p for p in snaps_root.iterdir() if p.is_dir()], reverse=True)
        for old in islice(snaps, MAX_SNAPSHOTS, None):
            for f in old.glob("*"):
                f.unlink(missing_ok=True)
            try:
                old.rmdir()
            except Exception:
                pass

    @classmethod
    def save_snapshot(cls, term: str, item: dict):
        """Save `item` as the latest version for `term`, and archive it if it changed."""
        safe_term = cls._safe_term(term)
        term_dir = snapshots_root() / safe_term
        latest_dir = term_dir / "latest"

        stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        snap_dir = term_dir / "snapshots" / stamp

        latest_dir.mkdir(parents=True, exist_ok=True)
        key = cls._repo_key(item)

        item_copy = item.copy()
        item_copy["searchTerm"] = term

        body = json.dumps(item_copy, ensure_ascii=False, indent=2)
        latest_path = latest_dir / f"{key}.json"

        prev = cls._read_text_or_none(latest_path)
        cls._atomic_write_text(latest_path, body)

        if prev != body:
            snap_dir.mkdir(parents=True, exist_ok=True)
            cls._atomic_write_text(snap_dir / f"{key}.json", body)

        cls._prune_snapshots(term_dir)
