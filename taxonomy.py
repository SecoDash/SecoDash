"""Loads an IEEE-style taxonomy tree and matches free text against its terms.

The taxonomy file is a nested JSON tree (dicts with "name"/"children",
plain strings, or arbitrary nested dicts/lists); `_flatten` walks all of
these shapes into a flat set of term strings, and `match` scores a query
against every term using token-set Jaccard similarity.
"""
import json
import re
from pathlib import Path
from typing import Any, List, Set

from state import logger
from config import STOP_WORDS


class TaxonomyMatcher:
    def __init__(self, filepath: Path):
        self.ready = False
        self.terms: List[str] = []
        self.tokens: List[Set[str]] = []

        if not filepath.exists():
            logger.warning(f"Taxonomy file missing: {filepath}. Semantic mapping disabled.")
            return

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)

            raw_terms = set()
            self._flatten(data, raw_terms)
            self.terms = sorted([t for t in raw_terms if t.strip()])
            self.tokens = [self._tokenize(t) for t in self.terms]
            self.ready = True
        except Exception as e:
            logger.error(f"Failed to load taxonomy: {e}")

    def _flatten(self, obj: Any, out_set: Set[str]):
        """Recursively collect every "name" value and dict key/string leaf into out_set."""
        if isinstance(obj, dict):
            if "name" in obj:
                out_set.add(str(obj["name"]))
            if "children" in obj:
                self._flatten(obj["children"], out_set)

            # Generic fallback for dicts that don't follow the {name, children} shape.
            if "name" not in obj and "children" not in obj:
                for k, v in obj.items():
                    out_set.add(str(k))
                    self._flatten(v, out_set)
        elif isinstance(obj, list):
            for item in obj:
                self._flatten(item, out_set)
        elif isinstance(obj, str):
            out_set.add(obj)

    def _tokenize(self, text: str) -> Set[str]:
        if not text:
            return set()
        clean = re.sub(r"[^a-z0-9 +.\-]", " ", text.lower())
        return {w for w in clean.split() if w and w not in STOP_WORDS}

    def match(self, text: str, k: int = 5) -> List[str]:
        """Return up to k taxonomy terms ranked by Jaccard similarity to `text`."""
        if not self.ready or not text:
            return []

        q_tokens = self._tokenize(text)
        if not q_tokens:
            return []

        scores = []
        for i, t_tokens in enumerate(self.tokens):
            inter = len(q_tokens & t_tokens)
            if inter:
                union = len(q_tokens) + len(t_tokens) - inter
                scores.append((i, inter / union))

        scores.sort(key=lambda x: x[1], reverse=True)
        return [self.terms[i] for i, score in scores[:k]]
