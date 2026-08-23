"""LLM-backed classification helpers, with a human-review audit trail.

Talks to an OpenAI-compatible chat-completions endpoint (LM Studio,
Ollama, vLLM, etc. via `LLM_URL`/`LLM_MODEL`). Every classification call
degrades gracefully to a safe default when the endpoint is unavailable,
so the rest of the pipeline never has to special-case "no LLM configured".

When `generate_audit` is enabled, every prompt/response pair that
contributes to a scored feature is logged and can be exported to Excel
(or CSV, if openpyxl isn't installed) for human spot-checking.
"""
import json
import threading
import time
from datetime import datetime
from typing import List, Optional, Tuple, Callable
import requests
import pandas as pd

from config import LLM_URL, LLM_MODEL, MAX_RETRIES, REQUEST_TIMEOUT
from state import SHUTDOWN, logger


class AIAssistant:
    """Wraps LLM requests, domain-classification fallbacks, GenAI-usage detection, and audit logging."""

    def __init__(self):
        self.url = LLM_URL
        self.model = LLM_MODEL
        self.is_available = self._check_health()

        self.audit_log: List[dict] = []
        self.audit_lock = threading.Lock()
        self.audit_sink: Optional[Callable[[dict], None]] = None
        self.audit_repos: set = set()
        self.generate_audit = False  # Toggled via the --generate-audit CLI flag.

    def _check_health(self) -> bool:
        try:
            resp = requests.get(self.url.replace("/chat/completions", "/models"), timeout=5)
            return resp.status_code == 200
        except Exception:
            return False

    def _ask(self, prompt: str, system_prompt: str = "", max_tokens: int = 16384) -> Optional[str]:
        """Send one chat-completion request with retry/back-off. Returns None on any failure."""
        if not self.is_available:
            return None

        for attempt in range(MAX_RETRIES):
            if SHUTDOWN.requested:
                return None
            try:
                payload = {
                    "model": self.model,
                    "messages": [],
                    "temperature": 0.0,
                    "max_tokens": max_tokens,
                    "stream": False,
                }

                if system_prompt:
                    payload["messages"].append({"role": "system", "content": system_prompt})

                payload["messages"].append({"role": "user", "content": prompt})

                resp = requests.post(self.url, json=payload, timeout=60)

                if resp.status_code == 200:
                    try:
                        response_data = resp.json()
                        if "choices" in response_data and len(response_data["choices"]) > 0:
                            content = response_data["choices"][0]["message"]["content"]
                            if content is not None:
                                return content.strip()
                        logger.warning(f"[AI] Unexpected response structure: {response_data}")
                        return None
                    except ValueError as json_err:
                        logger.warning(f"[AI] Failed to parse JSON response: {json_err}")
                        return None
                else:
                    logger.warning(f"[AI] Server rejected request ({resp.status_code}): {resp.text[:200]}")
                    # Only retry on server errors / rate limiting; 4xx client errors are not retryable.
                    if resp.status_code >= 500 or resp.status_code == 429:
                        wait_time = 2 ** attempt
                        time.sleep(wait_time)
                        logger.info(f"[AI] Retrying after {wait_time}s due to {resp.status_code}")
                        continue
                    else:
                        return None

            except requests.exceptions.Timeout:
                logger.warning(f"[AI] Request timeout (attempt {attempt + 1}/{MAX_RETRIES})")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(2 ** attempt)
                    continue
                return None
            except requests.exceptions.ConnectionError:
                logger.warning(f"[AI] Connection error (attempt {attempt + 1}/{MAX_RETRIES})")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(2 ** attempt)
                    continue
                return None
            except Exception as e:
                logger.error(f"[AI] Unexpected error: {e}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(2 ** attempt)
                    continue
                return None

        return None

    def _maybe_audit(self, task: str, repo_or_ctx: str, prompt: str, raw_ans: str, parsed: str):
        """Record a prompt/response pair if auditing is on and this context is in scope.

        In scope means: a keyword-level task (always logged), or a repository
        that fell into the random 10% audit sample selected in main.py.
        """
        if not self.generate_audit:
            return

        if task.startswith("Keyword") or repo_or_ctx in self.audit_repos:
            entry = {
                "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Context": repo_or_ctx,
                "Task": task,
                "Model": self.model,
                "Prompt": prompt,
                "Raw_Response": raw_ans,
                "Parsed_Decision": parsed,
                "Human_Verdict": "",
                "Notes": "",
            }
            with self.audit_lock:
                self.audit_log.append(entry)

            # Flush immediately so the audit trail survives a crash mid-run.
            if self.audit_sink:
                self.audit_sink(entry)

    def export_audit_log_to_excel(self, filepath: str, records: Optional[List[dict]] = None):
        """Export the audit trail to Excel for human verification (CSV fallback if openpyxl is missing)."""
        data = records if records is not None else self.audit_log
        if not data:
            return
        try:
            df = pd.DataFrame(data)
            try:
                df.to_excel(filepath, index=False)
            except ImportError:
                csv_path = filepath.replace(".xlsx", ".csv")
                logger.warning(f"openpyxl missing. Falling back to CSV for AI audit: {csv_path}")
                df.to_csv(csv_path, index=False)
        except Exception as e:
            logger.error(f"Failed to export AI audit log: {e}")

    def classify_keyword(self, keyword: str) -> bool:
        """LLM fallback for keyword domain-relevance classification (software ecosystems / CS / SE / IT)."""
        if not self.is_available:
            return True

        sys_prompt = "Return ONLY JSON with keys: ok (boolean), domain (string), confidence (0..1)."
        prompt = (
            f"Classify a user keyword for relevance to software ecosystems / CS / SE / IT.\n"
            f"Examples:\n"
            f'{{"ok": true, "domain": "software-engineering", "confidence": 0.9}}\n'
            f'{{"ok": false, "domain": "unrelated", "confidence": 0.1}}\n'
            f"Keyword: {keyword}"
        )

        ans = self._ask(prompt, system_prompt=sys_prompt)
        is_ok = True
        try:
            if ans:
                # Strip markdown code-fence formatting the LLM sometimes adds around JSON.
                clean_ans = ans.strip("`").removeprefix("json").strip()
                data = json.loads(clean_ans)
                is_ok = bool(data.get("ok", True))
        except json.JSONDecodeError:
            pass

        self._maybe_audit("Keyword Classification", keyword, prompt, ans or "NO_RESPONSE", str(is_ok))
        return is_ok

    def classify_ops_fallback(self, text: str, repo_context: str = "") -> Tuple[int, int]:
        """LLM fallback for AIOps/MLOps domain classification when keyword heuristics find nothing."""
        if not text.strip() or not self.is_available:
            return 0, 0

        prompt = (
            "Classify this repository's primary domain.\n"
            "A = AIOps: IT-ops automation, root-cause analysis, log anomaly detection, self-healing.\n"
            "M = MLOps: ML pipeline orchestration, model training/deployment/versioning, feature stores.\n"
            "B = both A and M are clearly present. N = neither. If evidence is weak, answer N.\n"
            "Reply with exactly one letter: A, M, B, or N.\n\nText:\n" + text[:4000]
        )

        ans = (self._ask(prompt, max_tokens=5000) or "").upper()
        parsed = (1, 1) if "B" in ans else (1, 0) if "A" in ans else (0, 1) if "M" in ans else (0, 0)

        self._maybe_audit("Classify Ops Proxy", repo_context, prompt, ans or "NO_RESPONSE", str(parsed))
        return parsed

    def classify_commits(self, messages: Optional[List[str]], repo_context: str = "") -> int:
        """Flag commit histories that look templated/AI-generated rather than typical human shorthand."""
        if not messages or not self.is_available:
            return 0
        prompt = (
            "Audit these commit messages for AI-assistance signals.\n"
            "Answer YES only if several messages are templated, unusually verbose, "
            "or read like generated changelog text (e.g. 'Implement X as requested').\n"
            "Answer NO for normal human shorthand, typos, or terse fixes. If unsure, answer NO.\n"
            "Reply with exactly one word: YES or NO.\n\nCommits:\n" +
            "\n".join(f"- {m}" for m in messages[:20])
        )
        ans = self._ask(prompt, max_tokens=5000)
        parsed = 1 if ans and "YES" in ans.upper() else 0
        self._maybe_audit("Classify Commits", repo_context, prompt, ans or "NO_RESPONSE", str(parsed))
        return parsed

    def classify_readme(self, text: Optional[str], repo_context: str = "") -> int:
        """Classify a README as AI- or human-authored based on structural/stylistic signals."""
        if not text or len(text.strip()) < 50 or not self.is_available:
            return 0
        prompt = (
            "Classify this README as 'AI' or 'Human' authored.\n"
            "AI signals: uniform heading structure, generic hype phrasing, no project-specific "
            "voice, repeated transitions like 'Additionally'/'In conclusion'.\n"
            "Human signals: inconsistent formatting, personal tone, project-specific quirks. "
            "If unsure, answer Human.\n"
            "Reply with exactly one word: AI or Human.\n\nREADME:\n" + text[:3000]
        )
        ans = self._ask(prompt, max_tokens=5000)
        parsed = 1 if ans and "AI" in ans.upper() and "HUMAN" not in ans.upper() else 0
        self._maybe_audit("Classify README", repo_context, prompt, ans or "NO_RESPONSE", str(parsed))
        return parsed

    def classify_code(self, code: Optional[str], repo_context: str = "") -> int:
        """Classify a source snippet as AI- or human-authored based on structural/stylistic signals."""
        if not code or not code.strip() or not self.is_available:
            return 0
        prompt = (
            "Classify this source snippet as 'AI' or 'Human' authored.\n"
            "AI signals: exhaustive docstrings on trivial functions, boilerplate error handling, "
            "generic names, comments explaining obvious code.\n"
            "Human signals: sparse comments, project-specific shortcuts, inconsistent style. "
            "If unsure, answer Human.\n"
            "Reply with exactly one word: AI or Human.\n\nCode:\n" + code[:3000]
        )
        ans = self._ask(prompt, max_tokens=5000)
        parsed = 1 if ans and "AI" in ans.upper() and "HUMAN" not in ans.upper() else 0
        self._maybe_audit("Classify Code", repo_context, prompt, ans or "NO_RESPONSE", str(parsed))
        return parsed


llm_assistant = AIAssistant()
