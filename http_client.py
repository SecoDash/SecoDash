"""Resilient HTTP client for the GitHub and GitLab REST APIs.

Handles auth headers, rate-limit back-off (403/429), transient server
errors (5xx), and network exceptions with exponential back-off, honoring
graceful shutdown requests between retries.
"""
import time
from typing import List, Optional
import requests

from config import MAX_RETRIES, GITHUB_TOKEN, GITLAB_TOKEN, REQUEST_TIMEOUT
from state import SHUTDOWN, logger


class RobustHTTPClient:
    def __init__(self, platform: str, token: str):
        self.platform = platform
        self.headers = {"User-Agent": "SecoDash/2.0"}

        if platform == "github":
            self.headers["Accept"] = "application/vnd.github+json"
            if token:
                self.headers["Authorization"] = f"Bearer {token}"
        elif platform == "gitlab" and token:
            self.headers["PRIVATE-TOKEN"] = token

    def safe_get(self, url: str, params: Optional[dict] = None) -> Optional[requests.Response]:
        """GET with retry/back-off. Returns None on shutdown, exhausted retries, or a 4xx error."""
        backoff = 1.0
        for attempt in range(MAX_RETRIES):
            if SHUTDOWN.requested:
                return None
            try:
                resp = requests.get(url, headers=self.headers, params=params, timeout=REQUEST_TIMEOUT)
                if resp.status_code == 200:
                    return resp

                if resp.status_code in (403, 429):
                    reset = resp.headers.get("X-RateLimit-Reset")
                    wait = max(1, int(reset) - int(time.time())) + 1 if reset else min(backoff, 60)
                    logger.warning(f"[{self.platform}] Rate limited. Waiting {wait}s...")
                    time.sleep(wait)
                    logger.info(f"[{self.platform}] Resuming operations after sleep.")
                    backoff = min(backoff * 2, 60)
                    continue

                if resp.status_code >= 500:
                    logger.warning(f"[{self.platform}] Server error {resp.status_code}. Retry {attempt + 1}/{MAX_RETRIES}")
                    time.sleep(min(backoff, 5))
                    logger.info(f"[{self.platform}] Resuming operations after sleep.")
                    backoff = min(backoff * 1.5, 5)
                    continue

                # Non-rate-limit 4xx: not retryable, fail fast.
                return None

            except requests.RequestException as e:
                logger.warning(f"[{self.platform}] Network error: {e}. Retry {attempt + 1}/{MAX_RETRIES}")
                time.sleep(min(backoff, 10))
                logger.info(f"[{self.platform}] Resuming operations after sleep.")
                backoff = min(backoff * 2, 30)

        return None

    def safe_get_paginated(self, url: str, params: Optional[dict] = None, max_pages: int = 100) -> Optional[List[dict]]:
        """Follow `page`/`per_page` pagination until a short page or max_pages is hit."""
        results = []
        p = params.copy() if params else {}
        p.setdefault("per_page", 100)
        p.setdefault("page", 1)

        while p["page"] <= max_pages and not SHUTDOWN.requested:
            resp = self.safe_get(url, p)
            if not resp:
                return None
            data = resp.json()
            if not isinstance(data, list):
                return None
            results.extend(data)
            if len(data) < p["per_page"]:
                break
            p["page"] += 1

        return results


gh_client = RobustHTTPClient("github", GITHUB_TOKEN)
gl_client = RobustHTTPClient("gitlab", GITLAB_TOKEN)
