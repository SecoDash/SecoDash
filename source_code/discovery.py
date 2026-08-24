"""Repository discovery: keyword search against the GitHub and GitLab APIs.

Two sampling strategies are supported for GitHub (see search_github_repos):
  - "popularity": a star-sorted sliding cursor that walks from the highest-
    starred matches downward, re-querying with `stars:<=N` once a page's
    worth of results has been exhausted, to get past the API's 1000-result
    cap on any single query.
  - "stratified": repositories are bucketed into star-count bins, and each
    bin is randomly sub-sampled (via a date-sliding cursor per bin) so the
    final sample isn't dominated by a handful of mega-popular repos.

GitLab search is simpler (the Projects API doesn't expose the same
star-count cursor limitations) and is randomly sampled from a single pool.
"""
from datetime import datetime, timedelta
from typing import List, Optional, Set
import numpy as np

from config import GITHUB_API, GITLAB_API
from http_client import gh_client, gl_client
from state import SHUTDOWN, logger


def search_github_repos(keyword: str, max_repos: int, sampling: str, rng: np.random.Generator,
                         exclude_urls: Optional[Set[str]] = None) -> List[dict]:
    seen_urls = set(exclude_urls or set())
    results = []

    if sampling == "popularity":
        # Sliding cursor by star count: repeatedly query the top of the
        # remaining pool, then narrow with `stars:<=lowest_seen` so the next
        # query picks up where this one left off (works around the API's
        # 1000-result-per-query cap).
        current_max_stars = None
        while len(results) < max_repos and not SHUTDOWN.requested:
            query = f'{keyword} in:name,description,readme'
            if current_max_stars is not None:
                query += f' stars:<={current_max_stars}'

            page, items_fetched, page_lowest_stars = 1, 0, None
            exhausted_query = False

            while page <= 10 and len(results) < max_repos and not SHUTDOWN.requested:
                resp = gh_client.safe_get(f"{GITHUB_API}/search/repositories",
                                           params={"q": query, "sort": "stars", "order": "desc", "per_page": 100, "page": page})
                if not resp:
                    exhausted_query = True
                    break

                items = resp.json().get("items", [])
                if not items:
                    exhausted_query = True
                    break

                for item in items:
                    items_fetched += 1
                    url = item.get("html_url")
                    page_lowest_stars = item.get("stargazers_count")

                    if url and url not in seen_urls:
                        seen_urls.add(url)
                        results.append(item)
                        if len(results) >= max_repos:
                            break

                # A short page means there are no more matches for this query.
                if len(items) < 100:
                    exhausted_query = True
                    break

                page += 1

            if items_fetched == 0 or page_lowest_stars is None or exhausted_query:
                break  # Search space for this query is fully exhausted.

            # Hit the 1000-item cap (page 10): slide the cursor to keep going.
            if current_max_stars is not None and page_lowest_stars >= current_max_stars:
                current_max_stars -= 1
            else:
                current_max_stars = page_lowest_stars

            if current_max_stars < 0:
                break

        return results

    else:
        # Stratified sampling by star-count bin, each bin sub-sampled via a
        # date-sliding cursor (sorted by `updated`) to avoid biasing toward
        # whatever repos happen to sort first.
        bins = [
            f'{keyword} in:name,description,readme stars:0..10',
            f'{keyword} in:name,description,readme stars:11..100',
            f'{keyword} in:name,description,readme stars:101..1000',
            f'{keyword} in:name,description,readme stars:>1000',
        ]
        target_per_bin = max(1, max_repos // len(bins))
        deficit = 0

        for idx, base_query in enumerate(bins):
            if SHUTDOWN.requested or len(results) >= max_repos:
                break

            current_target = target_per_bin + deficit
            if idx == len(bins) - 1:
                current_target = max_repos - len(results)

            pool = []
            current_max_date = None
            pool_target = current_target * 3

            while len(pool) < pool_target and not SHUTDOWN.requested:
                query = base_query
                if current_max_date:
                    query += f' pushed:<={current_max_date}'

                page, items_fetched, page_oldest_date = 1, 0, None
                exhausted_query = False

                while page <= 10 and len(pool) < pool_target and not SHUTDOWN.requested:
                    resp = gh_client.safe_get(f"{GITHUB_API}/search/repositories",
                                               params={"q": query, "sort": "updated", "order": "desc", "per_page": 100, "page": page})
                    if not resp:
                        exhausted_query = True
                        break

                    items = resp.json().get("items", [])
                    if not items:
                        exhausted_query = True
                        break

                    for item in items:
                        items_fetched += 1
                        url = item.get("html_url")
                        if item.get("pushed_at"):
                            page_oldest_date = item.get("pushed_at").split('T')[0]

                        if url and url not in seen_urls:
                            seen_urls.add(url)
                            pool.append(item)
                            if len(pool) >= pool_target:
                                break

                    if len(items) < 100:
                        exhausted_query = True  # Exhausted this date range entirely.
                        break

                    page += 1

                if items_fetched == 0 or not page_oldest_date or exhausted_query:
                    break  # Nothing more available for this bin.

                try:
                    dt = datetime.strptime(page_oldest_date, "%Y-%m-%d") - timedelta(days=1)
                    next_date_str = dt.strftime("%Y-%m-%d")
                    if current_max_date == next_date_str:
                        break
                    current_max_date = next_date_str
                except Exception:
                    break

            sampled_count = min(len(pool), current_target)
            if sampled_count > 0:
                indices = rng.choice(len(pool), size=sampled_count, replace=False)
                results.extend([pool[i] for i in indices])

            deficit = current_target - sampled_count

        return results[:max_repos]


def search_gitlab_repos(keyword: str, max_repos: int, rng: np.random.Generator,
                         exclude_urls: Optional[Set[str]] = None) -> List[dict]:
    seen_urls = set(exclude_urls or set())
    pool = []
    page = 1

    while len(pool) < max_repos * 3 and not SHUTDOWN.requested:
        resp = gl_client.safe_get(f"{GITLAB_API}/projects",
                                   params={"search": keyword, "order_by": "last_activity_at", "sort": "desc", "per_page": 100, "page": page})
        if not resp:
            break
        items = resp.json()
        if not items:
            break

        pool.extend(items)

        if len(items) < 100:
            break  # Partial page: no more results.

        page += 1
        if page > 10:
            break

    fresh_pool = [item for item in pool if item.get("web_url") not in seen_urls]
    results = []
    sampled_count = min(len(fresh_pool), max_repos)
    if sampled_count > 0:
        indices = rng.choice(len(fresh_pool), size=sampled_count, replace=False)
        results = [fresh_pool[i] for i in indices]

    return results
