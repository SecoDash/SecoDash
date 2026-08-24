"""Fetches and enriches raw repository data from GitHub and GitLab.

Two responsibilities live here:
  1. `extract_packages` - best-effort, regex-based dependency manifest
     parsing across a dozen ecosystems (no per-language TOML/YAML parser
     dependencies, by design - keeps this module dependency-light).
  2. `enrich_raw_github` / `enrich_raw_gitlab` - pull the full raw record
     (README, file tree, languages, contributors, commit history,
     packages, ...) for a single repository, ready for derivation.py.
"""
import base64
import json
import re
from typing import Dict, List, Optional, Union
from urllib.parse import quote

from state import logger
from config import GITHUB_API, GITLAB_API, CODE_EXTENSIONS, MANIFEST_FILES
from http_client import gh_client, gl_client


def compute_file_hints(paths: Optional[List[str]]) -> Dict[str, Optional[int]]:
    """Derive cheap boolean signals from a repo's file tree (no content needed)."""
    if paths is None:
        return {k: None for k in ["has_dockerfile", "has_citation_cff", "has_docs_dir", "has_tests", "has_workflows", "has_mlops_tools"]}

    joined = "\n".join([p.lower() for p in paths])

    def has(pattern: str) -> int:
        return int(bool(re.search(pattern, joined, re.MULTILINE)))

    return {
        "has_dockerfile": has(r"(^|/)dockerfile"),
        "has_citation_cff": has(r"(^|/)citation\.cff$"),
        "has_docs_dir": has(r"(^|/)(docs?|gh-pages)/"),
        "has_tests": has(r"(^|/)(tests?)/") | has(r"(_test\.py|_test\.rs|\.spec\.[jt]s|test_.*\.py)"),
        "has_workflows": has(r"^\.github/workflows/|^\.gitlab-ci\.yml"),
        "has_mlops_tools": has(r"(^|/)\.dvc(/|$)|dvc\.yaml|mlflow|kubeflow"),
    }


def gh_fetch_file(owner: str, repo: str, path: str) -> Optional[str]:
    """Fetch and base64-decode a single file's contents from the GitHub Contents API."""
    resp = gh_client.safe_get(f"{GITHUB_API}/repos/{owner}/{repo}/contents/{path}")
    if not resp:
        return None
    data = resp.json()
    if isinstance(data, list) or data.get("type") == "dir":
        return None
    try:
        return base64.b64decode(data.get("content", "")).decode("utf-8", errors="ignore")
    except Exception:
        return None


def extract_packages(paths: Optional[List[str]], fetcher_func, fetcher_args: dict) -> Union[List[str], int, None]:
    """Parse declared dependencies from whichever manifest files are present.

    `fetcher_func`/`fetcher_args` abstract over the platform (GitHub vs.
    GitLab file-content APIs) so this function stays platform-agnostic.
    Each ecosystem's manifest format is parsed with a small best-effort
    regex/line-scanner rather than a full parser, since only the
    dependency *names* are needed, not a complete resolution graph.

    Returns:
        - a de-duplicated list of package names if any manifest was found
          and successfully parsed,
        - `1` if a recognized manifest file exists but none of the
          dedicated parsers above extracted anything from it (fallback
          "packages present, count unknown" signal),
        - `[]` if no recognized manifest file was found at all.
    """
    if paths is None:
        return None
    lower_paths = {p.lower(): p for p in paths}

    def try_fetch(name: str) -> Optional[str]:
        name = name.lower()
        for lp, orig in lower_paths.items():
            if lp.endswith(name):
                return fetcher_func(path=orig, **fetcher_args)
        return None

    packages = []

    # --- Python: requirements.txt --------------------------------------
    req_txt = try_fetch("requirements.txt")
    if req_txt:
        for line in req_txt.splitlines():
            pkg = re.split(r"[<>=!~\[]", line.strip())[0].strip()
            if pkg and not pkg.startswith("#"):
                packages.append(pkg)

    # --- Node.js: package.json -------------------------------------------
    pkg_json = try_fetch("package.json")
    if pkg_json:
        try:
            data = json.loads(pkg_json)
            for sec in ("dependencies", "devDependencies"):
                if isinstance(data.get(sec), dict):
                    packages.extend(data[sec].keys())
        except Exception:
            pass

    # --- R: DESCRIPTION (Depends:/Imports: fields) -----------------------
    desc = try_fetch("DESCRIPTION")
    if desc:
        for line in desc.splitlines():
            if line.startswith("Depends:") or line.startswith("Imports:"):
                deps = re.split(r"[, ]+", line.split(":", 1)[1].strip())
                packages.extend([d.split("(")[0].strip() for d in deps if d and d != "R"])

    # --- Julia: Project.toml ---------------------------------------------
    proj_toml = try_fetch("Project.toml")
    if proj_toml:
        try:
            for line in proj_toml.splitlines():
                if "=" in line and not line.strip().startswith("#"):
                    key = line.split("=")[0].strip().strip('"')
                    if key in ["deps", "compat"]:
                        packages.append(key)
                if line.strip().startswith("["):
                    pkg = line.strip().strip("[]")
                    if pkg and pkg not in ["deps", "compat"]:
                        packages.append(pkg)
        except Exception:
            pass

    # --- C++: CMakeLists.txt (find_package / target_link_libraries) -----
    cmake = try_fetch("CMakeLists.txt")
    if cmake:
        for line in cmake.splitlines():
            if "find_package(" in line.lower():
                pkg_match = re.search(r"find_package\s*\(\s*([^\s)]+)", line, re.I)
                if pkg_match:
                    packages.append(pkg_match.group(1))
            elif "target_link_libraries(" in line.lower():
                libs = re.findall(r"target_link_libraries\s*\([^)]+\)", line, re.I)
                for lib in libs:
                    for item in lib.split():
                        if item and not item.startswith("target_link_libraries") and item not in ["(", ")"]:
                            packages.append(item.strip())

    # --- C#/F#: *.csproj / *.fsproj (PackageReference) --------------------
    csproj_files = [p for p in paths if p.lower().endswith((".csproj", ".fsproj"))]
    for csproj in csproj_files:
        content = fetcher_func(path=csproj, **fetcher_args)
        if content:
            for line in content.splitlines():
                if "PackageReference" in line:
                    pkg_match = re.search(r'Include="([^"]+)"', line)
                    if pkg_match:
                        packages.append(pkg_match.group(1))

    # --- PHP: composer.json ------------------------------------------------
    composer = try_fetch("composer.json")
    if composer:
        try:
            data = json.loads(composer)
            for sec in ("require", "require-dev"):
                if isinstance(data.get(sec), dict):
                    packages.extend(data[sec].keys())
        except Exception:
            pass

    # --- Perl: cpanfile ------------------------------------------------------
    cpanfile = try_fetch("cpanfile")
    if cpanfile:
        for line in cpanfile.splitlines():
            if line.strip() and not line.strip().startswith("#"):
                pkg_match = re.search(r"(?:requires|recommends|suggests)\s+['\"]?([^\s'\"]+)", line)
                if pkg_match:
                    packages.append(pkg_match.group(1))

    # --- Rust: Cargo.toml ----------------------------------------------------
    # Line-scanner (no TOML dependency, consistent with the rest of this
    # function). Handles `name = "..."` and `name = { version = "...", ... }`
    # forms inside [dependencies] / [dev-dependencies] / [build-dependencies],
    # plus dotted target-specific sections such as
    # [target.'cfg(windows)'.dependencies.winapi]. Workspace-inherited
    # dependencies (`name.workspace = true`) are captured by name only, not resolved.
    cargo_toml = try_fetch("Cargo.toml")
    if cargo_toml:
        in_deps_section = False
        dep_section_names = ("dependencies", "dev-dependencies", "build-dependencies")
        for line in cargo_toml.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            section_match = re.match(r"^\[(.+)\]$", stripped)
            if section_match:
                section_name = section_match.group(1).strip().strip('"')
                if section_name in dep_section_names:
                    in_deps_section = True
                elif any(section_name.endswith("." + s) for s in dep_section_names):
                    # e.g. target.'cfg(windows)'.dependencies -> crates appear as
                    # their own sub-lines below, not in the section header itself.
                    in_deps_section = True
                elif any(("." + s + ".") in section_name for s in dep_section_names):
                    # e.g. target.'cfg(windows)'.dependencies.winapi -> crate name
                    # is the trailing path segment.
                    packages.append(section_name.split(".")[-1].strip())
                    in_deps_section = False
                else:
                    in_deps_section = False
                continue
            if in_deps_section and "=" in stripped:
                pkg = stripped.split("=", 1)[0].strip().strip('"')
                if pkg:
                    packages.append(pkg)

    # --- Go: go.mod ------------------------------------------------------------
    go_mod = try_fetch("go.mod")
    if go_mod:
        in_require_block = False
        for line in go_mod.splitlines():
            stripped = line.strip()
            if stripped.startswith("require ("):
                in_require_block = True
                continue
            if in_require_block:
                if stripped == ")":
                    in_require_block = False
                    continue
                mod_match = re.match(r"^([^\s]+)\s+v[\d]", stripped)
                if mod_match:
                    packages.append(mod_match.group(1))
            elif stripped.startswith("require "):
                mod_match = re.match(r"^require\s+([^\s]+)\s+v[\d]", stripped)
                if mod_match:
                    packages.append(mod_match.group(1))

    # --- Java: pom.xml (Maven) --------------------------------------------------
    pom_xml = try_fetch("pom.xml")
    if pom_xml:
        for dep_block in re.findall(r"<dependency>(.*?)</dependency>", pom_xml, re.S):
            artifact_match = re.search(r"<artifactId>([^<]+)</artifactId>", dep_block)
            if artifact_match:
                packages.append(artifact_match.group(1).strip())

    # --- Java/Kotlin: build.gradle / build.gradle.kts (Gradle) -------------------
    gradle_files = [p for p in paths if p.lower().endswith(("build.gradle", "build.gradle.kts"))]
    for gradle_path in gradle_files:
        content = fetcher_func(path=gradle_path, **fetcher_args)
        if content:
            for line in content.splitlines():
                dep_match = re.search(
                    r"(?:implementation|api|compile|testImplementation|runtimeOnly|compileOnly|kapt|annotationProcessor)"
                    r"\s*[\(\s]\s*['\"]([^'\"]+)['\"]",
                    line,
                )
                if dep_match:
                    packages.append(dep_match.group(1).strip())

    # --- Conda: environment.yml --------------------------------------------------
    conda_env = try_fetch("environment.yml")
    if conda_env:
        in_deps_section = False
        for line in conda_env.splitlines():
            raw_line = line.rstrip("\n")
            stripped = raw_line.strip()
            if re.match(r"^dependencies:\s*$", stripped):
                in_deps_section = True
                continue
            if in_deps_section:
                if stripped and not raw_line.startswith((" ", "\t", "-")):
                    in_deps_section = False  # dedented back out of the dependencies block
                    continue
                item_match = re.match(r"^-\s*([A-Za-z0-9_.\-]+)", stripped)
                if item_match:
                    pkg = re.split(r"[<>=! ]", item_match.group(1))[0]
                    if pkg and pkg.lower() != "pip":
                        packages.append(pkg)

    # --- Erlang: rebar.config ----------------------------------------------------
    # rebar.config is Erlang term syntax, not TOML/JSON/YAML, so this pulls the
    # first atom of each tuple inside the `{deps, [...]}` list rather than doing
    # a full term parse. Can pick up noise from nested tuples (e.g. `git`, `tag`
    # inside a `{git, "...", {tag, "..."}}` dependency spec); good enough to move
    # Erlang from "always zero" to "roughly right", not a substitute for a real parser.
    rebar_config = try_fetch("rebar.config")
    if rebar_config:
        deps_match = re.search(r"\{deps,\s*\[(.*?)\]\}", rebar_config, re.S)
        if deps_match:
            deps_block = deps_match.group(1)
            noisy_tokens = {"git", "tag", "branch", "ref", "raw"}
            for dep_match in re.finditer(r"\{\s*([a-z_][a-zA-Z0-9_]*)\s*,", deps_block):
                pkg = dep_match.group(1)
                if pkg and pkg not in noisy_tokens:
                    packages.append(pkg)

    if packages:
        return list(set(packages))

    # No dedicated parser matched - fall back to a general manifest-file scan
    # that only reports "packages likely exist" (1) without a concrete list.
    for p in paths:
        p_lower = p.lower()
        for manifest in MANIFEST_FILES.keys():
            m_lower = manifest.lower()
            if m_lower.startswith("*."):
                if p_lower.endswith(m_lower[1:]):
                    return 1
            else:
                if p_lower.endswith(m_lower):
                    return 1

    return []


def enrich_raw_github(item: dict) -> dict:
    """Build the full raw record for one GitHub repository search result."""
    owner = item.get("owner", {}).get("login", "")
    name = item.get("name", "")
    default_branch = item.get("default_branch", "main")

    readme = None
    r_resp = gh_client.safe_get(f"{GITHUB_API}/repos/{owner}/{name}/readme")
    if r_resp:
        try:
            readme = base64.b64decode(r_resp.json().get("content", "")).decode("utf-8", "ignore")
        except Exception:
            pass

    tree_paths = None
    t_resp = gh_client.safe_get(f"{GITHUB_API}/repos/{owner}/{name}/git/trees/{default_branch}", params={"recursive": "1"})
    if t_resp:
        tree_paths = [node.get("path", "") for node in t_resp.json().get("tree", [])]

    raw_sample_code = None
    if tree_paths:
        code_path = next((p for p in tree_paths if p.lower().endswith(CODE_EXTENSIONS)), None)
        if code_path:
            raw_sample_code = gh_fetch_file(owner, name, code_path)

    l_resp = gh_client.safe_get(f"{GITHUB_API}/repos/{owner}/{name}/languages")
    languages = l_resp.json() if l_resp else None

    contributors = gh_client.safe_get_paginated(f"{GITHUB_API}/repos/{owner}/{name}/contributors", {"anon": "true"})
    branches = gh_client.safe_get_paginated(f"{GITHUB_API}/repos/{owner}/{name}/branches")

    c_total, c_msgs = None, None
    c_resp = gh_client.safe_get(f"{GITHUB_API}/repos/{owner}/{name}/commits", {"sha": default_branch, "per_page": 1})
    if c_resp:
        c_total = 1
        link = c_resp.headers.get("Link", "")
        m = re.search(r'[?&]page=(\d+)>; rel="last"', link)
        if m:
            c_total = int(m.group(1))
        m_resp = gh_client.safe_get(f"{GITHUB_API}/repos/{owner}/{name}/commits", {"sha": default_branch, "per_page": 30})
        if m_resp:
            c_msgs = [c.get("commit", {}).get("message", "").split("\n")[0] for c in m_resp.json()]

    d_resp = gh_client.safe_get_paginated(f"{GITHUB_API}/repos/{owner}/{name}/releases", max_pages=10)
    release_dls = sum(a.get("download_count", 0) for r in (d_resp or []) for a in r.get("assets", [])) if d_resp is not None else None

    pkgs = extract_packages(tree_paths, gh_fetch_file, {"owner": owner, "repo": name})

    return {
        "source": "github",
        "html_url": item.get("html_url"),
        "owner": owner,
        "repo_title": name,
        "description": item.get("description", ""),
        "topics": item.get("topics", []),
        "stargaze_count": item.get("stargazers_count"),
        "forks_count": item.get("forks_count"),
        "watchers_count": item.get("subscribers_count", item.get("watchers_count")),
        "has_issues": int(bool(item.get("has_issues"))),
        "license_key": (item.get("license") or {}).get("key") if item.get("license") else None,
        "created_at": item.get("created_at"),
        "pushed_at": item.get("pushed_at"),
        "raw_readme": readme,
        "raw_tree_paths": tree_paths,
        "raw_sample_code": raw_sample_code,
        "raw_languages": languages,
        "raw_packages": pkgs if isinstance(pkgs, list) else [],
        "raw_packages_indicator": 1 if pkgs else 0,
        "raw_contributors": [c.get("login") for c in contributors if c.get("login")] if contributors else [],
        "raw_contributors_count": len(contributors) if contributors is not None else None,
        "raw_branches_count": len(branches) if branches is not None else None,
        "raw_commits_total": c_total,
        "raw_commits_msgs": c_msgs,
        "raw_release_downloads": release_dls,
    }


def gl_fetch_file(project_id: int, ref: str, path: str) -> Optional[str]:
    """Fetch a single file's raw contents from the GitLab Repository Files API."""
    enc_path = quote(path, safe="")
    resp = gl_client.safe_get(f"{GITLAB_API}/projects/{project_id}/repository/files/{enc_path}/raw", params={"ref": ref})
    return resp.text if resp else None


def enrich_raw_gitlab(item: dict) -> dict:
    """Build the full raw record for one GitLab project search result."""
    pid = item.get("id")
    ref = item.get("default_branch", "main")

    readme = gl_fetch_file(pid, ref, "README.md")
    t_data = gl_client.safe_get_paginated(f"{GITLAB_API}/projects/{pid}/repository/tree", {"recursive": "true"}, max_pages=20)
    tree_paths = [node.get("path", "") for node in t_data] if t_data is not None else None

    raw_sample_code = None
    if tree_paths:
        code_path = next((p for p in tree_paths if p.lower().endswith(CODE_EXTENSIONS)), None)
        if code_path:
            raw_sample_code = gl_fetch_file(pid, ref, code_path)

    l_resp = gl_client.safe_get(f"{GITLAB_API}/projects/{pid}/languages")
    languages = l_resp.json() if l_resp else None
    contribs = gl_client.safe_get_paginated(f"{GITLAB_API}/projects/{pid}/repository/contributors")
    branches = gl_client.safe_get_paginated(f"{GITLAB_API}/projects/{pid}/repository/branches")

    stat_resp = gl_client.safe_get(f"{GITLAB_API}/projects/{pid}", {"statistics": "true"})
    c_total = (stat_resp.json().get("statistics") or {}).get("commit_count") if stat_resp else None

    m_resp = gl_client.safe_get(f"{GITLAB_API}/projects/{pid}/repository/commits", {"per_page": 30})
    c_msgs = [c.get("title", "") for c in m_resp.json()] if m_resp else None

    d_resp = gl_client.safe_get_paginated(f"{GITLAB_API}/projects/{pid}/releases", max_pages=10)
    release_dls = sum(a.get("downloads", 0) for r in (d_resp or []) for a in r.get("assets", {}).get("links", [])) if d_resp is not None else None

    pkgs = extract_packages(tree_paths, gl_fetch_file, {"project_id": pid, "ref": ref})

    issues_url = (item.get("_links") or {}).get("issues")
    has_issues = int(bool(issues_url)) if issues_url is not None else None

    return {
        "source": "gitlab",
        "html_url": item.get("web_url"),
        "owner": (item.get("namespace") or {}).get("full_path", ""),
        "repo_title": item.get("name"),
        "description": item.get("description", ""),
        "topics": item.get("tag_list", []),
        "stargaze_count": item.get("star_count"),
        "forks_count": item.get("forks_count"),
        "watchers_count": None,
        "has_issues": has_issues,
        "license_key": item.get("license", {}).get("key") if item.get("license") else None,
        "created_at": item.get("created_at"),
        "pushed_at": item.get("last_activity_at"),
        "raw_readme": readme,
        "raw_tree_paths": tree_paths,
        "raw_sample_code": raw_sample_code,
        "raw_languages": languages,
        "raw_packages": pkgs if isinstance(pkgs, list) else [],
        "raw_packages_indicator": 1 if pkgs else 0,
        "raw_contributors": [c.get("name") for c in contribs if c.get("name")] if contribs else [],
        "raw_contributors_count": len(contribs) if contribs is not None else None,
        "raw_branches_count": len(branches) if branches is not None else None,
        "raw_commits_total": c_total,
        "raw_commits_msgs": c_msgs,
        "raw_release_downloads": release_dls,
    }
