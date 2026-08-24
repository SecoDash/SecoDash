"""Builds a repository knowledge graph from a derived checkpoint.

Produces two exports from the same graph:
  - a pruned, AI-readable JSON file of subject-predicate-object triples
    (degree-1 non-repo nodes removed, since a language/topic/package
    mentioned by only one repository adds little to cross-repo analysis)
  - a full GEXF file with computed node colors/sizes/labels, ready to open
    directly in Gephi for visual exploration.
"""
import json
import os
import math
from pathlib import Path

import networkx as nx
from state import logger

# Distinct RGB color per node type, used for the Gephi (GEXF) export.
NODE_COLORS = {
    'repo': {'r': 31, 'g': 119, 'b': 180, 'a': 1.0},         # Blue
    'language': {'r': 214, 'g': 39, 'b': 40, 'a': 1.0},      # Red
    'topic': {'r': 44, 'g': 160, 'b': 44, 'a': 1.0},         # Green
    'taxonomy': {'r': 148, 'g': 103, 'b': 189, 'a': 1.0},    # Purple
    'contributor': {'r': 255, 'g': 127, 'b': 14, 'a': 1.0},  # Orange
    'package': {'r': 227, 'g': 119, 'b': 194, 'a': 1.0},     # Pink
}
NODE_SIZE_RANGE = (10.0, 100.0)  # (min, max) node size in Gephi units.


def build_knowledge_graph(checkpoint_file: str, json_out: Path, gexf_out: Path):
    """Read a derived checkpoint and write the pruned JSON + full GEXF exports.

    Args:
        checkpoint_file: path to checkpoint_derived_<base_name>.jsonl.
        json_out: destination for the pruned AI-readable triples JSON.
        gexf_out: destination for the full Gephi-ready GEXF graph.
    """
    if not os.path.exists(checkpoint_file):
        logger.error(f"Checkpoint not found: {checkpoint_file}. Run the collector pipeline first.")
        return

    logger.info(f"Generating Knowledge Graph from {checkpoint_file}...")

    graph = nx.Graph()
    triples = []

    with open(checkpoint_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)

            repo_name = rec.get("repo_title") or rec.get("html_url")
            if not repo_name:
                continue

            graph.add_node(repo_name, type='repo')

            def add_edge(rel_type, target_name, node_type):
                if not target_name:
                    return
                target_id = f"{node_type}:{target_name}"
                if not graph.has_node(target_id):
                    graph.add_node(target_id, type=node_type)
                graph.add_edge(repo_name, target_id, relation=rel_type)

                triples.append({
                    "subject": repo_name,
                    "predicate": rel_type,
                    "object": target_name,
                    "tid": target_id,
                })

            add_edge("HAS_LANGUAGE", rec.get("primary_language"), "language")
            for topic in rec.get("topics", []):
                add_edge("HAS_TOPIC", topic, "topic")
            for tag in rec.get("taxonomy_tags", []):
                add_edge("HAS_TAXONOMY", tag, "taxonomy")
            for contributor in rec.get("contributors", []):
                add_edge("HAS_CONTRIBUTOR", contributor, "contributor")
            for pkg in rec.get("packages", []):
                add_edge("USES_PACKAGE", pkg, "package")

    # --- Export 1: AI-readable pruned triples (JSON) ------------------------
    # Drop non-repo nodes with degree 1: entities mentioned by only a single
    # repository add noise rather than cross-repo structure.
    to_remove = {n for n, deg in graph.degree() if deg == 1 and graph.nodes[n].get('type') != 'repo'}
    pruned_triples = [{k: v for k, v in t.items() if k != "tid"} for t in triples if t["tid"] not in to_remove]

    with open(json_out, 'w', encoding='utf-8') as f:
        json.dump({"metadata": {"nodes": graph.number_of_nodes() - len(to_remove)}, "triples": pruned_triples}, f, indent=2)
    logger.info(f"Saved AI-readable knowledge graph to {json_out}")

    # --- Export 2: full visual graph for Gephi (GEXF) -----------------------
    logger.info("Calculating visuals (colors, sizes, and labels) for Gephi export...")

    degrees = dict(graph.degree())
    max_deg = max(degrees.values()) if degrees else 1
    min_size, max_size = NODE_SIZE_RANGE

    for n in graph.nodes():
        deg = degrees.get(n, 0)
        node_type = graph.nodes[n].get('type', 'repo')

        # Strip the "type:" prefix used for uniqueness back off for display.
        display_name = n.split(":", 1)[1] if ":" in n and node_type != 'repo' else n
        graph.nodes[n]['label'] = f"[{node_type.capitalize()}] {display_name}"

        # Logarithmic size scale so a handful of hub nodes don't dwarf everything else.
        scale = math.log(deg + 1) / math.log(max_deg + 1) if max_deg > 1 else 0.5
        node_size = min_size + (scale * (max_size - min_size))
        node_color = NODE_COLORS.get(node_type, {'r': 128, 'g': 128, 'b': 128, 'a': 1.0})

        graph.nodes[n]['viz'] = {'color': node_color, 'size': node_size}

    try:
        # Modern GEXF 1.3, which Gephi expects natively.
        nx.write_gexf(graph, gexf_out, version="1.3")
        logger.info(f"Saved modern GEXF 1.3 (Gephi Native) to {gexf_out}")
    except Exception:
        # Older networkx versions reject "1.3" - fall back to the library default.
        nx.write_gexf(graph, gexf_out)
        logger.info(f"Saved standard GEXF to {gexf_out} (update networkx to silence Gephi warnings)")
