"""
Graph-Based NLP Evaluation Engine for ReasonGraph.

Evaluates the quality of a parsed reasoning graph using three NLP metrics:

  Metric A — Step Coherence (S_coh):
      Uses a cross-encoder NLI model to score how logically each step
      follows from the previous one (entailment probability on each edge).

  Metric B — Non-Redundancy / Efficiency (S_eff):
      Uses sentence embeddings (MiniLM) to detect repetitive / circular
      reasoning nodes.  S_eff = 1 - max pairwise cosine similarity.

  Metric C — Structural Validity (S_struct):
      Uses NetworkX to verify the graph is a valid DAG with at least one
      source (entry) node and one sink (terminal) node.

  Composite — Reasoning Quality Index (RQI):
      RQI = 0.5 * S_coh + 0.3 * S_eff + 0.2 * S_struct

All heavy models are lazy-loaded and then cached in module-level variables,
so the first call is slow (~seconds) and all subsequent calls are fast.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Module-level lazy caches
# ─────────────────────────────────────────────────────────────────────────────
_cross_encoder: Optional[Any] = None   # sentence_transformers CrossEncoder
_sentence_model: Optional[Any] = None  # SentenceTransformer (MiniLM)

_NLI_MODEL  = "cross-encoder/nli-deberta-v3-large"
_SENT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# NLI label order for cross-encoder/nli-deberta-v3-large:
# index 0 → contradiction, 1 → entailment, 2 → neutral
_ENTAILMENT_IDX = 1


def _get_cross_encoder():
    """Lazy-load and cache the NLI cross-encoder."""
    global _cross_encoder
    if _cross_encoder is None:
        try:
            from sentence_transformers import CrossEncoder
            logger.info(f"Loading NLI cross-encoder: {_NLI_MODEL} ...")
            _cross_encoder = CrossEncoder(_NLI_MODEL)
            logger.info("NLI cross-encoder loaded.")
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for graph evaluation. "
                "Install with: pip install sentence-transformers"
            )
    return _cross_encoder


def _get_sentence_model():
    """Lazy-load and cache the sentence embedding model."""
    global _sentence_model
    if _sentence_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading sentence-transformer: {_SENT_MODEL} ...")
            _sentence_model = SentenceTransformer(_SENT_MODEL)
            logger.info("Sentence-transformer loaded.")
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for graph evaluation. "
                "Install with: pip install sentence-transformers"
            )
    return _sentence_model


# ─────────────────────────────────────────────────────────────────────────────
# Metric implementations
# ─────────────────────────────────────────────────────────────────────────────

def _compute_s_coh(
    graph: nx.DiGraph,
    node_text: Dict[str, str],
) -> Tuple[float, List[Dict]]:
    """
    Metric A — Step-to-Step Coherence.

    For every directed edge (u -> v) compute the NLI entailment probability
    P(entailment | u -> v).  The score is the mean over all edges.

    Returns
    -------
    s_coh : float in [0, 1]
    edge_scores : list of {source, target, score} dicts
    """
    edges = list(graph.edges())
    if not edges:
        return 1.0, []

    encoder = _get_cross_encoder()

    pairs = [(node_text.get(u, u), node_text.get(v, v)) for u, v in edges]

    import numpy as np
    from scipy.special import softmax as scipy_softmax

    # predict() returns raw logits for each class
    raw_logits = encoder.predict(pairs, apply_softmax=False)
    # raw_logits shape: (n_edges, 3) or (3,) for a single pair
    if raw_logits.ndim == 1:
        raw_logits = raw_logits[None, :]  # add batch dim

    probs = scipy_softmax(raw_logits, axis=1)  # (n_edges, 3)
    entailment_probs = probs[:, _ENTAILMENT_IDX].tolist()

    edge_scores = [
        {"source": u, "target": v, "score": round(float(score), 4)}
        for (u, v), score in zip(edges, entailment_probs)
    ]

    s_coh = float(np.mean(entailment_probs))
    return s_coh, edge_scores


def _compute_s_eff(node_text: Dict[str, str]) -> float:
    """
    Metric B — Non-Redundancy / Efficiency.

    Embed every node with MiniLM, then find the maximum pairwise cosine
    similarity among distinct nodes.

    S_eff = 1 - max_pairwise_cosine_similarity

    A score close to 1 means nodes are highly diverse (good).
    A score close to 0 means there is a near-duplicate pair (bad / circular).

    Returns float in [0, 1].
    """
    texts = list(node_text.values())
    if len(texts) < 2:
        return 1.0  # trivially non-redundant

    model = _get_sentence_model()

    import numpy as np

    embeddings = model.encode(texts, normalize_embeddings=True)  # (n, d) – unit vectors
    # Cosine similarity matrix = dot product of unit vectors
    sim_matrix = embeddings @ embeddings.T  # (n, n)

    # Mask the diagonal (self-similarity = 1.0)
    np.fill_diagonal(sim_matrix, -1.0)
    max_sim = float(sim_matrix.max())

    s_eff = 1.0 - max(max_sim, 0.0)
    return s_eff


def _compute_s_struct(graph: nx.DiGraph) -> float:
    """
    Metric C — Structural Validity.

    Checks:
      1. The graph is a directed acyclic graph (DAG).
      2. There is at least one source node (in-degree = 0).
      3. There is at least one sink node (out-degree = 0).

    Scoring:
      All three checks pass  -> 1.0
      Only the DAG check passes -> 0.5
      Not a DAG              -> 0.0
    """
    if not nx.is_directed_acyclic_graph(graph):
        return 0.0

    has_source = any(graph.in_degree(n) == 0 for n in graph.nodes())
    has_sink   = any(graph.out_degree(n) == 0 for n in graph.nodes())

    if has_source and has_sink:
        return 1.0
    return 0.5  # DAG but no clear entry/exit


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_graph(
    nodes: List[Dict],
    edges: List[Dict],
) -> Dict:
    """
    Evaluate a reasoning graph and return NLP quality metrics.

    Parameters
    ----------
    nodes : list of {"id": str, "text": str}
    edges : list of {"source": str, "target": str}

    Returns
    -------
    dict with keys:
        rqi        - Reasoning Quality Index  (composite, 0-1)
        s_coh      - Step coherence score     (0-1)
        s_eff      - Non-redundancy score     (0-1)
        s_struct   - Structural validity      (0 / 0.5 / 1)
        edge_scores - list of {source, target, score} per directed edge
        node_count  - int
        edge_count  - int
        error       - str or None  (set if evaluation partially failed)
    """
    # Build a text lookup and the NetworkX graph
    node_text: Dict[str, str] = {n["id"]: n.get("text", n["id"]) for n in nodes}

    graph = nx.DiGraph()
    graph.add_nodes_from(node_text.keys())
    graph.add_edges_from((e["source"], e["target"]) for e in edges)

    error_msg: Optional[str] = None

    # Metric C (cheapest - pure graph, no model)
    try:
        s_struct = _compute_s_struct(graph)
    except Exception as exc:
        logger.error(f"s_struct computation failed: {exc}")
        s_struct = 0.0
        error_msg = str(exc)

    # Metric B
    try:
        s_eff = _compute_s_eff(node_text)
    except Exception as exc:
        logger.error(f"s_eff computation failed: {exc}")
        s_eff = 0.0
        error_msg = str(exc)

    # Metric A
    try:
        s_coh, edge_scores = _compute_s_coh(graph, node_text)
    except Exception as exc:
        logger.error(f"s_coh computation failed: {exc}")
        s_coh = 0.0
        edge_scores = []
        error_msg = str(exc)

    # Composite RQI
    rqi = 0.5 * s_coh + 0.3 * s_eff + 0.2 * s_struct

    return {
        "rqi":         round(rqi,      4),
        "s_coh":       round(s_coh,    4),
        "s_eff":       round(s_eff,    4),
        "s_struct":    round(s_struct, 4),
        "edge_scores": edge_scores,
        "node_count":  len(nodes),
        "edge_count":  len(edges),
        "error":       error_msg,
    }
