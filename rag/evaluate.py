import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


def load_jsonl(path: Path) -> List[Dict]:
    out: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def load_doc_ids(index_dir: Path) -> List[str]:
    with open(index_dir / "doc_ids.json", "r", encoding="utf-8") as f:
        return json.load(f)


def load_recipes_map(jsonl_path: Path) -> Dict[str, Dict]:
    m: Dict[str, Dict] = {}
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            rid = str(d.get("id", ""))
            if rid:
                m[rid] = d
    return m


def encode_query(model: SentenceTransformer, query: str) -> np.ndarray:
    vec = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    return vec.astype("float32")


def tokenize_ingredients(ingr_list: List[str]) -> List[str]:
    return [str(x).strip().lower() for x in ingr_list if str(x).strip()]


def build_validation_queries(
    recipes: Dict[str, Dict],
    n_queries: int,
    seed: int,
) -> List[Dict]:
    """
    Self-retrieval validation:
    - Pick random recipes.
    - Build query from 3-5 normalized ingredients (+ dish_type='soup' if in title).
    - Expected answer is the same recipe id.
    """
    rng = random.Random(seed)
    all_ids = list(recipes.keys())
    rng.shuffle(all_ids)

    queries: List[Dict] = []
    for rid in all_ids:
        r = recipes[rid]
        title = str(r.get("title", "")).lower()
        ingr_norm = tokenize_ingredients(r.get("ingredients_normalized", []))
        if len(ingr_norm) < 3:
            continue

        dish_type = "soup" if "soup" in title else ""
        k = rng.randint(3, 5)
        picked = rng.sample(ingr_norm, k=min(k, len(ingr_norm)))

        query_text = (dish_type + " " + ", ".join(picked)).strip()
        queries.append(
            {
                "query": query_text,
                "dish_type": dish_type,
                "ingredients": picked,
                "expected_id": rid,
            }
        )
        if len(queries) >= n_queries:
            break

    return queries


def faiss_retrieve(
    index: faiss.Index,
    doc_ids: List[str],
    qvec: np.ndarray,
    k: int,
) -> List[Tuple[str, float]]:
    sims, idxs = index.search(qvec, k)
    out: List[Tuple[str, float]] = []
    for j in range(idxs.shape[1]):
        i = int(idxs[0, j])
        if 0 <= i < len(doc_ids):
            out.append((doc_ids[i], float(sims[0, j])))
    return out


def rerank_candidates(
    candidates: List[Tuple[str, float]],
    recipes: Dict[str, Dict],
    dish_type: str,
    user_ingredients: List[str],
) -> List[Tuple[str, float]]:
    """
    Same rerank idea as in retriever.py (kept here self-contained for evaluation).
    Non-linear part: hard filter on overlap >= 1 when ingredients provided.
    """
    dish_kw = (dish_type or "").strip().lower()
    user_set = set([x.strip().lower() for x in user_ingredients if x.strip()])

    alpha = 0.35
    beta = 0.15

    scored: List[Tuple[str, float]] = []
    for rid, sim in candidates:
        r = recipes.get(rid)
        if not r:
            continue

        title = str(r.get("title", "")).lower()
        ingr_norm = [str(x).lower() for x in r.get("ingredients_normalized", [])]
        ingr_set = set(ingr_norm)

        overlap = len(user_set & ingr_set) if user_set else 0
        overlap_ratio = (overlap / max(1, len(user_set))) if user_set else 0.0

        if user_set and overlap == 0:
            continue

        score = float(sim) + alpha * overlap_ratio
        if dish_kw and dish_kw in title:
            score += beta

        scored.append((rid, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


def hit_at_k(ranked_ids: List[str], expected_id: str, k: int) -> float:
    return 1.0 if expected_id in ranked_ids[:k] else 0.0


def mrr_at_k(ranked_ids: List[str], expected_id: str, k: int) -> float:
    top = ranked_ids[:k]
    for i, rid in enumerate(top, start=1):
        if rid == expected_id:
            return 1.0 / i
    return 0.0


def evaluate(
    index: faiss.Index,
    doc_ids: List[str],
    recipes: Dict[str, Dict],
    model: SentenceTransformer,
    queries: List[Dict],
    retrieve_k: int,
) -> Dict:
    metrics = {
        "baseline_hit@5": 0.0,
        "baseline_hit@10": 0.0,
        "baseline_mrr@10": 0.0,
        "rerank_hit@5": 0.0,
        "rerank_hit@10": 0.0,
        "rerank_mrr@10": 0.0,
    }

    for q in tqdm(queries, desc="Evaluating"):
        qtext = q["query"]
        expected = q["expected_id"]
        dish_type = q.get("dish_type", "")
        ingr = q.get("ingredients", [])

        qvec = encode_query(model, qtext)
        candidates = faiss_retrieve(index, doc_ids, qvec, retrieve_k)

        baseline_ranked_ids = [rid for rid, _ in candidates]

        reranked = rerank_candidates(candidates, recipes, dish_type, ingr)
        rerank_ranked_ids = [rid for rid, _ in reranked]

        metrics["baseline_hit@5"] += hit_at_k(baseline_ranked_ids, expected, 5)
        metrics["baseline_hit@10"] += hit_at_k(baseline_ranked_ids, expected, 10)
        metrics["baseline_mrr@10"] += mrr_at_k(baseline_ranked_ids, expected, 10)

        metrics["rerank_hit@5"] += hit_at_k(rerank_ranked_ids, expected, 5)
        metrics["rerank_hit@10"] += hit_at_k(rerank_ranked_ids, expected, 10)
        metrics["rerank_mrr@10"] += mrr_at_k(rerank_ranked_ids, expected, 10)

    n = max(1, len(queries))
    for k in list(metrics.keys()):
        metrics[k] /= n
    metrics["n_queries"] = len(queries)
    metrics["retrieve_k"] = retrieve_k
    return metrics


def save_jsonl(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to recipes.jsonl")
    parser.add_argument("--index-dir", default="data/index")
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--n-queries", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--retrieve-k", type=int, default=25)
    parser.add_argument("--out-dir", default="data/validation")
    args = parser.parse_args()

    index_dir = Path(args.index_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    recipes = load_recipes_map(Path(args.data))
    doc_ids = load_doc_ids(index_dir)
    index = faiss.read_index(str(index_dir / "faiss.index"))
    model = SentenceTransformer(args.model)

    queries = build_validation_queries(recipes, args.n_queries, args.seed)
    if not queries:
        raise RuntimeError("Could not build validation queries (check ingredients_normalized).")

    save_jsonl(out_dir / "queries.jsonl", queries)
    metrics = evaluate(index, doc_ids, recipes, model, queries, args.retrieve_k)

    with open(out_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("\nValidation saved:")
    print(f"- {out_dir / 'queries.jsonl'}")
    print(f"- {out_dir / 'results.json'}")

    print("\nMetrics:")
    for k, v in metrics.items():
        if k in {"n_queries", "retrieve_k"}:
            continue
        print(f"- {k}: {v:.4f}")
    print(f"- n_queries: {metrics['n_queries']}")
    print(f"- retrieve_k: {metrics['retrieve_k']}")


if __name__ == "__main__":
    main()
