import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


def load_doc_ids(index_dir: Path) -> List[str]:
    with open(index_dir / "doc_ids.json", "r", encoding="utf-8") as f:
        return json.load(f)


def load_recipes_map(jsonl_path: Path) -> Dict[str, Dict]:
    """
    Map: recipe_id -> full recipe dict (for building results).
    """
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


def tokenize_ingredients(user_ingredients: str) -> List[str]:
    # Very simple tokenization for MVP: split by commas, lowercase, strip.
    parts = [p.strip().lower() for p in user_ingredients.split(",")]
    return [p for p in parts if p]


def rerank(
    candidates: List[Tuple[str, float]],
    recipes: Dict[str, Dict],
    dish_type: str,
    user_ingredients: List[str],
) -> List[Tuple[str, float, Dict]]:
    """
    Non-linear logic:
    - Base score: FAISS similarity
    - + alpha * ingredient_overlap_ratio
    - + beta if dish_type keyword in title
    - hard filter: must have at least 1 overlap if user provided ingredients
    """
    dish_kw = dish_type.strip().lower()
    user_set = set(user_ingredients)

    alpha = 0.35
    beta = 0.15

    out: List[Tuple[str, float, Dict]] = []
    for rid, sim in candidates:
        r = recipes.get(rid)
        if not r:
            continue

        title = str(r.get("title", "")).lower()
        ingr_norm = [str(x).lower() for x in r.get("ingredients_normalized", [])]
        ingr_set = set(ingr_norm)

        overlap = len(user_set & ingr_set) if user_set else 0
        overlap_ratio = (overlap / max(1, len(user_set))) if user_set else 0.0

        # Hard filter (non-linear): if user specified ingredients, require >=1 overlap
        if user_set and overlap == 0:
            continue

        score = float(sim) + alpha * overlap_ratio
        if dish_kw and dish_kw in title:
            score += beta

        out.append(
            (
                rid,
                score,
                {
                    "title": r.get("title"),
                    "source_url": r.get("source_url"),
                    "matched_ingredients": sorted(list(user_set & ingr_set))[:10],
                    "faiss_similarity": float(sim),
                    "overlap": overlap,
                },
            )
        )

    out.sort(key=lambda x: x[1], reverse=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-dir", default="data/index")
    parser.add_argument("--data", required=True, help="Path to recipes.jsonl")
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--dish-type", required=True, help='Example: "soup"')
    parser.add_argument(
        "--ingredients",
        default="",
        help='Comma-separated ingredients, example: "chicken, carrots, onion"',
    )
    parser.add_argument("--k", type=int, default=25, help="Retrieve top-k from FAISS before rerank")
    parser.add_argument("--topn", type=int, default=5, help="Return top-N after reranking")
    args = parser.parse_args()

    index_dir = Path(args.index_dir)
    index = faiss.read_index(str(index_dir / "faiss.index"))
    doc_ids = load_doc_ids(index_dir)
    recipes = load_recipes_map(Path(args.data))
    model = SentenceTransformer(args.model)

    # Build retrieval query (MVP): dish type + ingredients as text
    q = args.dish_type.strip()
    if args.ingredients.strip():
        q = q + " " + args.ingredients.strip()

    qvec = encode_query(model, q)
    sims, idxs = index.search(qvec, args.k)

    candidates: List[Tuple[str, float]] = []
    for j in range(idxs.shape[1]):
        i = int(idxs[0, j])
        if i < 0 or i >= len(doc_ids):
            continue
        candidates.append((doc_ids[i], float(sims[0, j])))

    user_ingr = tokenize_ingredients(args.ingredients)
    ranked = rerank(candidates, recipes, args.dish_type, user_ingr)[: args.topn]

    print(f"Query: {q}")
    print(f"Candidates retrieved: {len(candidates)} | After filter/rerank: {len(ranked)}")
    print("-" * 80)

    for n, (rid, score, meta) in enumerate(ranked, start=1):
        print(f"{n}. {meta['title']}  | score={score:.4f}")
        print(f"   url: {meta['source_url']}")
        if user_ingr:
            print(f"   matched_ingredients: {meta['matched_ingredients']}")
        print(f"   faiss_similarity={meta['faiss_similarity']:.4f} overlap={meta['overlap']}")
        print("-" * 80)


if __name__ == "__main__":
    main()