import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


def load_jsonl(path: str) -> List[Dict]:
    items: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def build_corpus(docs: List[Dict]) -> Tuple[List[str], List[str]]:
    ids: List[str] = []
    texts: List[str] = []
    for d in docs:
        doc_id = d.get("id")
        text = d.get("text")
        if not doc_id or not text:
            continue
        ids.append(str(doc_id))
        texts.append(str(text))
    return ids, texts


def save_metadata(out_dir: Path, ids: List[str]) -> None:
    meta_path = out_dir / "doc_ids.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(ids, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        required=True,
        help="Path to processed recipes.jsonl (with 'id' and 'text')",
    )
    parser.add_argument(
        "--out-dir",
        default="data/index",
        help="Directory to save FAISS index and metadata",
    )
    parser.add_argument(
        "--model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="SentenceTransformer model name",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    docs = load_jsonl(args.data)
    ids, texts = build_corpus(docs)

    if not ids:
        raise RuntimeError("No valid documents found: missing 'id' or 'text'.")

    model = SentenceTransformer(args.model)

    embeddings: List[np.ndarray] = []
    for i in tqdm(range(0, len(texts), args.batch_size), desc="Embedding"):
        batch = texts[i : i + args.batch_size]
        vecs = model.encode(
            batch,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        embeddings.append(vecs)

    X = np.vstack(embeddings).astype("float32")
    dim = X.shape[1]

    index = faiss.IndexFlatIP(dim)  # cosine similarity via normalized vectors
    index.add(X)

    faiss.write_index(index, str(out_dir / "faiss.index"))
    save_metadata(out_dir, ids)

    print(f"Documents indexed: {len(ids)}")
    print(f"Embedding dim: {dim}")
    print(f"Saved: {out_dir / 'faiss.index'}")
    print(f"Saved: {out_dir / 'doc_ids.json'}")


if __name__ == "__main__":
    main()