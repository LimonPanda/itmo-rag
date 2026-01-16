import argparse
import ast
import csv
import json
from pathlib import Path
from typing import List, Set


#script for cleaning csv dataset and making final processed jsonl file 


COLUMNS = ["id", "title", "ingredients", "instructions", "url", "source", "ner"]


def load_stopwords(path: str) -> Set[str]:
    stop: Set[str] = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            w = line.strip().lower()
            if w and not w.startswith("#"):
                stop.add(w)
    return stop


def safe_parse_list(cell: str) -> List[str]:
    """
    Parse list-like strings stored in CSV, e.g.:
    ["1 cup sugar", "2 eggs"]
    """
    if not cell:
        return []
    try:
        val = ast.literal_eval(cell)
    except Exception:
        return []
    if isinstance(val, list):
        return [str(x) for x in val]
    return []


def join_steps(steps: List[str]) -> str:
    cleaned = [s.strip() for s in steps if s and str(s).strip()]
    return "\n".join(cleaned)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-csv", required=True)
    parser.add_argument("--out-jsonl", required=True)
    parser.add_argument("--stopwords", required=True)
    parser.add_argument("--has-header", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--make-sample", action="store_true")
    parser.add_argument("--sample-size", type=int, default=20)
    args = parser.parse_args()

    stop = load_stopwords(args.stopwords)

    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sample_path = out_path.with_name(out_path.stem + "_sample" + out_path.suffix)

    written = 0
    skipped = 0
    sample_written = 0

    with open(args.in_csv, "r", encoding="utf-8", newline="") as fin, open(
        out_path, "w", encoding="utf-8"
    ) as fout:
        reader = csv.reader(fin)

        if args.has_header:
            next(reader, None)

        for row in reader:
            if args.limit and written >= args.limit:
                break

            if len(row) < len(COLUMNS):
                skipped += 1
                continue

            recipe_id = row[0].strip()
            title = row[1].strip()
            ingredients_raw = safe_parse_list(row[2])
            instructions_steps = safe_parse_list(row[3])
            source_url = row[4].strip()
            ingredients_norm = [
                x.strip().lower() for x in safe_parse_list(row[6])
            ]

            instructions = join_steps(instructions_steps)
            ingredients_norm = [x for x in ingredients_norm if x and x not in stop]

            if not recipe_id or not title or not source_url:
                skipped += 1
                continue
            if not ingredients_raw or not instructions:
                skipped += 1
                continue

            doc = {
                "id": recipe_id,
                "title": title,
                "ingredients_raw": ingredients_raw,
                "ingredients_normalized": ingredients_norm,
                "instructions": instructions,
                "source_url": source_url,
                "text": (
                    f"Title: {title}\n"
                    f"Ingredients: {', '.join(ingredients_norm)}\n"
                    f"Instructions: {instructions[:800]}"
                ),
            }

            fout.write(json.dumps(doc, ensure_ascii=False) + "\n")
            written += 1

            if args.make_sample and sample_written < args.sample_size:
                with open(sample_path, "a", encoding="utf-8") as sf:
                    sf.write(json.dumps(doc, ensure_ascii=False) + "\n")
                sample_written += 1

    print(f"Written JSONL: {written}")
    print(f"Skipped rows: {skipped}")
    if args.make_sample:
        print(f"Sample written: {sample_written} → {sample_path}")
    print(f"Output file: {out_path}")


if __name__ == "__main__":
    main()
