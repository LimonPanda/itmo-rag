import csv
from collections import Counter
from urllib.parse import urlparse
import argparse

# skript for getting names of webpages and how many recepies we have from each
def get_domain(url: str) -> str:
    if not url:
        return ""
    if not url.startswith(("http://", "https://")):
        url = "http://" + url
    try:
        netloc = urlparse(url).netloc.lower()
        return netloc[4:] if netloc.startswith("www.") else netloc
    except Exception:
        return ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-path", required=True)
    parser.add_argument("--url-col", type=int, required=True, help="0-based index of source_url column")
    parser.add_argument("--limit", type=int, default=0, help="0 = read all rows")
    parser.add_argument("--top", type=int, default=30)
    args = parser.parse_args()

    counter: Counter[str] = Counter()
    rows = 0

    with open(args.csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if args.url_col >= len(row):
                continue
            domain = get_domain(row[args.url_col])
            if domain:
                counter[domain] += 1
            rows += 1
            if args.limit and rows >= args.limit:
                break

    print(f"Rows scanned: {rows}")
    for domain, cnt in counter.most_common(args.top):
        print(f"{cnt:>8}  {domain}")


if __name__ == "__main__":
    main()