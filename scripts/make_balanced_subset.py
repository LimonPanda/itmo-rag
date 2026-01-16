import csv
from urllib.parse import urlparse
import argparse

#script for taking only 10k recepies from desired websites 

def normalize_domain(url: str) -> str:
    if not url:
        return ""
    url = url.strip()
    if not url.startswith(("http://", "https://")):
        url = "http://" + url
    netloc = urlparse(url).netloc.lower()
    if netloc.startswith("www."):
        netloc = netloc[4:]
    return netloc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-csv", required=True, help="Path to original large CSV")
    parser.add_argument("--out-csv", required=True, help="Path to output subset CSV")
    parser.add_argument("--url-col", type=int, required=True, help="0-based index of source_url column")
    parser.add_argument("--has-header", action="store_true")
    args = parser.parse_args()

    MAX_DELISH = 3880
    MAX_TASTE = 6120

    delish_count = 0
    taste_count = 0
    written = 0
    read_rows = 0

    with open(args.in_csv, "r", encoding="utf-8", newline="") as fin, open(
        args.out_csv, "w", encoding="utf-8", newline=""
    ) as fout:
        reader = csv.reader(fin)
        writer = csv.writer(fout)

        if args.has_header:
            header = next(reader, None)
            if header is not None:
                writer.writerow(header)

        for row in reader:
            read_rows += 1
            if args.url_col >= len(row):
                continue

            domain = normalize_domain(row[args.url_col])

            if domain == "delish.com" and delish_count < MAX_DELISH:
                writer.writerow(row)
                delish_count += 1
                written += 1

            elif domain == "tasteofhome.com" and taste_count < MAX_TASTE:
                writer.writerow(row)
                taste_count += 1
                written += 1

            if delish_count >= MAX_DELISH and taste_count >= MAX_TASTE:
                break

    print("Done.")
    print(f"Rows read: {read_rows}")
    print(f"Delish recipes: {delish_count}")
    print(f"Taste of Home recipes: {taste_count}")
    print(f"Total written: {written}")
    print(f"Output file: {args.out_csv}")


if __name__ == "__main__":
    main()
