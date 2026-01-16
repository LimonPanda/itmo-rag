import argparse
import csv
from urllib.parse import urlparse

#script for taking only necessary domains from main dataset
#for this project delish.com and tasteofhome.com are used
#customizible if needed

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
    parser.add_argument("--in-csv", required=True, help="Path to input CSV")
    parser.add_argument("--out-csv", required=True, help="Path to output CSV")
    parser.add_argument(
        "--url-col",
        type=int,
        required=True,
        help="0-based column index of the source_url field",
    )
    parser.add_argument(
        "--domains",
        nargs="+",
        required=True,
        help="Domains to keep, e.g. tasteofhome.com delish.com",
    )
    parser.add_argument(
        "--has-header",
        action="store_true",
        help="Set if the input CSV has a header row",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="0 = no limit; otherwise stop after writing this many rows",
    )
    args = parser.parse_args()

    keep = {d.lower().lstrip("www.") for d in args.domains}

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
            if domain in keep:
                writer.writerow(row)
                written += 1

                if args.max_rows and written >= args.max_rows:
                    break

    print(f"Read rows: {read_rows}")
    print(f"Written rows: {written}")
    print(f"Output: {args.out_csv}")


if __name__ == "__main__":
    main()