import pandas as pd
import csv
import argparse
import logging
from tqdm import tqdm
from typing import Set


def get_ids_to_drop(duplicates_filename: str) -> Set[int]:
    ids_to_drop = set()
    with open(duplicates_filename, "r") as f_in:
        for line in f_in:
            _, id1, _, id2 = line.strip().split(",")
            ids_to_drop.add(int(id1))
    return ids_to_drop


def drop_duplicates(csv_filename: str, duplicates_filename: str, output_filename: str, chunksize: int):
    ids_to_drop = get_ids_to_drop(duplicates_filename)
    logging.info(f"Got {len(ids_to_drop)} ids to drop")

    fieldnames = ["id", "author", "date", "hash", "message", "diff", "repo"]
    with open(output_filename, "w") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

    reader = pd.read_csv(csv_filename, chunksize=chunksize)
    n_dropped = 0
    for chunk in tqdm(reader, total=2846334 // chunksize + 1):
        n_before_drop = len(chunk)
        dedup_chunk = chunk.loc[~chunk["id"].isin(ids_to_drop)]
        n_dropped += n_before_drop - len(dedup_chunk)

        dedup_chunk[["id", "author", "date", "hash", "message", "diff", "repo"]].to_csv(
            output_filename, mode="a", index=False, header=False
        )
        logging.info(
            f"{(n_before_drop - len(dedup_chunk)) / n_before_drop * 100:.2f}% duplicates in last chunk"
            f" ({n_dropped} duplicates total)"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="This script drops duplicates",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input_filename", type=str, default="../commits_no_diff_dup.csv", help="path to read .csv file with data")
    parser.add_argument(
        "--duplicates_filename",
        type=str,
        default="../deduplication/results_messages_100_new.pairs",
        help="path to file with duplicates ids",
    )
    parser.add_argument("--n_tokens_dir", type=str,
                        default="../commits_no_dup.csv",
                        help="path to save .csv file with data without duplicates")
    parser.add_argument("--chunksize", type=int, default=1000, help="# of examples to process at one step")
    args = parser.parse_args()

    drop_duplicates(csv_filename=args.csv_filename,
                    duplicates_filename=args.duplicates_filename,
                    output_filename=args.output_filename,
                    chunksize=args.chunksize)
