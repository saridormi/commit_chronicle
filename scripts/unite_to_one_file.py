import pandas as pd
import os
import csv
import argparse
import logging
from tqdm import tqdm

logging.basicConfig(
    filename="../logs/one_file.log",
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)


def unite_to_one_file(commit_data_dir, csv_filename, chunksize):
    # write header to csv file
    fieldnames = ["id", "author", "date", "hash", "message", "diff", "repo"]
    with open(csv_filename, "w") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

    cur_idx = 0
    for org in tqdm(os.listdir(commit_data_dir)):
        for repo in os.listdir(os.path.join(commit_data_dir, org)):
            # read data in chunks
            reader = pd.read_csv(os.path.join(commit_data_dir, org, repo), compression="gzip", chunksize=chunksize)
            cur_len = 0
            for i, chunk in enumerate(reader):
                # aggregate № examples so that each example from every repo has an unique id
                chunk["id"] = chunk.index
                chunk["id"] += cur_idx
                chunk["repo"] = f"{org}/{repo.split('.csv.gz')[0]}"
                chunk[["id", "author", "date", "hash", "message", "diff", "repo"]].to_csv(
                    csv_filename, mode="a", index=False, header=False
                )
                cur_len += chunk.shape[0]

            cur_idx += cur_len
            logging.info(
                f"[{org}] {cur_len} examples in {org}/{repo.split('.csv.gz')[0]}, first idx for next repo: {cur_idx}"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="This script processes collected commit data into single .csv file "
                    "and ensures that each example has an unique id.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--commit_data_dir",
        type=str,
        default="../extracted_data_csv",
        help="path to directory with collected commit data",
    )
    parser.add_argument(
        "--input_filename", type=str, default="../commits.csv", help="path for creating new file with all commit data"
    )
    parser.add_argument("--chunksize", type=int, default=1000, help="# of examples to process at one step")
    args = parser.parse_args()

    unite_to_one_file(commit_data_dir=args.commit_data_dir, csv_filename=args.csv_filename, chunksize=args.chunksize)
