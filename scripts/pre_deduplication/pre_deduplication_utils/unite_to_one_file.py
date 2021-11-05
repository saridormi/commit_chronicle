import pandas as pd
import os
import csv
import logging
from tqdm import tqdm


def unite_to_one_file(input_dir, output_filename, chunksize):
    # write header to csv file
    fieldnames = ["id", "author", "date", "hash", "message", "diff", "repo"]
    with open(output_filename, "w") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

    cur_idx = 0
    for org in tqdm(os.listdir(input_dir)):
        for repo in os.listdir(os.path.join(input_dir, org)):
            # read data in chunks
            reader = pd.read_csv(
                os.path.join(input_dir, org, repo, "commits.csv.gz"), compression="gzip", chunksize=chunksize
            )
            cur_len = 0
            for i, chunk in enumerate(reader):
                # aggregate № examples so that each example from every repo has an unique id
                chunk["id"] = chunk.index
                chunk["id"] += cur_idx
                chunk["repo"] = f"{org}/{repo}"
                chunk[["id", "author", "date", "hash", "message", "diff", "repo"]].to_csv(
                    output_filename, mode="a", index=False, header=False
                )
                cur_len += chunk.shape[0]

            cur_idx += cur_len
            logging.info(f"[{org}] {cur_len} examples in {org}/{repo}, first idx for next repo: {cur_idx}")
