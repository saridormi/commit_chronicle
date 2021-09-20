import argparse
import os
import csv
import logging
import json
import nltk
import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import Optional
from joblib import Parallel, delayed
nltk.download("punkt")


class PercentileProcessor:
    """
    This class tokenizes diffs and messages from data, calculates percentiles for number of tokens
    and drops examples out of [lower_percentile, upper_percentile] range.
    There is also an option to drop examples with more tokens in diffs than specific `diff_upper_bound` value.
    """
    def __init__(self,
                 lower_percentile: float = 0.05,
                 upper_percentile: float = 0.95,
                 diff_upper_bound: Optional[int] = None):
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile
        self.diff_upper_bound = diff_upper_bound

        self._ids_to_drop = set()
        self._diff_percentiles = {}
        self._message_percentiles = {}

    @staticmethod
    def _get_n_tokens_single_ex(id: int, string: str) -> str:
        try:
            return f"{id},{len(nltk.wordpunct_tokenize(string))}\n"
        except TypeError:
            logging.warning(f"TypeError with {id}")
            return f"{id},-1\n"

    @staticmethod
    def _get_n_tokens(input_filename: str, n_tokens_dir: str, chunksize: int):
        """
        This function saves the following info to `n_tokens_dir`:
        - number of tokens in diffs
        - number of tokens in messages
        for each example in `input_filename`
        """
        # make sure to clear target files
        os.makedirs(n_tokens_dir, exist_ok=True)
        open(os.path.join(n_tokens_dir, "n_tokens_diff.txt"), "w", encoding="utf-8").close()
        open(os.path.join(n_tokens_dir, "n_tokens_message.txt"), "w", encoding="utf-8").close()

        logging.info(f"Starting processing # tokens")

        reader = pd.read_csv(input_filename, chunksize=chunksize)
        for chunk in tqdm(reader):
            with Parallel(8) as pool:
                # get # tokens in diffs from current chuck
                diff_res = pool(delayed(PercentileProcessor._get_n_tokens_single_ex)
                                (item["id"], item["diff"]) for _, item in chunk[["id", "diff"]].iterrows())
                # get # tokens in messages from current chuck
                message_res = pool(delayed(PercentileProcessor._get_n_tokens_single_ex)
                                   (item["id"], item["message"]) for _, item in chunk[["id", "message"]].iterrows())
            # append results from current chunk to target files
            with open(os.path.join(n_tokens_dir, "n_tokens_diff.txt"), "a", encoding="utf-8") as file:
                file.writelines(diff_res)
            with open(os.path.join(n_tokens_dir, "n_tokens_message.txt"), "a", encoding="utf-8") as file:
                file.writelines(message_res)

        logging.info(f"Finished processing # tokens")

    def _calculate_percentiles(self, n_tokens_dir: str):
        """
        This function calculates 1%, 5%, 90%, 95%, 99% percentiles of # tokens in diffs and messages
        (and also keeps track of examples which produced `TypeError`s: their # tokens is expected to be -1)
        """
        diff_n_tokens = []
        message_n_tokens = []

        with open(os.path.join(n_tokens_dir, "n_tokens_diff.txt"), "r") as file:
            for line in file:
                id, n_tokens = line.strip().split(",")
                if n_tokens == "-1":
                    self._ids_to_drop.add(id)
                else:
                    diff_n_tokens.append(int(n_tokens))
        with open(os.path.join(n_tokens_dir, "n_tokens_message.txt"), "r") as file:
            for line in file:
                id, n_tokens = line.strip().split(",")
                if n_tokens == "-1":
                    self._ids_to_drop.add(id)
                else:
                    message_n_tokens.append(int(n_tokens))

        for q in [0.01, 0.05, 0.9, 0.95, 0.99]:
            self._diff_percentiles[q] = np.quantile(diff_n_tokens, q)
            self._message_percentiles[q] = np.quantile(message_n_tokens, q)

        with open(os.path.join(n_tokens_dir, "diff.json"), "w") as file:
            json.dump(self._diff_percentiles, file)
        with open(os.path.join(n_tokens_dir, "message.json"), "w") as file:
            json.dump(self._message_percentiles, file)

        logging.info("Finished calculating percentiles")
        logging.info("=== Message ===")
        logging.info(self._message_percentiles)
        logging.info("=== Diff ===")
        logging.info(self._diff_percentiles)

    def _get_ids_to_drop(self, n_tokens_dir: str):
        """
        This function saves ids of examples which:
        * have # tokens in diff or in message are out of (self.lower_percentile, self.upper_percentile) range
        * OPTIONAL: have # tokens in diff > self.diff_upper_bound
        * produced `TypeError`s (have -1 as # tokens)
        """
        with open(os.path.join(n_tokens_dir, "n_tokens_diff.txt"), "r") as file:
            for line in file:
                id, n_tokens = line.strip().split(",")
                if (int(n_tokens) < self._diff_percentiles[self.lower_percentile] or int(n_tokens) > self._diff_percentiles[self.upper_percentile]) \
                        or (self.diff_upper_bound and int(n_tokens) > self.diff_upper_bound):
                    self._ids_to_drop.add(id)

        with open(os.path.join(n_tokens_dir, "n_tokens_message.txt"), "r") as file:
            for line in file:
                id, n_tokens = line.strip().split(",")
                if int(n_tokens) < self._message_percentiles[self.lower_percentile] or int(n_tokens) > self._message_percentiles[self.upper_percentile]:
                    self._ids_to_drop.add(id)

    def _drop_ids(self, input_filename: str, output_filename: str, chunksize: int):
        """
        This function removes examples from `input_filename`
        and saves resuls to separate file `n_tokens_dir`
        """
        logging.info(f"Got {len(self._ids_to_drop)} ids to drop")
        self._ids_to_drop = set([int(i) if i.isdecimal() else i for i in self._ids_to_drop])

        fieldnames = ["id", "author", "date", "hash", "message", "diff", "repo"]
        with open(output_filename, "w") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

        reader = pd.read_csv(input_filename, chunksize=chunksize)
        n_dropped = 0
        for chunk in tqdm(reader):
            n_before_drop = len(chunk)
            clean_chunk = chunk.loc[~chunk["id"].isin(self._ids_to_drop)]
            n_dropped += n_before_drop - len(clean_chunk)

            clean_chunk[["id", "author", "date", "hash", "message", "diff", "repo"]].to_csv(
                output_filename, mode="a", index=False, header=False
            )
            logging.info(
                f"{(n_before_drop - len(clean_chunk)) / n_before_drop * 100:.2f}% outliers in last chunk"
                f" ({n_dropped} outliers total)"
            )

    def __call__(self, input_filename: str, output_filename: str, n_tokens_dir: str, chunksize: int):
        PercentileProcessor._get_n_tokens(input_filename=input_filename, n_tokens_dir=n_tokens_dir, chunksize=chunksize)
        self._calculate_percentiles(n_tokens_dir=n_tokens_dir)
        self._get_ids_to_drop(n_tokens_dir=n_tokens_dir)
        self._drop_ids(input_filename=input_filename, output_filename=output_filename, chunksize=chunksize)


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    parser = argparse.ArgumentParser(
        description="This script calculates percentiles for number of tokens in given .csv file",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input_filename",
        type=str,
        nargs=1,
        default="../commits_no_dup.csv",
        help="path to read .csv file with data")
    parser.add_argument(
        "--output_filename",
        type=str,
        nargs=1,
        default="../commits_no_outliers_2048.csv",
        help="path to save .csv file with processed data",
    )
    parser.add_argument(
        "--n_tokens_dir",
        type=str,
        nargs=1,
        default="../percentile",
        help="path to directory to save processed data",
    )
    parser.add_argument("--chunksize",
                        type=int,
                        nargs=1,
                        default=1000,
                        help="# of examples to process at one step")
    parser.add_argument("--lower_percentile",
                        type=float,
                        nargs="?",
                        default=0.05,
                        help="lower percentile for # tokens in diffs and messages")
    parser.add_argument("--upper_percentile",
                        type=float,
                        nargs="?",
                        default=0.95,
                        help="upper percentile for # tokens in diffs and messages")
    parser.add_argument("--diff_upper_bound",
                        type=int,
                        nargs="?",
                        help="specific upper bound for # tokens in diffs")
    args = parser.parse_args()

    processor = PercentileProcessor(lower_percentile=args.lower_percentile,
                                    upper_percentile=args.upper_percentile,
                                    diff_upper_bound=2048)
    processor(input_filename=args.input_filename,
              n_tokens_dir=args.n_tokens_dir,
              output_filename=args.output_filename,
              chunksize=args.chunksize)
