import os
import logging
import json
from tqdm import tqdm

from typing import Optional, List, Dict
from joblib import Parallel, delayed
from ...base_utils import FileProcessor

import nltk
import numpy as np


class OutliersProcessor(FileProcessor):
    """
    This class is used to drop commits with too big or too small diffs and messages.
    Examples with # tokens out of [lower_percentile, upper_percentile] range are considered outliers.

    Args:
        - lower_percentile: which percentile is used as lower bound (should be in (0, 1) range)
        - upper_percentile: which percentile is used as upper bound (should be in (0, 1) range)
        - chunksize: how many examples are processed at once
        - n_workers: how many workers to use when processing smth in parallel
        - diff_upper_bound: optional argument, allows to drop examples with more tokens in diffs than given constant value
            (in contrast with percentile which is calculated on data)
    """

    def __init__(
        self,
        lower_percentile: float,
        upper_percentile: float,
        chunksize: int,
        n_workers: int,
        diff_upper_bound: Optional[int] = None,
    ):
        super(OutliersProcessor, self).__init__(chunksize)
        self._lower_percentile = lower_percentile
        self._upper_percentile = upper_percentile
        self._n_workers = n_workers
        self._diff_upper_bound = diff_upper_bound

        self._ids_to_drop = set()
        self._diff_percentiles = {}
        self._message_percentiles = {}

    @staticmethod
    def _get_n_tokens_str(id: int, string: str) -> str:
        try:
            return f"{id},{len(nltk.wordpunct_tokenize(string))}\n"
        except TypeError:
            logging.warning(f"TypeError with {id}")
            return f"{id},-1\n"

    @staticmethod
    def _mods_to_str(mods: List[Dict[str, str]]) -> Optional[str]:
        """
        This helper method constructs single diff from all file modifications in one commit.

        Currently previous behavior is reproduced by skipping UNKNOWN modifications and adding header
         of the same format I used before.
        """
        commit_diff = []
        for mod in mods:
            if mod["change_type"] == "UNKNOWN":
                continue

            if mod["change_type"] == "ADD":
                file_diff = f"new file {mod['new_path']}\n"
            elif mod["change_type"] == "DELETE":
                file_diff = f"deleted file {mod['old_path']}\n"
            elif mod["change_type"] == "RENAME":
                file_diff = f"rename from {mod['old_path']}\nrename to {mod['new_path']}\n"
            elif mod["change_type"] == "COPY":
                file_diff = f"copy from {mod['old_path']}\ncopy to {mod['new_path']}\n"
            else:
                file_diff = f"{mod['new_path']}\n"

            file_diff += mod["diff"]
            commit_diff.append(file_diff)

        return " ".join(commit_diff)

    def _get_n_tokens(self, input_fname: str, n_tokens_dir: str):
        """
        This method saves the following info to `n_tokens_dir`:
          - number of tokens in diffs for each example
          - number of tokens in messages for each example
        """
        logging.info(f"Starting processing # tokens")

        reader = self._read_input(input_fname)
        for chunk in tqdm(reader, desc=f"Tokenizing {input_fname}", leave=False):
            with Parallel(self._n_workers) as pool:
                # calculate # tokens in diffs from current chuck
                diff_res = pool(
                    delayed(OutliersProcessor._get_n_tokens_str)(
                        item["id"], OutliersProcessor._mods_to_str(item["mods"])
                    )
                    for _, item in chunk[["id", "mods"]].iterrows()
                )
                # calculate # tokens in messages from current chuck
                message_res = pool(
                    delayed(OutliersProcessor._get_n_tokens_str)(item["id"], item["message"])
                    for _, item in chunk[["id", "message"]].iterrows()
                )
            # append results from current chunk to target files
            with open(os.path.join(n_tokens_dir, "n_tokens_diff.txt"), "a", encoding="utf-8") as file:
                file.writelines(diff_res)
            with open(os.path.join(n_tokens_dir, "n_tokens_message.txt"), "a", encoding="utf-8") as file:
                file.writelines(message_res)

        logging.info(f"Finished processing # tokens")

    def _get_percentiles(self, n_tokens_dir: str):
        """
        This method calculates 1%, 5%, 90%, 95%, 99% percentiles of # tokens in diffs and messages
        (and also aggregates ids of examples which produced `TypeError`s: their # tokens is expected to be -1)
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

    def _get_ids_to_drop(self, n_tokens_dir: str):
        """
        This method aggregates ids of examples which:
            - have # tokens in diff or in message out of [self._lower_percentile, self._upper_percentile] range
            - OPTIONAL: have # tokens in diff > self._diff_upper_bound
            - produced `TypeError`s (have -1 as # tokens)
        """
        with open(os.path.join(n_tokens_dir, "n_tokens_diff.txt"), "r") as file:
            for line in file:
                id, n_tokens = (int(i) for i in line.strip().split(","))
                if (
                    n_tokens < self._diff_percentiles[self._lower_percentile]
                    or n_tokens > self._diff_percentiles[self._upper_percentile]
                ) or (self._diff_upper_bound and n_tokens > self._diff_upper_bound):
                    self._ids_to_drop.add(id)

        with open(os.path.join(n_tokens_dir, "n_tokens_message.txt"), "r") as file:
            for line in file:
                id, n_tokens = line.strip().split(",")
                if (
                    int(n_tokens) < self._message_percentiles[self._lower_percentile]
                    or int(n_tokens) > self._message_percentiles[self._upper_percentile]
                ):
                    self._ids_to_drop.add(id)

    def _drop_ids(self, input_fname: str, out_fname: str):
        """
        This method removes marked examples from `input_filename` and saves resulting data to new file `out_fname`.
        """
        logging.info(f"Got {len(self._ids_to_drop)} ids to drop")

        self._prepare_outfile(out_fname)

        reader = self._read_input(input_fname)
        n_dropped = 0
        for chunk in tqdm(reader):
            n_before_drop = len(chunk)
            clean_chunk = chunk.loc[~chunk["id"].isin(self._ids_to_drop)]
            n_dropped += n_before_drop - len(clean_chunk)

            self._append_to_outfile(clean_chunk.to_dict(orient="records"), out_fname)

            logging.info(
                f"{(n_before_drop - len(clean_chunk)) / n_before_drop * 100:.2f}% outliers in last chunk"
                f" ({n_dropped} outliers total)"
            )

    def __call__(
        self,
        input_fname: str,
        out_fname: str,
        n_tokens_dir: str,
        percentile_dir: Optional[str] = None,
    ):
        """
        This method:
         - tokenizes diffs and messages and (optional) calculates percentiles for # of tokens
         - drops examples out of [lower_percentile, upper_percentile] range

         Args:
             - input_fname: path to read input data from
             - out_fname: path to save data without outliers
             - n_tokens_dir: path to save supplementary information like # of tokens for each example and percentiles
             - percentile_dir: (optional) path to directory with already computed percentiles; might be useful for
                dropping outliers from val/test by using percentiles from train, which has much more examples
        """
        self._get_n_tokens(input_fname=input_fname, n_tokens_dir=n_tokens_dir)

        if percentile_dir:
            # read precomputed percentiles
            with open(os.path.join(percentile_dir, "diff.json"), "r") as file:
                self._diff_percentiles = json.load(file, object_hook=lambda d: {float(k): v for k, v in d.items()})
            with open(os.path.join(percentile_dir, "message.json"), "r") as file:
                self._message_percentiles = json.load(file, object_hook=lambda d: {float(k): v for k, v in d.items()})
        else:
            # compute percentiles
            self._get_percentiles(n_tokens_dir=n_tokens_dir)

        self._get_ids_to_drop(n_tokens_dir=n_tokens_dir)
        self._drop_ids(input_fname=input_fname, out_fname=out_fname)
