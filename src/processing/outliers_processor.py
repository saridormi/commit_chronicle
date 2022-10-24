import json
import os
from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from ..utils import BaseProcessor


class OutliersProcessor(BaseProcessor):
    """This class is used to drop commits with too long or too short diffs and messages.

    Examples with # tokens out of [lower_percentile, upper_percentile] range are considered outliers.

    Args:
        lower_percentile: Percentile to use as a lower bound (should be in (0, 1) range).
        upper_percentile: Percentile to use as an upper bound (should be in (0, 1) range).
        data_format: In which format mined data is saved.
        diff_upper_bound: Specific upper bound for number of tokens in diffs. Optional,
            default value is None, and this step is skipped.
        chunksize: Number of examples to proccess at once (data is read in chunks). Optional, default value is 1000.
        n_workers: Maximum number of concurrently running jobs. Optional, default value is 1 (sequential execution).
        logger_name: Name of logger for this class. Optional, default value is None.
    """

    def __init__(
        self,
        lower_percentile: float,
        upper_percentile: float,
        data_format: str,
        diff_upper_bound: Optional[int] = None,
        chunksize: Optional[int] = None,
        n_workers: Optional[int] = None,
        logger_name: Optional[str] = None,
    ):
        super().__init__(chunksize=chunksize, n_workers=n_workers, data_format=data_format, logger_name=logger_name)
        self._lower_percentile = lower_percentile
        self._upper_percentile = upper_percentile
        self._diff_upper_bound = diff_upper_bound

        self._ids_to_drop: Set[int] = set()
        self._diff_percentiles: Dict[float, float] = {}
        self._message_percentiles: Dict[float, float] = {}

    def _get_n_tokens_str(self, string: str) -> int:
        """Splits given string by whitespaces and returns # of tokens."""
        return len(string.split())

    def _get_n_tokens_msg(self, id: int, msg: str) -> str:
        """
        Tokenizes given message and returns a string
            with id and # of tokens separated by ',' with '\n' at the end.
        """
        try:
            return f"{id},{self._get_n_tokens_str(msg)}\n"
        except TypeError as e:
            self.logger.warning(f"TypeError {e} with {id}")
            return f"{id},-1\n"

    def _get_n_tokens_mods(self, id: int, mods: List[Dict[str, str]]) -> str:
        """
        Tokenizes each diff in commit modifications and returns a string
             with id and # of tokens separated by ',' with '\n' at the end.
        """
        try:
            n_tokens = 0
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
                n_tokens += self._get_n_tokens_str(file_diff)
                n_tokens += self._get_n_tokens_str(mod["diff"])
            return f"{id},{n_tokens}\n"
        except TypeError as e:
            self.logger.warning(f"TypeError {e} with {id}")
            return f"{id},-1\n"

    def _get_n_tokens(self, in_fname: str, n_tokens_dir: str) -> None:
        """Tokenizes diff and messages and saves # of tokens in diffs and messages to separate files.

        Args:
            in_fname: Path to read input data from.
            n_tokens_dir: Path to directory to save # of tokens to.
        """
        self.logger.info(f"Starting processing # tokens in {in_fname}")

        open(os.path.join(n_tokens_dir, "n_tokens_diff.txt"), "w", encoding="utf-8").close()
        open(os.path.join(n_tokens_dir, "n_tokens_message.txt"), "w", encoding="utf-8").close()

        reader = self._read_input(in_fname)
        for chunk in tqdm(reader, desc=f"Tokenizing {in_fname}", leave=False):
            with Parallel(self._n_workers) as pool:
                # calculate # tokens in diffs from current chuck
                diff_res = pool(
                    delayed(self._get_n_tokens_mods)(item["id"], item["mods"])
                    for _, item in chunk[["id", "mods"]].iterrows()
                )
                # calculate # tokens in messages from current chuck
                message_res = pool(
                    delayed(self._get_n_tokens_msg)(item["id"], item["message"])
                    for _, item in chunk[["id", "message"]].iterrows()
                )
            # append results from current chunk to target files
            with open(os.path.join(n_tokens_dir, "n_tokens_diff.txt"), "a", encoding="utf-8") as file:
                file.writelines(diff_res)
            with open(os.path.join(n_tokens_dir, "n_tokens_message.txt"), "a", encoding="utf-8") as file:
                file.writelines(message_res)

        self.logger.info(f"Finished processing # tokens in {in_fname}")

    def _get_percentiles(self, n_tokens_dir: str) -> None:
        """Calculates 1%, 5%, 90%, 95%, 99% percentiles of # tokens in diffs and messages.

        Args:
            n_tokens_dir: Path to directory to read # of tokens from.
        """
        diff_n_tokens = []
        with open(os.path.join(n_tokens_dir, "n_tokens_diff.txt"), "r") as file:
            for line in file:
                id, n_tokens = (int(i) for i in line.strip().split(","))
                if n_tokens != -1:
                    diff_n_tokens.append(n_tokens)

        message_n_tokens = []
        with open(os.path.join(n_tokens_dir, "n_tokens_message.txt"), "r") as file:
            for line in file:
                id, n_tokens = (int(i) for i in line.strip().split(","))
                if n_tokens != -1:
                    message_n_tokens.append(n_tokens)

        for q in [0.01, 0.05, 0.9, 0.95, 0.99]:
            self._diff_percentiles[q] = np.quantile(diff_n_tokens, q)
            self._message_percentiles[q] = np.quantile(message_n_tokens, q)

        with open(os.path.join(n_tokens_dir, "diff.json"), "w") as file:
            json.dump(self._diff_percentiles, file)
        with open(os.path.join(n_tokens_dir, "message.json"), "w") as file:
            json.dump(self._message_percentiles, file)

    def _get_ids_to_drop(self, n_tokens_dir: str) -> None:
        """Aggregates ids of examples which either:
            * have # tokens in diff or in message out of [lower_percentile, upper_percentile] range
            * produced `TypeError`s (have -1 as # tokens)
            * OPTIONAL: have # tokens in diff > diff_upper_bound

        Args:
            n_tokens_dir: path to directory to read # of tokens from
        """
        self._ids_to_drop = set()

        with open(os.path.join(n_tokens_dir, "n_tokens_diff.txt"), "r") as file:
            for line in file:
                id, n_tokens = (int(i) for i in line.strip().split(","))
                if (
                    (n_tokens == -1)
                    or (
                        n_tokens < self._diff_percentiles[self._lower_percentile]
                        or n_tokens > self._diff_percentiles[self._upper_percentile]
                    )
                    or (self._diff_upper_bound and n_tokens > self._diff_upper_bound)
                ):
                    self._ids_to_drop.add(id)

        with open(os.path.join(n_tokens_dir, "n_tokens_message.txt"), "r") as file:
            for line in file:
                id, n_tokens = (int(i) for i in line.strip().split(","))
                if (n_tokens == -1) or (
                    n_tokens < self._message_percentiles[self._lower_percentile]
                    or n_tokens > self._message_percentiles[self._upper_percentile]
                ):
                    self._ids_to_drop.add(id)

    def prepare(self, in_fname: str, n_tokens_dir: str, percentile_dir: Optional[str] = None, **kwargs) -> None:  # type: ignore[override]
        """Tokenizes diffs and messages and calculates percentiles for # of tokens.

        Args:
            in_fname: Path to read input data from.
            n_tokens_dir: Path to folder to save supplementary information (# of tokens and percentiles).
            percentile_dir: Path to directory with already computed percentiles. Optional. Use-case: dropping outliers
                from val/test by percentiles calculated on train.
        """
        self._get_n_tokens(in_fname=in_fname, n_tokens_dir=n_tokens_dir)

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
        self.logger.info(f"Got {len(self._ids_to_drop)} outliers ids to drop")

    def process(self, chunk: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return chunk.loc[~chunk["id"].isin(self._ids_to_drop)]
