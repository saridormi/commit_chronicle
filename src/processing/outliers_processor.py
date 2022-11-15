import json
import os
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Set

import jsonlines
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
from typing_extensions import TypedDict

from ..utils import BaseProcessor


class DiffStats(TypedDict):
    id: int
    num_tokens: Optional[int]
    num_chars: Optional[int]
    num_mods: Optional[int]


class MessageStats(TypedDict):
    id: int
    num_tokens: Optional[int]
    num_chars: Optional[int]


class OutliersProcessor(BaseProcessor):
    """This class is used to drop outliers in terms of various diffs/messages statistics.

    Examples with stats out of [lower_percentile, upper_percentile] range are considered outliers.

    Currently, the following stats are considered:

    * number of tokens
    * number of characters
    * number of modified files (only applicable to diffs)

    Args:
        lower_percentile: Percentile to use as a lower bound (should be in [1, 100] range).
        upper_percentile: Percentile to use as an upper bound (should be in [1, 100] range).
        data_format: In which format mined data is saved.
        diff_upper_bound: Specific upper bound for number of tokens in diffs. Optional,
            default value is None, and this step is skipped.
        chunksize: Number of examples to proccess at once (data is read in chunks). Optional, default value is 1000.
        n_workers: Maximum number of concurrently running jobs. Optional, default value is 1 (sequential execution).
        logger_name: Name of logger for this class. Optional, default value is None.
        diff_statistics: You can pass a subset of `["num_tokens", "num_chars", "num_mods"]` here to only consider
          some of them (e.g. if you only want to filter outliers by # tokens, pass `diff_statistics=["num_tokens"]`).
        message_statistics: You can pass a subset of `["num_tokens", "num_chars"]` here to only consider
          some of them (e.g. if you only want to filter outliers by # tokens, pass `message_statistics=["num_tokens"]`).
    """

    def __init__(
        self,
        lower_percentile: int,
        upper_percentile: int,
        data_format: str,
        diff_upper_bound: Optional[int] = None,
        chunksize: Optional[int] = None,
        n_workers: Optional[int] = None,
        logger_name: Optional[str] = None,
        diff_statistics: Optional[Set[str]] = None,
        message_statistics: Optional[Set[str]] = None,
    ):
        super().__init__(chunksize=chunksize, n_workers=n_workers, data_format=data_format, logger_name=logger_name)
        self._lower_percentile = lower_percentile
        self._upper_percentile = upper_percentile
        self._diff_upper_bound = diff_upper_bound

        self._diff_statistics = {"num_tokens", "num_chars", "num_mods"}
        if diff_statistics:
            if any(given_stat not in self._diff_statistics for given_stat in diff_statistics):
                raise ValueError(
                    f"Unexpected key for diff statistics. Only the following keys are allowed: "
                    f"{self._diff_statistics}"
                )
            self._diff_statistics = diff_statistics

        self._message_statistics = {"num_tokens", "num_chars"}
        if message_statistics:
            if any(given_stat not in self._message_statistics for given_stat in message_statistics):
                raise ValueError(
                    f"Unexpected key for message statistics. Only the following keys are allowed: "
                    f"{self._message_statistics}"
                )
            self._message_statistics = message_statistics

        self._ids_to_drop: Set[int] = set()
        self._diff_percentiles: Dict[int, Dict[str, float]] = defaultdict(dict)
        self._message_percentiles: Dict[int, Dict[str, float]] = defaultdict(dict)

    @staticmethod
    def _get_n_tokens_str(string: str) -> int:
        """Splits given string by whitespaces and returns # of tokens."""
        return len(string.split())

    def _get_stats_msg(self, id: int, msg: str) -> MessageStats:
        """
        Tokenizes given message and returns a string
            with id and # of tokens separated by ',' with '\n' at the end.
        """
        try:
            num_tokens = OutliersProcessor._get_n_tokens_str(msg)
            num_chars = len(msg)
        except (TypeError, AttributeError):
            self.logger.error(f"TypeError when tokenizing message from id {id}; `{msg}`")
            num_tokens = None
            num_chars = None
        return {"id": id, "num_tokens": num_tokens, "num_chars": num_chars}

    def _get_stats_mods(self, id: int, mods: List[Dict[str, str]]) -> DiffStats:
        """
        Tokenizes each diff in commit modifications and returns a string
             with id and # of tokens separated by ',' with '\n' at the end.
        """
        try:
            num_tokens = 0
            num_chars = 0
            num_mods = len(mods)

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
                num_tokens += OutliersProcessor._get_n_tokens_str(file_diff)
                num_tokens += OutliersProcessor._get_n_tokens_str(mod["diff"])
                num_chars += len(file_diff)
                num_chars += len(mod["diff"])
        except (TypeError, AttributeError):
            self.logger.error(f"TypeError when tokenizing mods from id {id}; `{mods}`")
            num_tokens = None
            num_chars = None
            num_mods = None

        return {"id": id, "num_tokens": num_tokens, "num_chars": num_chars, "num_mods": num_mods}

    def _get_stats(self, in_fname: str, stats_dir: str) -> None:
        """Processes statistics of diffs and messages and saves to separate files.

        Args:
            in_fname: Path to read input data from.
            stats_dir: Path to directory to save statistics to.
        """
        self.logger.info(f"Starting processing stats from {in_fname}")

        open(os.path.join(stats_dir, "stats_diff.jsonl"), "w").close()
        open(os.path.join(stats_dir, "stats_message.jsonl"), "w").close()

        reader = self._read_input(in_fname)
        for chunk in tqdm(reader, desc=f"Processing stats from {in_fname}", leave=False):
            with Parallel(self._n_workers) as pool:
                # calculate diffs stats from current chuck
                diff_res: List[DiffStats] = pool(
                    delayed(self._get_stats_mods)(item["id"], item["mods"])
                    for _, item in chunk[["id", "mods"]].iterrows()
                )
                # calculate messages stats from current chuck
                message_res: List[MessageStats] = pool(
                    delayed(self._get_stats_msg)(item["id"], item["message"])
                    for _, item in chunk[["id", "message"]].iterrows()
                )
            # append results from current chunk to target files
            with jsonlines.open(os.path.join(stats_dir, "stats_diff.jsonl"), "a") as writer:
                writer.write_all(diff_res)
            with jsonlines.open(os.path.join(stats_dir, "stats_message.jsonl"), "a") as writer:
                writer.write_all(message_res)

        self.logger.info(f"Finished processing # tokens in {in_fname}")

    def _get_percentiles(self, stats_dir: str) -> None:
        """Calculates 1%, 5%, 90%, 95%, 99% percentiles of diffs and messages statistics.

        Args:
            stats_dir: Path to directory to read statistics from.
        """

        with jsonlines.open(os.path.join(stats_dir, "stats_diff.jsonl"), "r") as reader:
            diffs_stats: List[DiffStats] = [
                line for line in reader if all(value is not None for value in line.values())
            ]

        with jsonlines.open(os.path.join(stats_dir, "stats_message.jsonl"), "r") as reader:
            message_stats: List[MessageStats] = [
                line for line in reader if all(value is not None for value in line.values())
            ]

        for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
            for key in self._diff_statistics:
                assert key in ["num_tokens", "num_chars", "num_mods"]
                self._diff_percentiles[p][key] = np.percentile([diff[key] for diff in diffs_stats], p)  # type: ignore[literal-required]

            for key in self._message_statistics:
                assert key in ["num_tokens", "num_chars"]
                self._message_percentiles[p][key] = np.percentile([message[key] for message in message_stats], p)  # type: ignore[literal-required]

        with open(os.path.join(stats_dir, "diff.json"), "w") as file:
            json.dump(self._diff_percentiles, file)
        with open(os.path.join(stats_dir, "message.json"), "w") as file:
            json.dump(self._message_percentiles, file)

    def _read_percentiles(self, percentile_dir: str) -> None:
        """
        Reads precomputed percentiles and ensures that all numerical values are loaded
        as corresponding numerical types, not as strings.

        Args:
            percentile_dir: Path to read precomputed percentiles from.
        """
        with open(os.path.join(percentile_dir, "diff.json"), "r") as file:
            self._diff_percentiles = json.load(file)
            self._diff_percentiles = {
                int(p): {stat: float(self._diff_percentiles[p][stat]) for stat in self._diff_percentiles[p]}
                for p in self._diff_percentiles
            }
        with open(os.path.join(percentile_dir, "message.json"), "r") as file:
            self._message_percentiles = json.load(file)
            self._message_percentiles = {
                int(p): {stat: float(self._message_percentiles[p][stat]) for stat in self._message_percentiles[p]}
                for p in self._message_percentiles
            }

    def _get_ids_to_drop(self, stats_dir: str) -> None:
        """Aggregates ids of examples which either:
            * have statistics of diff/message out of corresponding [lower_percentile, upper_percentile] range
            * produced `TypeError`s (have None for statistics)
            * OPTIONAL: have # tokens in diff > diff_upper_bound

        Args:
            stats_dir: Path to directory to read statistics from.
        """
        self._ids_to_drop = set()

        with jsonlines.open(os.path.join(stats_dir, "stats_diff.jsonl"), "r") as reader:
            for line in reader:
                assert isinstance(line["id"], int)

                for key in self._diff_statistics:
                    assert isinstance(line[key], int) or line[key] is None

                    if (
                        line[key] is None
                        or (
                            line[key] < self._diff_percentiles[self._lower_percentile][key]
                            or line[key] > self._diff_percentiles[self._upper_percentile][key]
                        )
                        or (key == "num_tokens" and self._diff_upper_bound and line[key] > self._diff_upper_bound)
                    ):
                        self._ids_to_drop.add(line["id"])

        with jsonlines.open(os.path.join(stats_dir, "stats_message.jsonl"), "r") as reader:
            for line in reader:
                assert isinstance(line["id"], int)

                for key in self._message_statistics:
                    assert isinstance(line[key], int) or line[key] is None

                    if line[key] is None or (
                        line[key] < self._message_percentiles[self._lower_percentile][key]
                        or line[key] > self._message_percentiles[self._upper_percentile][key]
                    ):
                        self._ids_to_drop.add(line["id"])

    def prepare(self, in_fname: str, stats_dir: str, percentile_dir: Optional[str] = None, **kwargs) -> None:  # type: ignore[override]
        """Processes various statistics of diffs and messages, calculates percentiles and aggregates set of outliers ids.

        Args:
            in_fname: Path to read input data from.
            stats_dir: Path to folder to save supplementary information (# tokens, # characters, # mods, percentiles).
            percentile_dir: Path to directory with already computed percentiles. Optional. Use-case: dropping outliers
                from val/test by percentiles calculated on train.
        """
        self._get_stats(in_fname=in_fname, stats_dir=stats_dir)

        if percentile_dir:
            # read precomputed percentiles (and ensure that percentiles are loaded as numbers, not as strings)
            self._read_percentiles(percentile_dir=percentile_dir)
        else:
            # compute percentiles
            self._get_percentiles(stats_dir=stats_dir)

        self._get_ids_to_drop(stats_dir=stats_dir)
        self.logger.info(f"Got {len(self._ids_to_drop)} outliers to drop")

    def process(self, chunk: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return chunk.loc[~chunk["id"].isin(self._ids_to_drop)]
