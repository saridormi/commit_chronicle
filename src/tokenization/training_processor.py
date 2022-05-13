import json
import logging
import os
from typing import Any, Dict, List, Optional

import jsonlines
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from ..utils import BaseProcessor


class TrainingProcessor(BaseProcessor):
    """This class is used to convert data into necessary format for training:

    Currently, it includes the following steps:

    * Construct commit message history for each author
    * Tokenize diffs and messages
    * Save everything in format required by commit message completion training pipeline
        (https://github.com/JetBrains-Research/commit_message_generation)

    Args:
        diff_tokenizer_name_or_path: Path to tokenizer file or model name on HuggingFace hub for diffs.
        msg_tokenizer_name_or_path: Path to tokenizer file or model name on HuggingFace hub for messages
        **diff_kwargs: Keyword arguments for diff tokenizer's __call__ method.
        **msg_kwargs: Keyword arguments for message tokenizer's __call__ method.
        data_format: In which format mined data is saved.
        chunksize: Number of examples to proccess at once (data is read in chunks). Optional, default value is 1000.
        n_workers: Maximum number of concurrently running jobs. Optional, default value is 1 (sequential execution).
        logger_name: Name of logger for this class. Optional, default value is None.
    """

    def __init__(
        self,
        diff_tokenizer_name_or_path: str,
        msg_tokenizer_name_or_path: str,
        clean_temp_files: bool,
        diff_kwargs: Dict[str, Any],
        msg_kwargs: Dict[str, Any],
        data_format: str,
        chunksize: Optional[int] = None,
        n_workers: Optional[int] = None,
        logger_name: Optional[str] = None,
    ):
        super().__init__(chunksize=chunksize, logger_name=logger_name, n_workers=n_workers, data_format=data_format)

        if ".json" in diff_tokenizer_name_or_path:
            self._diff_tok = PreTrainedTokenizerFast(tokenizer_file=diff_tokenizer_name_or_path)
        else:
            self._diff_tok = AutoTokenizer.from_pretrained(diff_tokenizer_name_or_path, use_fast=True)

        if ".json" in msg_tokenizer_name_or_path:
            self._msg_tok = PreTrainedTokenizerFast(tokenizer_file=msg_tokenizer_name_or_path)
        else:
            self._msg_tok = AutoTokenizer.from_pretrained(msg_tokenizer_name_or_path, use_fast=True)

        self._clean_temp_files = clean_temp_files
        self._diff_kwargs = diff_kwargs
        self._msg_kwargs = msg_kwargs

    def _tokenize_diffs(self, diffs: List[str]) -> List[List[int]]:
        return self._diff_tok(diffs, **self._diff_kwargs).input_ids

    def _tokenize_messages(self, msgs: List[str]) -> List[List[int]]:
        return self._msg_tok(msgs, **self._msg_kwargs).input_ids

    def _preprocess_data(self, in_fname: str, output_dir: str, part: str) -> None:
        """This method does a several preprocessing steps:

        * adds info about position in history to each example
        * (only for train part) shuffles data
        * saves results to separate file

        Note:
            Assumes that commits from each author are already in correct chronological order,
            which is true for default PyDriller configuration.
        """
        reader = self._read_input(in_fname)
        for i, chunk in enumerate(tqdm(reader, desc=f"Iterating over {part} to save necessary columns to csv")):
            if i == 0:
                chunk[["author", "diff", "message"]].to_csv(
                    os.path.join(output_dir, f"temp_{part}.csv"), index=None, header=True, mode="w"
                )
            else:
                chunk[["author", "diff", "message"]].to_csv(
                    os.path.join(output_dir, f"temp_{part}.csv"), index=None, header=None, mode="a"
                )

        self.logger.info("Reading csv df")
        df = pd.read_csv(os.path.join(output_dir, f"temp_{part}.csv"), lineterminator="\n")
        self.logger.info("Aggregating history positions")
        df["pos_in_history"] = df.groupby("author").cumcount()

        if part == "train":
            self.logger.info("Shuffling train df")
            df = df.sample(frac=1.0, random_state=123)

        self.logger.info("Saving csv df")
        df.to_csv(os.path.join(output_dir, f"temp_{part}.csv"), index=None, header=True, mode="w")

    def _process_messages(self, output_dir: str, part: str) -> None:
        """Tokenizes messages, constructs commit message history for each author and saves to separate file."""
        reader = pd.read_csv(
            os.path.join(output_dir, f"temp_{part}.csv"),
            usecols=["message", "author"],
            dtype={"author": np.int32, "message": str},
            chunksize=self._chunksize,
            lineterminator="\n",
        )

        open(os.path.join(output_dir, f"msgs_{part}.jsonl"), mode="w").close()
        for chunk in tqdm(reader, desc=f"Tokenizing messages in chunks ({self._chunksize} rows)"):
            try:
                chunk["msg_input_ids"] = self._tokenize_messages(chunk["message"].tolist())
            except TypeError:
                chunk["msg_input_ids"] = self._tokenize_messages([str(msg) for msg in chunk["message"].tolist()])
            with jsonlines.open(os.path.join(output_dir, f"msgs_{part}.jsonl"), mode="a") as writer:
                for row in chunk[["msg_input_ids", "author"]].to_dict(orient="records"):
                    writer.write(row)

        df = pd.read_json(os.path.join(output_dir, f"msgs_{part}.jsonl"), orient="records", lines=True)

        history = df[["author", "msg_input_ids"]].groupby("author").agg(list)["msg_input_ids"].to_dict()
        with open(os.path.join(output_dir, f"{part}_history.json"), "w") as outfile:
            json.dump(history, outfile)

    def _process_diffs(self, output_dir: str, part: str) -> None:
        """Tokenizes diffs and saves all necessary information for working with commit message history
        to separate file.
        """
        reader = pd.read_csv(
            os.path.join(output_dir, f"temp_{part}.csv"),
            dtype={"author": np.int32, "pos_in_history": np.int32, "diff": str},
            usecols=["diff", "author", "pos_in_history"],
            chunksize=self._chunksize,
            lineterminator="\n",
        )
        open(os.path.join(output_dir, f"{part}.json"), mode="w").close()

        for chunk in tqdm(reader, desc=f"Tokenizing diffs in chunks ({self._chunksize} rows)"):
            chunk["diff_input_ids"] = self._tokenize_diffs(chunk["diff"].tolist())

            with jsonlines.open(os.path.join(output_dir, f"{part}.json"), mode="a") as writer:
                writer.write_all(chunk[["diff_input_ids", "pos_in_history", "author"]].to_dict(orient="records"))

    def __call__(self, in_fname: str, output_dir: str, part: str, **kwargs) -> None:
        """This method processes data into format required by our pipeline for training & evaluation
        of Transformer models for commit message completion task:
        https://github.com/JetBrains-Research/commit_message_generation

        It expects the following fields in input file:

        * "author"  - commit author
        * "date"    - commit timestamp
        * "diff"    - commit diff
        * "message" - commit message

        Note:
            Assumes that commits from each author are already in correct chronological order,
            which is true for default PyDriller configuration.
        """
        logging.info(f"Start processing {part}")
        self._preprocess_data(in_fname, output_dir, part)
        self._process_messages(output_dir, part)
        self._process_diffs(output_dir, part)
        logging.info(f"Finish processing {part}")

        if self._clean_temp_files:
            os.remove(os.path.join(output_dir, f"temp_{part}.csv"))
            os.remove(os.path.join(output_dir, f"msgs_{part}.jsonl"))
