import json
import logging
import os
from typing import Any, Dict, List, Optional, Union

import jsonlines
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer

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
        data_format: str,
        diff_tokenizer_name_or_path: Optional[str] = None,
        msg_tokenizer_name_or_path: Optional[str] = None,
        diff_kwargs: Optional[Dict[str, Any]] = None,
        msg_kwargs: Optional[Dict[str, Any]] = None,
        chunksize: Optional[int] = None,
        n_workers: Optional[int] = None,
        logger_name: Optional[str] = None,
    ):
        super().__init__(chunksize=chunksize, logger_name=logger_name, n_workers=n_workers, data_format=data_format)

        self._diff_tok = None
        if diff_tokenizer_name_or_path:
            self._diff_tok = AutoTokenizer.from_pretrained(diff_tokenizer_name_or_path)

        self._msg_tok = None
        if msg_tokenizer_name_or_path:
            self._msg_tok = AutoTokenizer.from_pretrained(msg_tokenizer_name_or_path)

        self._diff_kwargs = diff_kwargs
        self._msg_kwargs = msg_kwargs

    def _tokenize_diffs(self, diffs: List[str]) -> List[List[int]]:
        return self._diff_tok(diffs, **self._diff_kwargs).input_ids

    def _tokenize_messages(self, msgs: List[str]) -> List[List[int]]:
        return self._msg_tok(msgs, **self._msg_kwargs).input_ids

    @staticmethod
    def _get_diff_from_mods(commits: List[Dict[str, Union[str, List[str]]]], line_sep: str) -> List[str]:

        diffs: List[str] = []

        for mods in commits:
            diff = ""
            for mod in mods:
                if mod["change_type"] == "UNKNOWN":
                    continue
                elif mod["change_type"] == "ADD":
                    file_diff = f"new file {mod['new_path']}"
                elif mod["change_type"] == "DELETE":
                    file_diff = f"deleted file {mod['old_path']}"
                elif mod["change_type"] == "RENAME":
                    file_diff = f"rename from {mod['old_path']}{line_sep}rename to {mod['new_path']}"
                elif mod["change_type"] == "COPY":
                    file_diff = f"copy from {mod['old_path']}{line_sep}copy to {mod['new_path']}"
                else:
                    file_diff = f"{mod['new_path']}"
                diff += file_diff + line_sep + mod["diff"] + line_sep
            diffs.append(diff)
        return diffs

    def _preprocess_data(self, in_fname: str, output_dir: str, part: str, temp_dir: str, line_sep: str) -> None:
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
            chunk["diff"] = TrainingProcessor._get_diff_from_mods(chunk["mods"].tolist(), line_sep=line_sep)
            if i == 0:
                chunk[["author", "diff", "message"]].to_csv(
                    os.path.join(temp_dir, f"temp_{part}.csv"), index=None, header=True, mode="w"
                )
            else:
                chunk[["author", "diff", "message"]].to_csv(
                    os.path.join(temp_dir, f"temp_{part}.csv"), index=None, header=None, mode="a"
                )

        self.logger.info("Reading csv df")
        df = pd.read_csv(os.path.join(temp_dir, f"temp_{part}.csv"), lineterminator="\n")
        self.logger.info("Aggregating history positions")
        df["pos_in_history"] = df.groupby("author").cumcount()

        if part == "train":
            self.logger.info("Shuffling train df")
            df = df.reset_index(drop=False)
            df = df.sample(frac=1.0, random_state=123)
            with open(os.path.join(temp_dir, f"{part}_index.csv"), "w") as f:
                f.writelines(f"{index}\n" for index in df["index"].tolist())

        self.logger.info("Saving csv df")
        df.to_csv(os.path.join(temp_dir, f"temp_{part}.csv"), index=None, header=True, mode="w")

    def _process_messages(self, output_dir: str, part: str, temp_dir: str) -> None:
        """Tokenizes messages, constructs commit message history for each author and saves to separate file."""
        reader = pd.read_csv(
            os.path.join(temp_dir, f"temp_{part}.csv"),
            usecols=["message", "author"],
            dtype={"author": np.int32, "message": str},
            chunksize=self._chunksize,
            lineterminator="\n",
        )

        open(os.path.join(temp_dir, f"msgs_{part}.jsonl"), mode="w").close()
        for chunk in tqdm(reader, desc=f"Tokenizing messages in chunks ({self._chunksize} rows)"):
            try:
                chunk["msg_input_ids"] = self._tokenize_messages(chunk["message"].tolist())
            except TypeError:
                chunk["msg_input_ids"] = self._tokenize_messages([str(msg) for msg in chunk["message"].tolist()])
            with jsonlines.open(os.path.join(temp_dir, f"msgs_{part}.jsonl"), mode="a") as writer:
                for row in chunk[["msg_input_ids", "author"]].to_dict(orient="records"):
                    writer.write(row)

        df = pd.read_json(os.path.join(temp_dir, f"msgs_{part}.jsonl"), orient="records", lines=True)
        history = df[["author", "msg_input_ids"]].groupby("author").agg(list)["msg_input_ids"].to_dict()
        with open(os.path.join(output_dir, "messages", f"{part}_history.json"), "w") as outfile:
            json.dump(history, outfile)

    def _process_diffs(self, output_dir: str, part: str, temp_dir: str) -> None:
        """Tokenizes diffs and saves all necessary information for working with commit message history
        to separate file.
        """
        reader = pd.read_csv(
            os.path.join(temp_dir, f"temp_{part}.csv"),
            dtype={"author": np.int32, "pos_in_history": np.int32, "diff": str},
            usecols=["diff", "author", "pos_in_history"],
            chunksize=self._chunksize,
            lineterminator="\n",
        )
        open(os.path.join(output_dir, "diffs", f"{part}.json"), mode="w").close()
        for chunk in tqdm(reader, desc=f"Tokenizing diffs in chunks ({self._chunksize} rows)"):
            chunk["diff_input_ids"] = self._tokenize_diffs(chunk["diff"].tolist())

            with jsonlines.open(os.path.join(output_dir, "diffs", f"{part}.json"), mode="a") as writer:
                writer.write_all(chunk[["diff_input_ids", "pos_in_history", "author"]].to_dict(orient="records"))

    def process_single_col(
        self,
        in_fname: str,
        output_dir: str,
        part: str,
        diffs_or_messages: str,
        line_sep: str,
        preprocess_data: Optional[bool] = True,
        clean_temp_files: Optional[bool] = False,
        temp_dir: [Optional[str]] = None,
        **kwargs,
    ) -> None:
        """This method processes only diffs or only messages into format required by our pipeline for training & evaluation
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
        logging.info(f"Start processing {diffs_or_messages} in {part}")

        if not temp_dir:
            temp_dir = output_dir

        if preprocess_data:
            self._preprocess_data(in_fname, output_dir, part, temp_dir, line_sep=line_sep)
        assert f"temp_{part}.csv" in os.listdir(temp_dir)

        if diffs_or_messages == "diffs":
            self._process_diffs(output_dir, part, temp_dir)
        elif diffs_or_messages == "messages":
            self._process_messages(output_dir, part, temp_dir)

        logging.info(f"Finish processing {diffs_or_messages} in {part}")

        if clean_temp_files:
            os.remove(os.path.join(temp_dir, f"temp_{part}.csv"))
            os.remove(os.path.join(temp_dir, f"msgs_{part}.jsonl"))

    @staticmethod
    def truncate_diffs(in_fname: str, output_dir: str, part: str, context_len: int):
        with jsonlines.open(in_fname, mode="r") as reader:
            with jsonlines.open(os.path.join(output_dir, f"{part}.json"), mode="w") as writer:
                for example in tqdm(reader):
                    if len(example["diff_input_ids"]) > context_len:
                        example["diff_input_ids"] = example["diff_input_ids"][:context_len]
                        example["truncated"] = True
                    else:
                        example["truncated"] = False
                    writer.write(example)

    def __call__(
        self,
        in_fname: str,
        output_dir: str,
        part: str,
        line_sep: str,
        clean_temp_files: Optional[bool] = False,
        temp_dir: [Optional[str]] = None,
        **kwargs,
    ) -> None:
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
        if not temp_dir:
            temp_dir = output_dir

        logging.info(f"Start processing {part}")
        self._preprocess_data(in_fname, output_dir, part, temp_dir, line_sep=line_sep)
        self._process_messages(output_dir, part, temp_dir)
        self._process_diffs(output_dir, part, temp_dir)
        logging.info(f"Finish processing {part}")

        if clean_temp_files:
            os.remove(os.path.join(temp_dir, f"temp_{part}.csv"))
            os.remove(os.path.join(temp_dir, f"msgs_{part}.jsonl"))
