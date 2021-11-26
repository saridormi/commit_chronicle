import os
import json
import jsonlines
import logging
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from typing import List, Dict, Any
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer, PreTrainedTokenizerFast


class TrainingProcessor:
    """
    This classes is used to convert data into necessary format for training:
    1) Construct history for each author
    2) Tokenize diffs and messages
    3) Save everything to format for my training pipeline
    """

    def __init__(
        self,
        diff_tokenizer_name_or_path: str,
        msg_tokenizer_name_or_path: str,
        diff_kwargs: Dict[str, Any],
        msg_kwargs: Dict[str, Any],
    ):
        self._le = LabelEncoder()
        self._test_and_val_authors = set()
        self._diff_tok = PreTrainedTokenizerFast(tokenizer_file=diff_tokenizer_name_or_path)
        self._msg_tok = AutoTokenizer.from_pretrained(msg_tokenizer_name_or_path)
        self._diff_kwargs = diff_kwargs
        self._diff_kwargs.max_length = self._diff_kwargs.max_length - 2
        self._msg_kwargs = msg_kwargs

    def _tokenize_diffs(self, diffs: List[str]) -> List[List[int]]:
        res = []
        for diff in diffs:
            res.append(
                self._diff_tok.convert_tokens_to_ids(["[CLS]"])
                + self._diff_tok(diff, **self._diff_kwargs).input_ids
                + self._diff_tok.convert_tokens_to_ids(["[SEP]"])
            )
        return res

    def _tokenize_messages(self, msgs: List[str]) -> List[List[int]]:
        res = []
        for msg in tqdm(msgs, leave=False):
            res.append(self._msg_tok(msg, **self._msg_kwargs).input_ids)
        return res

    def __call__(self, input_filename: str, output_dir: str, part: str):
        """
        Expects following columns in input file:
        - `author` - some unique string for each number
        - `date` - timestamp for each commit
        - `diff` - diff
        - `message` - message
        """
        logging.info(f"Processing {input_filename}")
        df = pd.read_csv(input_filename)
        df = df.sort_values(by=["author", "date"])
        df["pos_in_history"] = df.groupby("author").cumcount()

        logging.info("Tokenizing messages")
        msg_input_ids = self._tokenize_messages(df["message"].tolist())

        logging.info("Constructing history")
        history = defaultdict(list)
        for msg, id in zip(msg_input_ids, df["author"].tolist()):
            history[id].append(msg)

        logging.info("Saving history")
        with open(os.path.join(output_dir, f"{part}_history.json"), "w") as outfile:
            json.dump(history, outfile)

        if part == "train":
            df = df.sample(frac=1.0, random_state=123)
        df[["diff", "pos_in_history", "author"]].to_csv(os.path.join(output_dir, f"temp_{part}.csv"), index=None)
        reader = pd.read_csv(os.path.join(output_dir, f"temp_{part}.csv"), chunksize=10000)
        open(os.path.join(output_dir, f"{part}.json"), mode="w").close()

        logging.info("Tokenizing diffs")
        for chunk in tqdm(reader, "Tokenizing diffs in chunks"):
            diff_input_ids = self._tokenize_diffs(chunk["diff"].tolist())
            chunk["diff_input_ids"] = diff_input_ids

            with jsonlines.open(os.path.join(output_dir, f"{part}.json"), mode="a") as writer:
                for row in chunk[["diff_input_ids", "pos_in_history", "author"]].to_dict(orient="records"):
                    writer.write(row)
