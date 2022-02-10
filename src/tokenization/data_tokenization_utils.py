import os
import json
import jsonlines
import logging
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from typing import List, Dict, Any
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer, PreTrainedTokenizer


class TrainingProcessor:
    """
    This class is used to convert data into necessary format for training:
    1) Convert personal information about each author to unique id
    2) Construct history for each author
    3) Tokenize diffs and messages
    4) Save everything in format required by my training pipeline
    """

    def __init__(
        self,
        diff_tokenizer_path: str,
        msg_tokenizer_name: str,
        diff_kwargs: Dict[str, Any],
        msg_kwargs: Dict[str, Any],
    ):
        self._le = LabelEncoder()
        self._diff_tok = PreTrainedTokenizer(tokenizer_file=diff_tokenizer_path)
        self._msg_tok = AutoTokenizer.from_pretrained(msg_tokenizer_name)
        self._diff_kwargs = diff_kwargs
        self._diff_kwargs.max_length = self._diff_kwargs.max_length - 2
        self._msg_kwargs = msg_kwargs

    def _tokenize_diffs(self, diffs: List[str]) -> List[List[int]]:
        res = []
        for diff in diffs:
            res.append(self._diff_tok(diff, **self._diff_kwargs).input_ids)
        return res

    def _tokenize_messages(self, msgs: List[str]) -> List[List[int]]:
        res = []
        for msg in tqdm(msgs, leave=False):
            res.append(self._msg_tok(msg, **self._msg_kwargs).input_ids)
        return res

    def __call__(self, input_fname: str, output_dir: str, part: str):
        """
        Expects following columns in input file:
        - `author` - commit author
        - `date` - commit timestamp
        - `mods` - commit modifications
        - `message` - commit message
        """
        logging.info(f"Processing {input_fname}")
        df = pd.read_json(
            input_fname,
            orient="records",
            lines=True,
        )
        df = df.sort_values(by=["author", "date"])
        df["pos_in_history"] = df.groupby("author").cumcount()
        df["author"] = self._le.fit_transform(df["author"])
        with open(os.path.join(output_dir, "authors_mapping.json"), "w") as outfile:
            json.dump(dict(zip(self._le.classes_, self._le.transform(self._le.classes_))), outfile)

        logging.info("Tokenizing messages")
        msg_input_ids = self._tokenize_messages(df["message"].tolist())

        logging.info("Constructing history")
        history = defaultdict(list)
        for msg, id in zip(msg_input_ids, df["author"].tolist()):
            history[id].append(msg)

        logging.info("Saving history")
        with open(os.path.join(output_dir, f"{part}_history.json"), "w") as outfile:
            json.dump(history, outfile)

        # shuffle train
        if part == "train":
            df = df.sample(frac=1.0, random_state=123)

        df[["mods", "pos_in_history", "author"]].to_csv(os.path.join(output_dir, f"temp_{part}.csv"), index=None)
        reader = pd.read_csv(os.path.join(output_dir, f"temp_{part}.csv"), chunksize=10000)
        open(os.path.join(output_dir, f"{part}.json"), mode="w").close()

        logging.info("Tokenizing diffs")
        for chunk in tqdm(reader, "Tokenizing diffs in chunks"):
            diff_input_ids = self._tokenize_diffs(chunk["mods"].tolist())
            chunk["diff_input_ids"] = diff_input_ids

            with jsonlines.open(os.path.join(output_dir, f"{part}.json"), mode="a") as writer:
                for row in chunk[["diff_input_ids", "pos_in_history", "author"]].to_dict(orient="records"):
                    writer.write(row)
