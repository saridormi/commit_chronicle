import os
import json
import argparse
import logging
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from typing import List
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer


class TrainingProcessor:
    def __init__(self, diff_tokenizer_name_or_path: str, msg_tokenizer_name_or_path: str):
        self._le = LabelEncoder()
        self._test_and_val_authors = set()
        self._diff_tok = AutoTokenizer.from_pretrained(diff_tokenizer_name_or_path)
        self._msg_tok = AutoTokenizer.from_pretrained(msg_tokenizer_name_or_path)

    def _tokenize_diffs(self, diffs: List[str]) -> List[List[int]]:
        res = []
        for diff in tqdm(diffs):
            res.append(
                self._diff_tok(
                    diff, truncation=True, add_special_tokens=True, max_length=500, return_attention_mask=False
                ).input_ids
            )
        return res

    def _tokenize_messages(self, msgs: List[str]) -> List[List[int]]:
        res = []
        for msg in tqdm(msgs):
            res.append(self._msg_tok(msg, truncation=True, return_attention_mask=False).input_ids)
        return res

    def _process_part(self, input_filename: str, output_dir: str, part: str):
        logging.info(f"Processing {part}")
        df = pd.read_csv(input_filename)
        df["pos_in_history"] = df.groupby("author").cumcount()

        logging.info("Tokenizing diffs")
        diff_input_ids = self._tokenize_diffs(df["diff"].tolist())
        logging.info("Tokenizing messages")
        msg_input_ids = self._tokenize_messages(df["message"].tolist())

        logging.info("Constructing history")
        history = defaultdict(list)
        for msg, id in zip(msg_input_ids, df["author"].tolist()):
            history[id].append(msg)

        logging.info("Saving history")
        with open(os.path.join(output_dir, f"{part}_history.json"), "w") as outfile:
            json.dump(history, outfile)

        logging.info("Saving data")
        df["diff_input_ids"] = diff_input_ids
        # shuffle train
        if part == "train":
            df = df.sample(frac=1.0, random_state=123)
        df[["diff_input_ids", "pos_in_history", "author"]].to_json(
            os.path.join(output_dir, f"{part}.json"), lines=True, orient="records"
        )

    def __call__(self, input_root_dir: str, output_dir: str):
        for part in ["train", "val", "test"]:
            self._process_part(
                input_filename=os.path.join(input_root_dir, f"{part}.csv"), output_dir=output_dir, part=part
            )


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    parser = argparse.ArgumentParser(
        description="This script tokenizes and saves train/val/test parts of dataset in a necessary format.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input_root_dir",
        type=str,
        help="path to directory with input data",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="path to save processed data",
    )
    parser.add_argument(
        "--diff_tok",
        type=str,
        default="microsoft/codebert-base",
        help="name or path for diff tokenizer",
    )
    parser.add_argument(
        "--msg_tok",
        type=str,
        default="distilgpt2",
        help="name or path for message tokenizer",
    )
    args = parser.parse_args()
    proc = TrainingProcessor(diff_tokenizer_name_or_path=args.diff_tok, msg_tokenizer_name_or_path=args.msg_tok)
    proc(input_root_dir=args.input_root_dir, output_dir=args.output_dir)
