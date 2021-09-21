import os
import json
import argparse
import logging
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from typing import List
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer


class TrainingProcessor:
    def __init__(self, diff_tokenizer_name_or_path: str, msg_tokenizer_name_or_path: str):
        self._le = LabelEncoder()
        self._test_and_val_authors = set()
        self._diff_tok = AutoTokenizer.from_pretrained(diff_tokenizer_name_or_path)
        self._msg_tok = AutoTokenizer.from_pretrained(msg_tokenizer_name_or_path)

    def _split_by_repos(self, input_filename: str, output_dir: str):
        df = pd.read_csv(input_filename)
        # sort by author and date
        df.sort_values(by=["author", "date"], inplace=True)
        # convert authors from (name, email) pairs to unique ids
        df["author"] = self._le.fit_transform(df["author"])

        train_repos, test_repos = train_test_split(
            df["repo"].unique().tolist(), test_size=0.01, shuffle=True, random_state=123
        )
        train_repos, val_repos = train_test_split(train_repos, test_size=0.01, shuffle=True, random_state=123)

        # split by repos and keep track of val & test authors
        train_df = df.loc[df["repo"].isin(train_repos)]
        val_df = df.loc[df["repo"].isin(val_repos)]
        test_df = df.loc[df["repo"].isin(test_repos)]

        # drop authors which appear in val and test from train
        val_authors = val_df["author"].unique()
        test_authors = test_df["author"].unique()
        self._test_and_val_authors.update(list(test_authors) + list(val_authors))
        train_df = train_df.loc[~train_df["author"].isin(self._test_and_val_authors)]

        for df, part in [(train_df, "TRAIN"), (val_df, "VAL"), (test_df, "TEST")]:
            logging.info(f"=== {part} ===")
            logging.info(f"{len(df)} examples")
            logging.info(f"{df['repo'].nunique()} repos")
            logging.info(f"{df['author'].nunique()} authors")

        train_df[["id", "author", "date", "hash", "message", "diff", "repo"]].to_csv(
            os.path.join(output_dir, "train.csv"), index=False, header=True
        )
        val_df[["id", "author", "date", "hash", "message", "diff", "repo"]].to_csv(
            os.path.join(output_dir, "val.csv"), index=False, header=True
        )
        test_df[["id", "author", "date", "hash", "message", "diff", "repo"]].to_csv(
            os.path.join(output_dir, "test.csv"), index=False, header=True
        )

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

    def __call__(self, input_filename: str, output_dir: str):
        self._split_by_repos(input_filename=input_filename, output_dir=output_dir)
        for part in ["train", "val", "test"]:
            self._process_part(input_filename=os.path.join(output_dir, f"{part}.csv"), output_dir=output_dir, part=part)


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    parser = argparse.ArgumentParser(
        description="This script splits data on train/val/test, tokenizes and saves in a necessary format.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input_filename", type=str, default="../commits_drop_unchanged.csv", help="path to .csv file with data"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="../training_data/",
        help="path to save training data",
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
    proc(input_filename=args.input_filename, output_dir=args.output_dir)
