import argparse
import re
import csv
import logging
import pandas as pd
from tqdm import tqdm
from typing import Optional, Tuple
from joblib import Parallel, delayed


class MessageFilter:
    """
    Class for filtering undesirable patterns from messages via regexes.
    ------------------
    Reused some regexes from https://github.com/Tbabm/PRSummarizer
    Copyright (c) 2019 Zhongxin Liu
    """
    @staticmethod
    def _filter_emails(message: str) -> str:
        return re.sub(r"(?!:^|\s)[\w.-]*@(?=[a-z\d][^.]*\.)[a-z\d.-]*[^.]", "", message)

    @staticmethod
    def _filter_urls(message: str) -> str:
        return re.sub(r"https?://[-a-zA-Z0-9@:%._+~#?&=/]+(?=($|[^-a-zA-Z0-9@:%._+~#?=/]))", "", message)

    @staticmethod
    def _filter_issue_ref(message: str) -> str:
        """
        Filtering the following patterns:
        * #123, [#123], (#123), <#123>
        * GH-123
        * gh-123
        * [#123]
        * CAT-123 (Jira project id)
        """
        x = re.sub("[\[\(<]?#[\d]+[\]\)>]?", "", message)
        x = re.sub("GH-[\d]+", "", x)
        x = re.sub("gh-[\d]+", "", x)
        x = re.sub("([A-Z][A-Z0-9]+-[0-9]+)", "", x)
        return x

    @staticmethod
    def _filter_signature(message: str) -> str:
        """
        Filter various signatures from messages:
        * Not sure about specific tools/repos, but these kinds of signatures appear quite often
            - `Signed-off-by: <username>`
            - `Co-authored-by: <username>`
            - `Also-by: <username>`
            - `Reviewed-by: <username>`
        * https://github.com/google/moe: `Created by MOE: <some link>\nMOE_MIGRATED_REVID=<some number>`
        * https://github.com/facebook/fbshipit:
            - `Differential Revision: <some number>`
            - `Pulled By: <username>`
            - `fbshipit-source-id: <some sha-like string>`
        * https://github.com/google/copybara:
            - `BUG=<some number>`
            - `FIXES=<some number>`
            - `Change-Id: <some sha-like string>`
            - `PiperOrigin-RevId: <some number>`
            - `BAZEL_VERSION_REV_ID: <some number>`
        """
        x = re.sub(r"(signed[-| ]off[-| ]by|co[-| ]authored[-| ]by|also[-| ]by|reviewed[-| ]by|pulled[-| ]by).*?(\n|$)",
                   "", message, flags=re.IGNORECASE)
        x = re.sub(r"Created by MOE:.*?\nMOE_MIGRATED_REVID=.*?($|\n)", "", x)
        x = re.sub(r"(fbshipit-source-id|Differential Revision|Change-Id|PiperOrigin-RevId|BAZEL_VERSION_REV_ID).*?($|\n)",
                   "", x, flags=re.IGNORECASE)
        x = re.sub(r"(BUG=|FIXED=)\d*?($|\n)", "", x)
        return x

    @staticmethod
    def _filter_at_pattern(message: str) -> str:
        return re.sub(r"@\S+", "", message)

    @staticmethod
    def _filter_sha(message: str) -> str:
        x = re.sub(r"(^|\s)[\dA-Fa-f-]{7,}(?=(\s|$))", "", message)
        x = re.sub(r"\bI[0-9a-fA-F]{6,40}\b", "", x)  # gerrit
        return x

    @staticmethod
    def _filter(id: int, message: str) -> Tuple[int, str]:
        try:
            x = MessageFilter._filter_emails(message)
            x = MessageFilter._filter_urls(x)
            x = MessageFilter._filter_issue_ref(x)
            x = MessageFilter._filter_signature(x)
            x = MessageFilter._filter_at_pattern(x)
            x = MessageFilter._filter_sha(x)
            x = x.strip()
        except TypeError:
            logging.error(f"TypeError with {id}")
            x = ""
        return id, x

    @staticmethod
    def _check_ascii(id: int, message: str, diff: str) -> Optional[int]:
        if isinstance(message, str) and isinstance(diff, str) and message.isascii() and diff.isascii():
            return id

    @staticmethod
    def filter(input_filename: str, output_filename: str, chunksize: int):
        fieldnames = ["id", "author", "date", "hash", "message", "diff", "repo"]
        with open(output_filename, "w") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

        logging.info(f"Starting processing")

        reader = pd.read_csv(input_filename, chunksize=chunksize)
        for chunk in tqdm(reader):

            with Parallel(8) as pool:
                filtered_messages = pool(
                    delayed(MessageFilter._filter)(id=id, message=item)
                    for id, item in chunk["message"].items()
                )

            chunk["message"] = pd.Series({i: msg for i, msg in filtered_messages})
            chunk = chunk.loc[chunk.message.str.len() > 0]

            with Parallel(8) as pool:
                ascii_ids = pool(
                    delayed(MessageFilter._check_ascii)(id=id, message=item["message"], diff=item["diff"])
                    for id, item in chunk[["message", "diff"]].iterrows()
                )
            ascii_ids = [x for x in ascii_ids if x is not None]
            chunk = chunk.loc[ascii_ids]
            chunk[["id", "author", "date", "hash", "message", "diff", "repo"]].to_csv(
                output_filename, mode="a", index=False, header=False)

        logging.info(f"Finished processing")


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    parser = argparse.ArgumentParser(
        description="This script calculates percentiles for number of tokens in given .csv file",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input_filename",
        type=str,
        default="../commits_no_outliers_2048.csv",
        help="path to .csv file with data")
    parser.add_argument(
        "--output_filename",
        type=str,
        default="../filtered_commits.csv",
        help="path to save .csv file with filtered commits",
    )
    parser.add_argument("--chunksize", type=int, default=1000, help="# of examples to process at one step")
    args = parser.parse_args()

    MessageFilter.filter(input_filename=args.input_filename, output_filename=args.output_filename, chunksize=args.chunksize)
