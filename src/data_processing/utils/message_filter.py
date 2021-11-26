import re
import csv
import logging
import pandas as pd
from tqdm import tqdm
from typing import Tuple, Optional
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
        x = re.sub(
            r"(signed[-| ]off[-| ]by|co[-| ]authored[-| ]by|also[-| ]by|reviewed[-| ]by|pulled[-| ]by).*?(\n|$)",
            "",
            message,
            flags=re.IGNORECASE,
        )
        x = re.sub(r"Created by MOE:.*?\nMOE_MIGRATED_REVID=.*?($|\n)", "", x)
        x = re.sub(
            r"(fbshipit-source-id|Differential Revision|Change-Id|PiperOrigin-RevId|BAZEL_VERSION_REV_ID).*?($|\n)",
            "",
            x,
            flags=re.IGNORECASE,
        )
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
    def _filter(message: str) -> str:
        try:
            x = MessageFilter._filter_emails(message)
            x = MessageFilter._filter_urls(x)
            x = MessageFilter._filter_issue_ref(x)
            x = MessageFilter._filter_signature(x)
            x = MessageFilter._filter_at_pattern(x)
            x = MessageFilter._filter_sha(x)
            x = x.strip()
        except TypeError:
            x = ""
        return x

    @staticmethod
    def filter(input_filename: str, output_filename: str, chunksize: int):
        fieldnames = ["id", "author", "date", "hash", "message", "diff", "repo"]
        with open(output_filename, "w") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

        logging.info("Starting processing")

        reader = pd.read_csv(input_filename, chunksize=chunksize)
        for chunk in tqdm(reader, desc=f"Processing messages from {input_filename}", leave=False):
            filtered_messages = []
            for _, message in chunk["message"].items():
                if isinstance(message, str) and message.isascii():
                    filtered_messages.append(MessageFilter._filter(message))
                else:
                    filtered_messages.append("")

            chunk["message"] = filtered_messages
            chunk = chunk.loc[chunk.message.str.len() > 0]
            chunk[["id", "author", "date", "hash", "message", "diff", "repo"]].to_csv(
                output_filename, mode="a", index=False, header=False
            )

        logging.info("Finished processing")
