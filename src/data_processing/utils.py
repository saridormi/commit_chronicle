import re
import os
import json
import hashlib
import jsonlines
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List, Dict, Tuple, Union, Optional, Set
from joblib import Parallel, delayed
from collections import Counter
from ..base_utils import BaseProcessor


class OutliersProcessor(BaseProcessor):
    """
    This class is used to drop commits with too long or too short diffs and messages.
    Examples with # tokens out of [lower_percentile, upper_percentile] range are considered outliers.

    Args:
        - lower_percentile: which percentile is used as lower bound (should be in (0, 1) range)
        - upper_percentile: which percentile is used as upper bound (should be in (0, 1) range)
        - chunksize: how many examples are processed at once
        - n_workers: how many workers are used to process smth in parallel
        - diff_upper_bound: optional argument, allows to drop examples with more tokens in diffs than given constant value
            (in contrast with percentile, which is calculated on data)
    """

    def __init__(
        self,
        lower_percentile: float,
        upper_percentile: float,
        chunksize: int,
        n_workers: int,
        diff_upper_bound: Optional[int] = None,
        logger_name: Optional[str] = None,
    ):
        super(OutliersProcessor, self).__init__(chunksize=chunksize, logger_name=logger_name, n_workers=n_workers)
        self._lower_percentile = lower_percentile
        self._upper_percentile = upper_percentile
        self._diff_upper_bound = diff_upper_bound

        self._ids_to_drop: Set[int] = set()
        self._diff_percentiles: Dict[float, float] = {}
        self._message_percentiles: Dict[float, float] = {}

    def _get_n_tokens_str(self, id: int, string: str) -> str:
        """
        This method tokenizes given string and returns string with id and # of tokens.

        Currently tokenization is performed simply by splitting on whitespaces.
        """
        try:
            return f"{id},{len(string.split())}\n"
        except TypeError:
            self.logger.warning(f"TypeError with {id}")
            return f"{id},-1\n"

    def _mods_to_str(self, mods: List[Dict[str, str]]) -> Optional[str]:
        """
        This method constructs single diff from all file modifications in one commit.

        Currently previous behavior is reproduced by skipping UNKNOWN modifications and adding header
         of the same format I used before, but I think it would be more reasonable to not consider filenames at all at
         this stage.
        """
        commit_diff = []
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

            file_diff += mod["diff"]
            commit_diff.append(file_diff)

        return " ".join(commit_diff)

    def _get_n_tokens(self, in_fname: str, n_tokens_dir: str):
        """
        This method tokenizes diff and messages and saves # of tokens in diffs and messages to separate files.

        Args:
            - in_fname: path to read input data from
            - n_tokens_dir: path to directory to save # of tokens to
        """
        self.logger.info(f"Starting processing # tokens in {in_fname}")

        reader = self._read_input(in_fname)
        for chunk in tqdm(reader, desc=f"Tokenizing {in_fname}", leave=False):
            with Parallel(self._n_workers) as pool:
                # calculate # tokens in diffs from current chuck
                diff_res = pool(
                    delayed(self._get_n_tokens_str)(item["id"], self._mods_to_str(item["mods"]))
                    for _, item in chunk[["id", "mods"]].iterrows()
                )
                # calculate # tokens in messages from current chuck
                message_res = pool(
                    delayed(self._get_n_tokens_str)(item["id"], item["message"])
                    for _, item in chunk[["id", "message"]].iterrows()
                )
            # append results from current chunk to target files
            with open(os.path.join(n_tokens_dir, "n_tokens_diff.txt"), "a", encoding="utf-8") as file:
                file.writelines(diff_res)
            with open(os.path.join(n_tokens_dir, "n_tokens_message.txt"), "a", encoding="utf-8") as file:
                file.writelines(message_res)

        self.logger.info(f"Finished processing # tokens in {in_fname}")

    def _get_percentiles(self, n_tokens_dir: str):
        """
        This method calculates 1%, 5%, 90%, 95%, 99% percentiles of # tokens in diffs and messages
        (and also aggregates ids of examples which produced `TypeError`s: their # tokens is expected to be -1).

        Args:
            - n_tokens_dir: path to directory to read # of tokens from
        """
        diff_n_tokens = []
        with open(os.path.join(n_tokens_dir, "n_tokens_diff.txt"), "r") as file:
            for line in file:
                id, n_tokens = (int(i) for i in line.strip().split(","))
                if n_tokens == -1:
                    self._ids_to_drop.add(id)
                else:
                    diff_n_tokens.append(n_tokens)

        message_n_tokens = []
        with open(os.path.join(n_tokens_dir, "n_tokens_message.txt"), "r") as file:
            for line in file:
                id, n_tokens = (int(i) for i in line.strip().split(","))
                if n_tokens == -1:
                    self._ids_to_drop.add(id)
                else:
                    message_n_tokens.append(n_tokens)

        for q in [0.01, 0.05, 0.9, 0.95, 0.99]:
            self._diff_percentiles[q] = np.quantile(diff_n_tokens, q)
            self._message_percentiles[q] = np.quantile(message_n_tokens, q)

        with open(os.path.join(n_tokens_dir, "diff.json"), "w") as file:
            json.dump(self._diff_percentiles, file)
        with open(os.path.join(n_tokens_dir, "message.json"), "w") as file:
            json.dump(self._message_percentiles, file)

    def _get_ids_to_drop(self, n_tokens_dir: str):
        """
        This method aggregates ids of examples which:
            - have # tokens in diff or in message out of [self._lower_percentile, self._upper_percentile] range
            - OPTIONAL: have # tokens in diff > self._diff_upper_bound
            - produced `TypeError`s (have -1 as # tokens)

        Args:
            - n_tokens_dir: path to directory to read # of tokens from
        """
        with open(os.path.join(n_tokens_dir, "n_tokens_diff.txt"), "r") as file:
            for line in file:
                id, n_tokens = (int(i) for i in line.strip().split(","))
                if (
                    n_tokens < self._diff_percentiles[self._lower_percentile]
                    or n_tokens > self._diff_percentiles[self._upper_percentile]
                ) or (self._diff_upper_bound and n_tokens > self._diff_upper_bound):
                    self._ids_to_drop.add(id)

        with open(os.path.join(n_tokens_dir, "n_tokens_message.txt"), "r") as file:
            for line in file:
                id, n_tokens = (int(i) for i in line.strip().split(","))
                if (
                    n_tokens < self._message_percentiles[self._lower_percentile]
                    or n_tokens > self._message_percentiles[self._upper_percentile]
                ):
                    self._ids_to_drop.add(id)

    def prepare(
        self,
        in_fname: str,
        n_tokens_dir: str,
        percentile_dir: Optional[str] = None,
    ):
        """
        This method tokenizes diffs and messages and calculates percentiles for # of tokens.

         Args:
             - in_fname: path to read input data from
             - n_tokens_dir: path to save supplementary information like # of tokens for each example and percentiles
             - percentile_dir: (optional) path to directory with already computed percentiles; might be useful for
                dropping outliers from val/test by using percentiles from train, which has much more examples
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


class PreDeduplicationProcessor(BaseProcessor):
    """
    This class is used to process data to format expected by code clones detection tool SourcererCC.
    """

    def __init__(self, project_id: int, chunksize: int, n_workers: int, logger_name: Optional[str] = None):

        super(PreDeduplicationProcessor, self).__init__(
            chunksize=chunksize, logger_name=logger_name, n_workers=n_workers
        )
        self._separators = r'[;.\[\]\(\)\~!\-\+\&\*/%<>\^\|\?\{\}=\#,"\\\:\$\'`@ +\n\r\t]'
        self._project_id = project_id
        self._n_workers = n_workers

    def _get_diff_from_mods(self, mods: List[Dict[str, str]]) -> str:
        """
        This method constructs single diff from all file modifications in one commit.

        We don't want to consider filenames when running duplicates search on diffs, so `old_path`/`new_path`/`change_type`
        fields are ignored.
        """
        return " ".join(mod["diff"] for mod in mods if mod["change_type"] != "UNKNOWN")

    def _hash_string(self, x: str) -> str:
        hash = hashlib.md5()
        hash.update(x.encode("utf-8"))
        return hash.hexdigest()

    def _split_by_several_separators(self, x: str) -> List[str]:
        return [y.strip() for y in re.split(self._separators, x) if y]

    def _preprocess_single_example(
        self, cur_id: int, cur_example: Union[str, List[Dict[str, str]]], data_col: str
    ) -> Tuple[str, int, int]:
        """
        This method does the following:
        1) Preprocesses examples
        2) Converts to following format:
          'token_hash@#@token1@@::@@frequency,token2@@::@@frequency,...'
        3) Calculates total # tokens and unique # tokens
        """
        # diff preprocessing
        if data_col != "message":
            processed_example = self._preprocess_single_diff(cur_id, cur_example)
        # message preprocessing
        else:
            processed_example = self._preprocess_single_msg(cur_id, cur_example)

        c = Counter(self._split_by_several_separators(processed_example))
        tokens_enc = (
            self._hash_string(processed_example) + "@#@" + ",".join(f"{token}@@::@@{freq}" for token, freq in c.items())
        )
        total_n_tokens = sum(c.values())
        unique_n_tokens = len(c)
        return tokens_enc, total_n_tokens, unique_n_tokens

    def _preprocess_single_diff(self, cur_id: int, cur_example: List[Dict[str, str]]) -> str:
        """
        This method preprocesses diffs, which currently includes the following:
            - cast to lowercase
        """
        try:
            processed_example = self._get_diff_from_mods(cur_example)
            processed_example = re.sub("@@.*?@@\n", "", processed_example)
        except TypeError as e:
            self.logger.error(f"[diff] {cur_id} produced TypeError {e}")
            processed_example = str(cur_example)
        return processed_example

    def _preprocess_single_msg(self, cur_id: int, cur_example: str) -> str:
        """
        This method preprocesses messages, which currently includes the following:
            - remove filenames and '@@ -0,0 +1 @@'-like git stuff
        """
        try:
            processed_example = cur_example.lower()
        except AttributeError as e:
            self.logger.error(f"[message] {cur_id} produced AttributeError {e}")
            processed_example = str(cur_example)
        return processed_example

    def preprocess_single_example(self, cur_example: str, cur_id: int, data_col: str):
        if not isinstance(cur_id, int):
            try:
                cur_id = int(cur_id)
            except ValueError:
                self.logger.error(f"`id` is expected to be `int`, got {cur_id} of `{type(cur_id)} instead")
                return ""

        tokens_enc, total_n_tokens, unique_n_tokens = self._preprocess_single_example(
            cur_example=cur_example, cur_id=cur_id, data_col=data_col
        )
        return f"{self._project_id},{cur_id},{total_n_tokens},{unique_n_tokens},{tokens_enc}\n"

    def process(self, chunk: pd.DataFrame, data_col: str) -> List[str]:
        with Parallel(self._n_workers) as pool:
            res = pool(
                delayed(self.preprocess_single_example)(
                    cur_id=item["id"], cur_example=item[data_col], data_col=data_col
                )
                for _, item in chunk[["id", data_col]].iterrows()
            )
        return res


class PostDeduplicationProcessor(BaseProcessor):
    """
    This class is used to drop duplicates found by code clones detection tool SourcererCC.
    """

    def __init__(
        self,
        chunksize: int,
        n_workers: int,
        logger_name: Optional[str] = None,
    ):
        super(PostDeduplicationProcessor, self).__init__(
            chunksize=chunksize, logger_name=logger_name, n_workers=n_workers
        )
        self._train_full_clones: Set[str] = set()
        self._ids_to_drop: Set[int] = set()

    def _extract_metadata(self, input_fname: str, deduplication_dir: str):
        """
        This method saves commits metadata (author, timestamp, repo, hash) from main dataset files into separate
        files.
        """

        full_out_fname = os.path.join(deduplication_dir, "metadata")
        self._prepare_outfile(full_out_fname)

        for i, part in enumerate(["train", "val", "test", "val_original", "test_original"]):
            self.logger.info(f"Extracting metadata from {part}")

            part_out_fname = os.path.join(deduplication_dir, f"{part}_metadata")
            self._prepare_outfile(part_out_fname)

            reader = self._read_input(f"{part}_{input_fname}")

            for chunk in tqdm(reader, desc=f"Iterating over {part} to extract metadata"):
                chunk["project_id"] = i + 1
                self._append_to_outfile(
                    chunk[["project_id", "id", "author", "date", "hash", "repo"]].to_dict(orient="records"),
                    part_out_fname,
                )
                self._append_to_outfile(
                    chunk[["project_id", "id", "author", "date", "hash", "repo"]].to_dict(orient="records"),
                    full_out_fname,
                )

    def _add_metadata(self, in_fname: str, out_fname: str, deduplication_dir: str):
        """
        This method adds metadata to each pair of clones.

        Initially clones are created in a format `project_id1,sample_id1,project_id2,sample_id2`.
        To make them more useful, we add to surrogate sample ids information like repo and commit hash.
        """
        self.logger.info(f"Adding metadata to {in_fname}")
        df = self._read_input(os.path.join(deduplication_dir, "metadata"), read_whole=True).sort_values(
            by=["project_id", "id"]
        )
        df.sort_index(axis=1, inplace=True)
        sorted_cols = {col: i for i, col in enumerate(df.columns.tolist())}

        # fast indexing on SourcererCC ids
        indexes = {}
        for idx, row in tqdm(df.iterrows()):
            indexes[(row["project_id"], row["id"])] = idx
        data = df.to_numpy()

        # clear target file
        open(out_fname, "w").close()

        metadata = []

        with open(in_fname, "r") as file:
            for i, line in tqdm(enumerate(file), desc=f"Iterating over {in_fname} to add metadata"):
                pr_1, s_1, pr_2, s_2 = (int(j) for j in line.strip().split(","))
                ex1 = data[indexes[(pr_1, s_1)]]
                ex2 = data[indexes[(pr_2, s_2)]]

                metadata.append(
                    {
                        "part_id1": ex1[sorted_cols["project_id"]],
                        "id1": ex1[sorted_cols["id"]],
                        "repo1": ex1[sorted_cols["repo"]],
                        "hash1": ex1[sorted_cols["hash"]],
                        "part_id2": ex2[sorted_cols["project_id"]],
                        "id2": ex2[sorted_cols["id"]],
                        "repo2": ex2[sorted_cols["repo"]],
                        "hash2": ex2[sorted_cols["hash"]],
                    }
                )

                if i % self._chunksize == 0:
                    with jsonlines.open(out_fname, mode="a") as writer:
                        writer.writeall(metadata)
                    metadata = []

        if len(metadata) > 0:
            with jsonlines.open(out_fname, mode="a") as writer:
                writer.writeall(metadata)

    def _get_full_clones(self, msg_clones_fname: str, diff_clones_fname: str, out_fname: str):
        """
        This method gets ids of examples from train which are duplicates to some examples from train/val/test in terms of
        both messages and diffs.
        """
        # get train clones by messages
        train_msgs_clones = set()
        with jsonlines.open(msg_clones_fname, "r") as reader:
            for line in tqdm(reader, desc="Reading message clones"):
                if line["part_id1"] == 1:
                    train_msgs_clones.add(
                        f"{line['part_id1']},{line['id1']},{line['repo1']},{line['hash1']},{line['part_id2']},{line['id2']},{line['repo2']},{line['hash2']}\n"
                    )
                elif line["part_id2"] == 1:
                    train_msgs_clones.add(
                        f"{line['part_id2']},{line['id2']},{line['repo2']},{line['hash2']},{line['part_id1']},{line['id1']},{line['repo1']},{line['hash1']}\n"
                    )

        # get train clones by diffs
        train_diffs_clones = set()

        with jsonlines.open(diff_clones_fname, "r") as reader:
            for line in tqdm(reader, desc="Reading diff clones"):
                if line["part_id1"] == 1:
                    train_diffs_clones.add(
                        f"{line['part_id1']},{line['id1']},{line['repo1']},{line['hash1']},{line['part_id2']},{line['id2']},{line['repo2']},{line['hash2']}\n"
                    )
                elif line["part_id2"] == 1:
                    train_diffs_clones.add(
                        f"{line['part_id2']},{line['id2']},{line['repo2']},{line['hash2']},{line['part_id1']},{line['id1']},{line['repo1']},{line['hash1']}\n"
                    )

        self._train_full_clones = train_msgs_clones.intersection(train_diffs_clones)

        with open(out_fname, "w") as file:
            file.writelines(list(self._train_full_clones))

        self._ids_to_drop = set(int(pair.split(",")[1]) for pair in self._train_full_clones)
        self.logger.info(f"Got {len(self._ids_to_drop)} clones ids to drop")

    def prepare(self, in_fname: str, msg_clones_fname: str, diff_clones_fname: str, deduplication_dir: str):
        self._extract_metadata(in_fname, deduplication_dir)

        self._add_metadata(
            in_fname=msg_clones_fname,
            out_fname=f"{msg_clones_fname.split('.')[0]}_metadata.txt",
            deduplication_dir=deduplication_dir,
        )

        self._add_metadata(
            in_fname=diff_clones_fname,
            out_fname=f"{diff_clones_fname.split('.')[0]}_metadata.txt",
            deduplication_dir=deduplication_dir,
        )

        self._get_full_clones(
            msg_clones_fname=f"{msg_clones_fname.split('.')[0]}_metadata.txt",
            diff_clones_fname=f"{diff_clones_fname.split('.')[0]}_metadata.txt",
            out_fname=os.path.join(deduplication_dir, "full_clones_metadata.txt"),
        )

    def process(self, chunk: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return chunk.loc[~chunk["id"].isin(self._ids_to_drop)]


class MessageProcessor(BaseProcessor):
    """
    This class is used to filter undesirable patterns from messages (currently, only via regexes).
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
            r"(signed(-| |)off(-| |)by|co(-| |)authored(-| |)by|also(-| |)by|reviewed(-| |)by|pulled(-| |)by).*?(\n|$)",
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
            x = MessageProcessor._filter_emails(message)
            x = MessageProcessor._filter_urls(x)
            x = MessageProcessor._filter_issue_ref(x)
            x = MessageProcessor._filter_signature(x)
            x = MessageProcessor._filter_at_pattern(x)
            x = MessageProcessor._filter_sha(x)
            x = x.strip()
        except TypeError:
            x = ""
        return x

    def process(self, chunk: pd.DataFrame) -> pd.DataFrame:
        with Parallel(self._n_workers) as pool:
            filtered_messages = pool(
                delayed(MessageProcessor._filter(message) if isinstance(message, str) and message.isascii() else "")
                for _, message in chunk["message"].items()
            )

        chunk["message"] = filtered_messages
        return chunk.loc[chunk.message.str.len() > 0]


class DiffProcessor(BaseProcessor):
    """
    This class is used to filter undesirable patterns from diffs.
    """

    def _filter_diff(self, diff: str) -> str:
        """
        This method filters single diff string.
        Currently filtering for diffs includes the following:
            - removing some unnecessary git stuff (e.g. @@ ... @@)
            - removing non-changed lines
        """
        diff_lines = diff.split("\n")
        processed_lines = []

        for line in diff_lines:
            line = line.strip()

            if line.startswith("-") and len(line) > 1:
                # lines that were removed
                # example: - version='2.0.2'
                processed_lines.append(line)

            elif line.startswith("+") and len(line) > 1:
                # lines that were added
                # example: + version='2.0.2'
                processed_lines.append(line)

            elif line.startswith("Binary files") and line.endswith("differ"):
                # example: Binary files <filename1> and <filename2> differ
                processed_lines.append(line)

        processed_diff = "\n".join(processed_lines)
        processed_diff = re.sub(" +", " ", processed_diff)
        return processed_diff

    def _filter_mods(self, mods: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        This method filters all modifications from single commit.
        """
        filtered_mods = []
        for mod in mods:
            if isinstance(mod["diff"], str) and mod["diff"].isascii():
                mod["diff"] = self._filter_diff(mod["diff"])
                filtered_mods.append(mod)
        return filtered_mods

    def process(self, chunk: pd.DataFrame, **kwargs) -> pd.DataFrame:
        with Parallel(self._n_workers) as pool:
            filtered_mods = pool(delayed(self._filter_mods(mods)) for _, mods in chunk["mods"].items())

        chunk["mods"] = filtered_mods
        return chunk
