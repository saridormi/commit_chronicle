import re
import nltk
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


class FinalProcessor(BaseProcessor):
    """This class is used to perform several simple operations with dataset before making it public.

    It deletes `mods` field, converts authors' personal information into unique ids
    and adds license type for each repository.

    Args:
        data_format: In which format mined data is saved.
        chunksize: Number of examples to proccess at once (data is read in chunks). Optional, default value is 1000.
        n_workers: Maximum number of concurrently running jobs. Optional, default value is 1 (sequential execution).
        logger_name: Name of logger for this class. Optional, default value is None.
    """

    def __init__(
        self,
        data_format: str,
        chunksize: Optional[int] = None,
        n_workers: Optional[int] = None,
        logger_name: Optional[str] = None,
    ):
        super().__init__(chunksize=chunksize, n_workers=n_workers, data_format=data_format, logger_name=logger_name)
        self._authors: Set[Tuple[str, str]] = set()
        self._authors_map: Dict[Tuple[str, str], int] = {}
        self._repo_license_map: Dict[str, str] = {}

    def _get_authors(self, in_fname: str, in_fnames: List[str]) -> None:
        """Builds an unique set of authors.

        Currently, all work is done when `train` part is given. For other parts, this method doesn't do anything.

        Args:
            in_fname: Path to current dataset part.
            in_fnames: List with paths to all dataset parts. When `train` part is given, authors will be gathered
                from these files.
        """
        if "train" in in_fname:
            for fname in in_fnames:
                reader = self._read_input(fname)
                for chunk in tqdm(reader, desc=f"Reading authors from {in_fname}", leave=False):
                    self._authors.update((tuple(author) for author in chunk["author"].tolist()))

            self._authors_map = {author: i for i, author in enumerate(self._authors)}

    def _get_licenses(self, license_in_fname: str) -> None:
        with open(license_in_fname, "r") as file:
            self._repo_license_map = json.load(file)

    def prepare(self, in_fname: str, license_in_fname: str, in_fnames: List[str], **kwargs) -> None:
        self._get_authors(in_fname=in_fname, in_fnames=in_fnames)
        self._get_licenses(license_in_fname=license_in_fname)

    def process(self, chunk: pd.DataFrame, **kwargs) -> pd.DataFrame:
        chunk = chunk.drop(columns="mods")
        chunk["author"] = chunk["author"].apply(tuple).map(self._authors_map)
        chunk["license"] = chunk["repo"].map(self._repo_license_map)
        return chunk


class OutliersProcessor(BaseProcessor):
    """This class is used to drop commits with too long or too short diffs and messages.

    Examples with # tokens out of [lower_percentile, upper_percentile] range are considered outliers.

    Args:
        lower_percentile: Percentile to use as a lower bound (should be in (0, 1) range).
        upper_percentile: Percentile to use as an upper bound (should be in (0, 1) range).
        data_format: In which format mined data is saved.
        diff_upper_bound: Specific upper bound for number of tokens in diffs. Optional,
            default value is None, and this step is skipped.
        chunksize: Number of examples to proccess at once (data is read in chunks). Optional, default value is 1000.
        n_workers: Maximum number of concurrently running jobs. Optional, default value is 1 (sequential execution).
        logger_name: Name of logger for this class. Optional, default value is None.
    """

    def __init__(
        self,
        lower_percentile: float,
        upper_percentile: float,
        data_format: str,
        diff_upper_bound: Optional[int] = None,
        chunksize: Optional[int] = None,
        n_workers: Optional[int] = None,
        logger_name: Optional[str] = None,
    ):
        super().__init__(chunksize=chunksize, n_workers=n_workers, data_format=data_format, logger_name=logger_name)
        self._lower_percentile = lower_percentile
        self._upper_percentile = upper_percentile
        self._diff_upper_bound = diff_upper_bound

        self._ids_to_drop: Set[int] = set()
        self._diff_percentiles: Dict[float, float] = {}
        self._message_percentiles: Dict[float, float] = {}

    def _get_n_tokens_str(self, string: str) -> int:
        """Splits given string by whitespaces and returns # of tokens."""
        return len(string.split())

    def _get_n_tokens_msg(self, id: int, msg: str) -> str:
        """
        Tokenizes given message and returns a string
            with id and # of tokens separated by ',' with '\n' at the end.
        """
        try:
            return f"{id},{self._get_n_tokens_str(msg)}\n"
        except TypeError as e:
            self.logger.warning(f"TypeError {e} with {id}")
            return f"{id},-1\n"

    def _get_n_tokens_mods(self, id: int, mods: List[Dict[str, str]]) -> str:
        """
        Tokenizes each diff in commit modifications and returns a string
             with id and # of tokens separated by ',' with '\n' at the end.
        """
        try:
            n_tokens = 0
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
                n_tokens += self._get_n_tokens_str(file_diff)
                n_tokens += self._get_n_tokens_str(mod["diff"])
            return f"{id},{n_tokens}\n"
        except TypeError as e:
            self.logger.warning(f"TypeError {e} with {id}")
            return f"{id},-1\n"

    def _get_n_tokens(self, in_fname: str, n_tokens_dir: str) -> None:
        """Tokenizes diff and messages and saves # of tokens in diffs and messages to separate files.

        Args:
            in_fname: Path to read input data from.
            n_tokens_dir: Path to directory to save # of tokens to.
        """
        self.logger.info(f"Starting processing # tokens in {in_fname}")

        open(os.path.join(n_tokens_dir, "n_tokens_diff.txt"), "w", encoding="utf-8").close()
        open(os.path.join(n_tokens_dir, "n_tokens_message.txt"), "w", encoding="utf-8").close()

        reader = self._read_input(in_fname)
        for chunk in tqdm(reader, desc=f"Tokenizing {in_fname}", leave=False):
            with Parallel(self._n_workers) as pool:
                # calculate # tokens in diffs from current chuck
                diff_res = pool(
                    delayed(self._get_n_tokens_mods)(item["id"], item["mods"])
                    for _, item in chunk[["id", "mods"]].iterrows()
                )
                # calculate # tokens in messages from current chuck
                message_res = pool(
                    delayed(self._get_n_tokens_msg)(item["id"], item["message"])
                    for _, item in chunk[["id", "message"]].iterrows()
                )
            # append results from current chunk to target files
            with open(os.path.join(n_tokens_dir, "n_tokens_diff.txt"), "a", encoding="utf-8") as file:
                file.writelines(diff_res)
            with open(os.path.join(n_tokens_dir, "n_tokens_message.txt"), "a", encoding="utf-8") as file:
                file.writelines(message_res)

        self.logger.info(f"Finished processing # tokens in {in_fname}")

    def _get_percentiles(self, n_tokens_dir: str) -> None:
        """Calculates 1%, 5%, 90%, 95%, 99% percentiles of # tokens in diffs and messages.

        Args:
            n_tokens_dir: Path to directory to read # of tokens from.
        """
        diff_n_tokens = []
        with open(os.path.join(n_tokens_dir, "n_tokens_diff.txt"), "r") as file:
            for line in file:
                id, n_tokens = (int(i) for i in line.strip().split(","))
                if n_tokens != -1:
                    diff_n_tokens.append(n_tokens)

        message_n_tokens = []
        with open(os.path.join(n_tokens_dir, "n_tokens_message.txt"), "r") as file:
            for line in file:
                id, n_tokens = (int(i) for i in line.strip().split(","))
                if n_tokens != -1:
                    message_n_tokens.append(n_tokens)

        for q in [0.01, 0.05, 0.9, 0.95, 0.99]:
            self._diff_percentiles[q] = np.quantile(diff_n_tokens, q)
            self._message_percentiles[q] = np.quantile(message_n_tokens, q)

        with open(os.path.join(n_tokens_dir, "diff.json"), "w") as file:
            json.dump(self._diff_percentiles, file)
        with open(os.path.join(n_tokens_dir, "message.json"), "w") as file:
            json.dump(self._message_percentiles, file)

    def _get_ids_to_drop(self, n_tokens_dir: str) -> None:
        """Aggregates ids of examples which either:
            * have # tokens in diff or in message out of [lower_percentile, upper_percentile] range
            * produced `TypeError`s (have -1 as # tokens)
            * OPTIONAL: have # tokens in diff > diff_upper_bound

        Args:
            n_tokens_dir: path to directory to read # of tokens from
        """
        self._ids_to_drop = set()

        with open(os.path.join(n_tokens_dir, "n_tokens_diff.txt"), "r") as file:
            for line in file:
                id, n_tokens = (int(i) for i in line.strip().split(","))
                if (
                    (n_tokens == -1)
                    or (
                        n_tokens < self._diff_percentiles[self._lower_percentile]
                        or n_tokens > self._diff_percentiles[self._upper_percentile]
                    )
                    or (self._diff_upper_bound and n_tokens > self._diff_upper_bound)
                ):
                    self._ids_to_drop.add(id)

        with open(os.path.join(n_tokens_dir, "n_tokens_message.txt"), "r") as file:
            for line in file:
                id, n_tokens = (int(i) for i in line.strip().split(","))
                if (n_tokens == -1) or (
                    n_tokens < self._message_percentiles[self._lower_percentile]
                    or n_tokens > self._message_percentiles[self._upper_percentile]
                ):
                    self._ids_to_drop.add(id)

    def prepare(self, in_fname: str, n_tokens_dir: str, percentile_dir: Optional[str] = None, **kwargs) -> None:
        """Tokenizes diffs and messages and calculates percentiles for # of tokens.

        Args:
            in_fname: Path to read input data from.
            n_tokens_dir: Path to folder to save supplementary information (# of tokens and percentiles).
            percentile_dir: Path to directory with already computed percentiles. Optional. Use-case: dropping outliers
                from val/test by percentiles calculated on train.
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
    """This class is used to process data to format expected by code clones detection tool SourcererCC.

    Args:
        project_id: An id required in SourcererCC, we use it to denote different dataset parts (train/val/test).
        data_format: In which format mined data is saved.
        chunksize: Number of examples to proccess at once (data is read in chunks). Optional, default value is 1000.
        n_workers: Maximum number of concurrently running jobs. Optional, default value is 1 (sequential execution).
        logger_name: Name of logger for this class. Optional, default value is None.
    """

    def __init__(
        self,
        project_id: int,
        data_format: str,
        chunksize: Optional[int] = None,
        n_workers: Optional[int] = None,
        logger_name: Optional[str] = None,
    ):
        super().__init__(chunksize=chunksize, n_workers=n_workers, data_format=data_format, logger_name=logger_name)
        self._separators = r'[;.\[\]\(\)\~!\-\_\+\&\*/%<>\^\|\?\{\}=\#,"\\\:\$\'`@ +\n\r\t]'
        self._project_id = project_id
        self._n_workers = n_workers

    def _get_diff_from_mods(self, mods: List[Dict[str, str]]) -> str:
        """Constructs single diff from all file modifications in one commit.

        We don't want to consider filenames when running duplicates search on diffs,
            so `old_path`/`new_path`/`change_type` fields are ignored.
        """
        return " ".join(mod["diff"] for mod in mods)

    def _hash_string(self, x: str) -> str:
        """Obtains hash of given string."""
        hash = hashlib.md5()
        hash.update(x.encode("utf-8"))
        return hash.hexdigest()

    def _split_by_several_separators(self, x: str) -> List[str]:
        """Splits given string by punctuation and whitespaces."""
        return [y.strip() for y in re.split(self._separators, x) if y]

    def _process_single_example(self, cur_id: int, cur_example: Union[str, List[Dict[str, str]]], data_col: str) -> str:
        """Converts a single example into format required by SourcererCC.

        It includes the following steps:

        * Preprocess example (different for diffs and messages)
        * Calculate total # tokens and unique # tokens
        * Obtain required spring representation:
            'project_id,sample_id,total_n_tokens,unique_n_tokens,token_hash@#@token1@@::@@frequency,...'
        """
        if not isinstance(cur_id, int):
            try:
                cur_id = int(cur_id)
            except ValueError:
                self.logger.error(f"`id` is expected to be `int`, got {cur_id} of `{type(cur_id)} instead")
                return ""

        # diff preprocessing
        if data_col != "message":
            processed_example = self._preprocess_mods(cur_id, cur_example)
        # message preprocessing
        else:
            processed_example = self._preprocess_msg(cur_id, cur_example)

        c = Counter(self._split_by_several_separators(processed_example))
        tokens_enc = (
            self._hash_string(processed_example) + "@#@" + ",".join(f"{token}@@::@@{freq}" for token, freq in c.items())
        )
        total_n_tokens = sum(c.values())
        unique_n_tokens = len(c)
        return f"{self._project_id},{cur_id},{total_n_tokens},{unique_n_tokens},{tokens_enc}\n"

    def _preprocess_mods(self, cur_id: int, cur_example: List[Dict[str, str]]) -> str:
        """Preprocesses modifications from single commit, which currently includes the following:

        * unite modifications into single diff string
        * remove '@@ xxx yyy @@' git stuff via regular expression
        """
        try:
            processed_example = self._get_diff_from_mods(cur_example)
            processed_example = re.sub("@@.*?@@\n", "", processed_example)
        except TypeError as e:
            self.logger.error(f"[diff] {cur_id} produced TypeError {e}")
            processed_example = str(cur_example)
        return processed_example

    def _preprocess_msg(self, cur_id: int, cur_example: str) -> str:
        """Preprocesses a single commit message, which currently includes the following:

        * cast to lowercase
        """
        try:
            processed_example = cur_example.lower()
        except AttributeError as e:
            self.logger.error(f"[message] {cur_id} produced AttributeError {e}")
            processed_example = str(cur_example)
        return processed_example

    def process(self, chunk: pd.DataFrame, data_col: str, **kwargs) -> List[str]:
        """Processes each example in a chunk into format required by SourcererCC.

        Args:
            chunk: Small subset of original dataset.
            data_col: Should be `message` to process messages or `mods` to process diffs.
        """
        with Parallel(self._n_workers) as pool:
            res = pool(
                delayed(self._process_single_example)(cur_id=item["id"], cur_example=item[data_col], data_col=data_col)
                for _, item in chunk[["id", data_col]].iterrows()
            )
        return res


class PostDeduplicationProcessor(BaseProcessor):
    """This class is used to drop duplicates found by code clones detection tool SourcererCC.

    Args:
        data_format: In which format mined data is saved.
        chunksize: Number of examples to proccess at once (data is read in chunks). Optional, default value is 1000.
        n_workers: Maximum number of concurrently running jobs. Optional, default value is 1 (sequential execution).
        logger_name: Name of logger for this class. Optional, default value is None.
    """

    def __init__(
        self,
        data_format: str,
        chunksize: Optional[int] = None,
        n_workers: Optional[int] = None,
        logger_name: Optional[str] = None,
    ):
        super().__init__(chunksize=chunksize, n_workers=n_workers, data_format=data_format, logger_name=logger_name)
        self._train_full_clones: Set[str] = set()
        self._ids_to_drop: Set[int] = set()

    def _extract_metadata(self, in_path: str, deduplication_dir: str, parts: List[str]) -> None:
        """Saves commits metadata (author, timestamp, repo, hash) from main dataset files to separate files.

        Args:
            in_path: Path to folder where input data is stored.
            parts: List of all parts in input dataset.
            deduplication_dir: Path to folder where files with found clones are stored.
        """

        full_out_fname = os.path.join(deduplication_dir, "metadata")
        self._prepare_outfile(full_out_fname)

        for i, part in enumerate(parts):
            self.logger.info(f"Extracting metadata from {part}")

            part_out_fname = os.path.join(deduplication_dir, f"{part}_metadata")
            self._prepare_outfile(part_out_fname)

            reader = self._read_input(os.path.join(in_path, part))

            for chunk in tqdm(reader, desc=f"Iterating over {part} to extract metadata"):
                chunk["project_id"] = i + 1
                self._append_to_outfile(
                    chunk[["project_id", "id", "author", "date", "hash", "repo"]],
                    part_out_fname,
                )
                self._append_to_outfile(
                    chunk[["project_id", "id", "author", "date", "hash", "repo"]],
                    full_out_fname,
                )

    def _add_metadata(self, in_fname: str, out_fname: str, deduplication_dir: str):
        """Adds metadata to each pair of clones.

        Initially clones are created in a format `project_id1,sample_id1,project_id2,sample_id2`, we add metadata about
            each example for further use.
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
                        writer.write_all(metadata)
                    metadata = []

        if len(metadata) > 0:
            with jsonlines.open(out_fname, mode="a") as writer:
                writer.write_all(metadata)

    def _get_full_clones(self, msg_clones_fname: str, diff_clones_fname: str, out_fname: str):
        """Builds a set of ids of examples from train which are completely identical to examples from train/val/test
            (both diffs and messages are the same).

        Args:
            msg_clones_fname: Path to file with clones in terms of messages.
            diff_clones_fname: Path to file with clones in terms of diffs.
            out_fname: Path to save resulting full clones.
        """
        # get train clones by messages
        train_msgs_clones = set()
        with jsonlines.open(msg_clones_fname, "r") as reader:
            for line in tqdm(reader, desc="Reading message clones"):
                if line["part_id1"] == 1 and line["part_id2"] != 1:
                    train_msgs_clones.add(
                        f"{line['part_id1']},{line['id1']},{line['repo1']},{line['hash1']},{line['part_id2']},{line['id2']},{line['repo2']},{line['hash2']}\n"
                    )
                elif line["part_id2"] == 1 and line["part_id1"] != 1:
                    train_msgs_clones.add(
                        f"{line['part_id2']},{line['id2']},{line['repo2']},{line['hash2']},{line['part_id1']},{line['id1']},{line['repo1']},{line['hash1']}\n"
                    )

        # get train clones by diffs
        train_diffs_clones = set()
        with jsonlines.open(diff_clones_fname, "r") as reader:
            for line in tqdm(reader, desc="Reading diff clones"):
                if line["part_id1"] == 1 and line["part_id2"] != 1:
                    train_diffs_clones.add(
                        f"{line['part_id1']},{line['id1']},{line['repo1']},{line['hash1']},{line['part_id2']},{line['id2']},{line['repo2']},{line['hash2']}\n"
                    )
                elif line["part_id2"] == 1 and line["part_id1"] != 1:
                    train_diffs_clones.add(
                        f"{line['part_id2']},{line['id2']},{line['repo2']},{line['hash2']},{line['part_id1']},{line['id1']},{line['repo1']},{line['hash1']}\n"
                    )

        self._train_full_clones = train_msgs_clones.intersection(train_diffs_clones)

        with open(out_fname, "w") as file:
            file.writelines(list(self._train_full_clones))

        self._ids_to_drop = set(int(pair.split(",")[1]) for pair in self._train_full_clones)
        self.logger.info(f"Got {len(self._ids_to_drop)} clones ids to drop")

    def prepare(
        self,
        in_fname: str,
        in_path: str,
        parts: List[str],
        msg_clones_fname: str,
        diff_clones_fname: str,
        deduplication_dir: str,
        is_ready: Optional[bool] = False,
        **kwargs,
    ) -> None:
        """Prepares a set of ids of fully identical entries between train and validation/test.

        During this process, metadata is extracted from input dataset and added to clones ids.

        Args:
            in_fname: Path to specific input file.
            in_path: Path to root folder with input data.
            parts: List of all parts in input dataset.
            msg_clones_fname: Path to file with clones in terms of messages.
            diff_clones_fname: Path to file with clones in terms of diffs.
            deduplication_dir: Path to folder where files with found clones are stored.
            is_ready: A flag to indicate cases when clones ids are already built. When it is set to True,
                this method doesn't do anything.
        """
        if is_ready:
            return

        self._extract_metadata(in_path, deduplication_dir, parts)

        self._add_metadata(
            in_fname=os.path.join(deduplication_dir, msg_clones_fname),
            out_fname=os.path.join(deduplication_dir, f"{msg_clones_fname.split('.')[0]}_metadata.txt"),
            deduplication_dir=deduplication_dir,
        )

        self._add_metadata(
            in_fname=os.path.join(deduplication_dir, diff_clones_fname),
            out_fname=os.path.join(deduplication_dir, f"{diff_clones_fname.split('.')[0]}_metadata.txt"),
            deduplication_dir=deduplication_dir,
        )

        self._get_full_clones(
            msg_clones_fname=os.path.join(deduplication_dir, f"{msg_clones_fname.split('.')[0]}_metadata.txt"),
            diff_clones_fname=os.path.join(deduplication_dir, f"{diff_clones_fname.split('.')[0]}_metadata.txt"),
            out_fname=os.path.join(deduplication_dir, "full_clones_metadata.txt"),
        )

    def process(self, chunk: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return chunk.loc[~chunk["id"].isin(self._ids_to_drop)]


class MessageProcessor(BaseProcessor):
    """
    This class is used to delete undesirable patterns from messages and filter messages.

    * Reused regexes for deleting emails, urls and SHA from
    Liu, Zhongxin, et al. "Automatic generation of pull request descriptions."
    2019 34th IEEE/ACM International Conference on Automated Software Engineering (ASE). IEEE, 2019.

    * Reused regexes for filtering bot and trivial messages from
    Liu, Zhongxin, et al. "Neural-machine-translation-based commit message generation: how far are we?."
    Proceedings of the 33rd ACM/IEEE International Conference on Automated Software Engineering. 2018.
    """

    @staticmethod
    def _filter_emails(message: str) -> str:
        return re.sub(r"(^|\s)<[\w.-]+@(?=[a-z\d][^.]*\.)[a-z\d.-]*[^.]>", "", message)

    @staticmethod
    def _filter_urls(message: str) -> str:
        return re.sub(r"https?://[-a-zA-Z0-9@:%._+~#?=/]+(?=($|[^-a-zA-Z0-9@:%._+~#?=/]))", "", message)

    @staticmethod
    def _filter_at_pattern(message: str) -> str:
        return re.sub(r"@\S+", "", message)

    @staticmethod
    def _filter_sha(message: str) -> str:
        x = re.sub(r"(^|\s)[\dA-Fa-f-]{7,}(?=(\s|$))", "", message)
        x = re.sub(r"(ref:)[\dA-Fa-f-]{7,}(?=(\s|$))", "", x)  # from yandex repos
        x = re.sub(r"\bI[0-9a-fA-F]{6,40}\b", "", x)  # gerrit
        return x

    @staticmethod
    def _filter_issue_ref(message: str) -> str:
        """
        Deletes issue numbers from the following patterns:

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
        Filters various signatures from messages

        * Not sure about specific tools/repos, but these kinds of signatures appear quite often
            * `Signed-off-by: <username>`
            * `Co-authored-by: <username>`
            * `Also-by: <username>`
            * `Reviewed-by: <username>`
            * `Former commit id: <id>`
        * https://github.com/google/moe: `Created by MOE: <some link>\nMOE_MIGRATED_REVID=<some number>`
        * https://github.com/facebook/fbshipit:
            * `Differential Revision: <some number>`
            * `Pulled By: <username>`
            * `fbshipit-source-id: <some sha-like string>`
        * https://github.com/google/copybara:
            * `BUG=<some number>`
            * `FIXES=<some number>`
            * `Change-Id: <some sha-like string>`
            * `PiperOrigin-RevId: <some number>`
            * `BAZEL_VERSION_REV_ID: <some number>`
        """
        x = re.sub(
            r"(signed(-| |)off(-| |)by|co(-| |)authored(-| |)by|also(-| |)by|reviewed(-| |)by|pulled(-| |)by|former("
            r"-| |)commit(-| |)id).*?(\n|$)",
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
    def _is_trivial_or_bot(message: str) -> bool:
        message = message.strip()
        # pad punctuation with spaces - expected format in given regular expressions
        message = message.translate(str.maketrans({key: " {0} ".format(key) for key in punctuation}))
        message = re.sub(" +", " ", message)

        patterns = [
            # for bot messages
            r"^ignore update \' .* \.$",
            # for shadow messages
            r"^update(d)? (changelog|gitignore|readme( . md| file)?)( \.)?$",
            r"^prepare version (v)?[ \d.]+$",
            r"^bump (up )?version( number| code)?( to (v)?[ \d.]+( - snapshot)?)?( \.)?$",
            r"^modify (dockerfile|makefile)( \.)?$",
            r"^update submodule(s)?( \.)?$",
        ]

        for pattern in patterns:
            if re.match(pattern, message, flags=re.IGNORECASE):
                return True

        return False

    @staticmethod
    def _filter(message: str) -> str:
        if not isinstance(message, str) or not message.isascii() or MessageProcessor._is_trivial_or_bot(message):
            return ""

        x = MessageProcessor._filter_emails(message)
        x = MessageProcessor._filter_urls(x)
        x = MessageProcessor._filter_issue_ref(x)
        x = MessageProcessor._filter_signature(x)
        x = MessageProcessor._filter_at_pattern(x)
        x = MessageProcessor._filter_sha(x)
        x = x.replace("\n", " ")
        x = x.strip()
        return x

    def process(self, chunk: pd.DataFrame, **kwargs) -> pd.DataFrame:
        with Parallel(self._n_workers) as pool:
            filtered_messages = pool(
                delayed(MessageProcessor._filter)(message) for _, message in chunk["message"].items()
            )

        chunk["message"] = filtered_messages
        return chunk.loc[chunk.message.str.len() > 0]


class DiffProcessor(BaseProcessor):
    """This class is used to delete undesirable patterns from diffs."""

    @staticmethod
    def _filter_diff(diff: str) -> str:
        """Filters single diff string.

        Currently, it includes the following:
            * removing some unnecessary git stuff (e.g. @@ ... @@)
            * removing non-changed lines
            * removing extra `\t` and `\r` symbols
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
        processed_diff = re.sub("[^\S\n]+", " ", processed_diff)
        return processed_diff

    @staticmethod
    def _filter_mods(mods: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Filters all modifications from single commit.
        """
        filtered_mods = []
        for mod in mods:
            if isinstance(mod["diff"], str) and mod["diff"].isascii():
                mod["diff"] = DiffProcessor._filter_diff(mod["diff"])
                filtered_mods.append(mod)
        return filtered_mods

    def process(self, chunk: pd.DataFrame, **kwargs) -> pd.DataFrame:
        with Parallel(self._n_workers) as pool:
            filtered_mods = pool(delayed(DiffProcessor._filter_mods)(mods) for _, mods in chunk["mods"].items())

        chunk["mods"] = filtered_mods
        return chunk
