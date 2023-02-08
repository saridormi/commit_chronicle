import hashlib
import os
import re
from collections import Counter
from typing import Dict, List, Optional, Tuple, Union

import jsonlines
from tqdm import tqdm

from ..utils import JsonlManager, get_logger


class PreDeduplicationProcessor:
    """This class is used to process data to format expected by code clones detection tool SourcererCC.

    Args:
        data_format: In which format mined data is saved.
        special_tokens: A list of special
        chunksize: Number of examples to process at once. Optional, default value is 1000.
        logger_name: Name of logger for this class. Optional, default value is None.
        n_workers: Maximum number of concurrently running jobs. Not used in this class, default value is None.
    """

    def __init__(
        self,
        data_format: str,
        special_tokens: List[str],
        chunksize: int = 1000,
        logger_name: Optional[str] = None,
        n_workers: Optional[int] = None,
    ):
        if data_format == "jsonl":
            self._data_manager = JsonlManager()
        else:
            raise ValueError("Given data format is not supported.")
        self.data_format = data_format
        self._chunksize = chunksize
        self._logger_name = logger_name
        self._n_workers = n_workers

        self._separators = r'[;.\[\]\(\)\~!\-\_\+\&\*/%<>\^\|\?\{\}=\#,"\\\:\$\'`@ +\n\r\t]'
        self._special_tokens = special_tokens
        self._commits_map: Dict[Tuple[str, str], int] = {}

    @property
    def logger(self):
        return get_logger(self._logger_name)

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

    def _process_single_example(
        self, project_id: int, cur_example: Union[str, List[Dict[str, str]]], cur_repo: str, cur_hash: str
    ) -> str:
        """Converts a single example into format required by SourcererCC.

        It includes the following steps:

        * Preprocess example (different for diffs and messages)
        * Calculate total # tokens and unique # tokens
        * Obtain required spring representation:
            'project_id,sample_id,total_n_tokens,unique_n_tokens,token_hash@#@token1@@::@@frequency,...'
        """
        # message preprocessing
        if isinstance(cur_example, str):
            processed_example = self._preprocess_msg(cur_repo=cur_repo, cur_hash=cur_hash, cur_message=cur_example)
        # diff preprocessing
        else:
            processed_example = self._preprocess_mods(cur_repo=cur_repo, cur_hash=cur_hash, cur_mods=cur_example)

        c = Counter(self._split_by_several_separators(processed_example))
        tokens_enc = (
            self._hash_string(processed_example) + "@#@" + ",".join(f"{token}@@::@@{freq}" for token, freq in c.items())
        )
        total_n_tokens = sum(c.values())
        unique_n_tokens = len(c)

        if (cur_repo, cur_hash) not in self._commits_map:
            self._commits_map[(cur_repo, cur_hash)] = len(self._commits_map)
        cur_unique_id = self._commits_map[(cur_repo, cur_hash)]

        return f"{project_id},{cur_unique_id},{total_n_tokens},{unique_n_tokens},{tokens_enc}\n"

    def _preprocess_mods(self, cur_hash: str, cur_repo: str, cur_mods: List[Dict[str, str]]) -> str:
        """Preprocesses modifications from single commit, which currently includes the following:

        * unite modifications into single diff string
        * remove '@@ xxx yyy @@' git stuff via regular expression
        """
        try:
            processed_example = self._get_diff_from_mods(cur_mods)
            processed_example = re.sub("@@.*?@@\n", "", processed_example)
        except TypeError as e:
            self.logger.error(f"[diff] Commit {cur_hash} from {cur_repo} produced TypeError {e}")
            processed_example = str(cur_mods)
        for token in self._special_tokens:
            processed_example = processed_example.replace(token, "")
        return processed_example

    def _preprocess_msg(self, cur_hash: str, cur_repo: str, cur_message: str) -> str:
        """Preprocesses a single commit message, which currently includes the following:

        * cast to lowercase
        """
        try:
            processed_example = cur_message.lower()
        except AttributeError as e:
            self.logger.error(f"[message] Commit {cur_hash} from {cur_repo} produced AttributeError {e}")
            processed_example = str(cur_message)
        for token in self._special_tokens:
            processed_example = processed_example.replace(token, "")
        return processed_example

    def save_map(self, out_path: str) -> None:
        """
        Save id <-> (repo, hash) mapping to given path.

        Note:
            JSONLines format with keys `repo`, `hash` and `id` is used.
            This mapping is necessary for dropping clones later.

        Args:
            out_path: Path to save mapping to.
        """
        with jsonlines.open(out_path, "w") as writer:
            writer.write_all(
                [{"repo": key[0], "hash": key[1], "id": self._commits_map[key]} for key in self._commits_map]
            )

    def __call__(self, input_dir: str, diff_fname: str, message_fname: str, part: str, project_id: int) -> None:
        """
        Process diffs and messages from all repositories in `input_dir` to format required by SourcererCC.

        Note:
            If you want to process several dataset parts (e.g. train, val, test), ensure that you use the same PreDeduplicationProcessor instance
            to process them one by one. It is necessary to ensure correct ids mapping, SourcererCC expects them to be unique.

        Args:
            input_dir: Path to root input directory with data.
            diff_fname: Path to txt file to save diffs to.
            message_fname: Path to txt file to save messages to.
            part: Name of current dataset part.
            project_id: Unique id for current dataset part.
        """
        self._data_manager.prepare_outfile(diff_fname, add_data_format=False)
        self._data_manager.prepare_outfile(message_fname, add_data_format=False)

        repos = sorted(os.listdir(input_dir))

        for repo in tqdm(repos, desc=f"Processing {part}"):
            self.logger.info(f"[{repo}] Start processing")

            reader = self._data_manager.read_input(
                os.path.join(input_dir, repo, f"commits.{self.data_format}.gz"),
                compression="gzip",
                add_data_format=False,
                chunksize=self._chunksize,
            )

            for chunk in tqdm(reader, desc=f"Iterating over {repo}", leave=False):
                diff_res: List[str] = [
                    self._process_single_example(
                        cur_repo=repo, cur_hash=item["hash"], cur_example=item["mods"], project_id=project_id
                    )
                    for _, item in chunk[["hash", "mods"]].iterrows()
                ]
                self._data_manager.append_to_outfile(data=diff_res, out_fname=diff_fname, add_data_format=False)

                message_res: List[str] = [
                    self._process_single_example(
                        cur_repo=repo, cur_hash=item["hash"], cur_example=item["message"], project_id=project_id
                    )
                    for _, item in chunk[["hash", "message"]].iterrows()
                ]
                self._data_manager.append_to_outfile(data=message_res, out_fname=message_fname, add_data_format=False)
