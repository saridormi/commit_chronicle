import hashlib
import re
from collections import Counter
from typing import Dict, List, Optional, Union

import pandas as pd
from joblib import Parallel, delayed

from ..utils import BaseProcessor


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

    def _process_single_example(self, cur_id: int, cur_example: Union[str, List[Dict[str, str]]]) -> str:
        """Converts a single example into format required by SourcererCC.

        It includes the following steps:

        * Preprocess example (different for diffs and messages)
        * Calculate total # tokens and unique # tokens
        * Obtain required spring representation:
            'project_id,sample_id,total_n_tokens,unique_n_tokens,token_hash@#@token1@@::@@frequency,...'
        """
        # message preprocessing
        if isinstance(cur_example, str):
            processed_example = self._preprocess_msg(cur_id, cur_example)
        # diff preprocessing
        else:
            processed_example = self._preprocess_mods(cur_id, cur_example)

        c = Counter(self._split_by_several_separators(processed_example))
        tokens_enc = (
            self._hash_string(processed_example) + "@#@" + ",".join(f"{token}@@::@@{freq}" for token, freq in c.items())
        )
        total_n_tokens = sum(c.values())
        unique_n_tokens = len(c)
        return f"{self._project_id},{cur_id},{total_n_tokens},{unique_n_tokens},{tokens_enc}\n"

    def _preprocess_mods(self, cur_id: int, cur_mods: List[Dict[str, str]]) -> str:
        """Preprocesses modifications from single commit, which currently includes the following:

        * unite modifications into single diff string
        * remove '@@ xxx yyy @@' git stuff via regular expression
        """
        try:
            processed_example = self._get_diff_from_mods(cur_mods)
            processed_example = re.sub("@@.*?@@\n", "", processed_example)
        except TypeError as e:
            self.logger.error(f"[diff] {cur_id} produced TypeError {e}")
            processed_example = str(cur_mods)
        return processed_example

    def _preprocess_msg(self, cur_id: int, cur_message: str) -> str:
        """Preprocesses a single commit message, which currently includes the following:

        * cast to lowercase
        """
        try:
            processed_example = cur_message.lower()
        except AttributeError as e:
            self.logger.error(f"[message] {cur_id} produced AttributeError {e}")
            processed_example = str(cur_message)
        return processed_example

    def process(self, chunk: pd.DataFrame, data_col: str, **kwargs) -> List[str]:
        """Processes each example in a chunk into format required by SourcererCC.

        Args:
            chunk: Small subset of original dataset.
            data_col: Should be `message` to process messages or `mods` to process diffs.
        """
        with Parallel(self._n_workers) as pool:
            res = pool(
                delayed(self._process_single_example)(cur_id=item["id"], cur_example=item[data_col])
                for _, item in chunk[["id", data_col]].iterrows()
            )
        return res
