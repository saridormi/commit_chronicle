import os
from typing import List, Optional

import pandas as pd
from tqdm import tqdm

from ..utils import JsonlManager, get_logger


class DiffExtractor:
    """This class is used to extract diffs from data to train tokenizer on."""

    def __init__(
        self,
        chunksize: int,
        data_format: str,
        logger_name: Optional[str] = None,
    ):
        if data_format == "jsonl":
            self._data_manager = JsonlManager()
        else:
            raise ValueError("Given data format is not supported.")
        self.data_format = data_format
        self._chunksize = chunksize
        self._logger_name = logger_name

    @property
    def logger(self):
        return get_logger(self._logger_name)

    def _extract_diffs(self, chunk: pd.DataFrame, line_sep: str) -> List[str]:
        chunk["diff"] = [line_sep.join([mod["diff"] for mod in commit]) + "\n" for commit in chunk["mods"].tolist()]
        return chunk["diff"].tolist()

    def __call__(self, input_dir: str, out_fname: str, line_sep: str) -> None:
        """Extracts first `n_examples` diffs from input file and saves them to separate file.
        If `n_examples` is not given, extracts all diffs from input file.
        """
        self._data_manager.prepare_outfile(out_fname, add_data_format=False)

        repos = sorted(os.listdir(input_dir))
        total_num_examples = 0
        for repo in tqdm(repos, desc=f"Processing input directory"):
            self.logger.info(f"[{repo}] Start processing")

            reader = self._data_manager.read_input(
                os.path.join(input_dir, repo, f"commits.{self.data_format}.gz"),
                compression="gzip",
                add_data_format=False,
                chunksize=self._chunksize,
            )

            for chunk in tqdm(reader, desc=f"Iterating over {repo}", leave=False):
                diffs: List[str] = self._extract_diffs(chunk, line_sep=line_sep)
                self._data_manager.append_to_outfile(data=diffs, out_fname=out_fname, add_data_format=False)

            self.logger.info(f"[{repo}] Finish processing")
