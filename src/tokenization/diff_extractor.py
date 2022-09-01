from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from ..utils import BaseProcessor


class DiffExtractor(BaseProcessor):
    """This class is used to extract diffs from data to train tokenizer on.

    In addition, it calculates percentiles on diff lengths and drops examples longer than specified percentile.

    Args:
        upper_percentile: Percentile to use as an upper bound (should be in (0, 1) range).
        data_format: In which format mined data is saved.
        chunksize: Number of examples to process at once (data is read in chunks). Optional, default value is 1000.
        n_workers: Maximum number of concurrently running jobs. Optional, default value is 1 (sequential execution).
        logger_name: Name of logger for this class. Optional, default value is None.
    """

    def __init__(
        self,
        upper_percentile: float,
        data_format: str,
        chunksize: Optional[int] = None,
        n_workers: Optional[int] = None,
        logger_name: Optional[str] = None,
    ):
        super().__init__(chunksize=chunksize, n_workers=n_workers, data_format=data_format, logger_name=logger_name)
        self._upper_percentile = upper_percentile
        self._percentiles: Dict[float, float] = {}

    def prepare(self, in_fname: str, line_sep: str, **kwargs) -> None:
        """Calculates percentiles on diff lengths."""
        diff_lens = []
        reader = self._read_input(in_fname)
        for chunk in tqdm(reader, leave=False, desc=f"Iterating over {in_fname} to compute diff lens percentiles"):
            diff_lens.extend([len(line_sep.join([mod["diff"] for mod in commit])) for commit in chunk["mods"].tolist()])
        for q in [0.01, 0.05, 0.9, 0.95, 0.99]:
            self._percentiles[q] = np.quantile(diff_lens, q)
        self.logger.info(f"{self._percentiles}")

    def process(self, chunk: pd.DataFrame, line_sep: str, **kwargs) -> List[str]:
        chunk["diff"] = [line_sep.join([mod["diff"] for mod in commit]) + "\n" for commit in chunk["mods"].tolist()]
        chunk["diff_len"] = [len(diff) for diff in chunk["diff"].tolist()]
        chunk = chunk.loc[chunk["diff_len"] <= self._percentiles[self._upper_percentile]]
        return chunk["diff"].tolist()

    def extract_diffs(self, in_fname: str, out_fname: str, line_sep: str, n_examples: Optional[int] = None) -> None:
        """Extracts first `n_examples` diffs from input file and saves them to separate file.
        If `n_examples` is not given, extracts all diffs from input file.
        """
        self.logger.info(f"Starting processing {in_fname}")

        self._prepare_outfile(out_fname, add_data_format=False)
        self.prepare(in_fname, line_sep)

        if not n_examples:
            reader = self._read_input(in_fname)
            for chunk in tqdm(reader, leave=False):
                processed_chunk = self.process(chunk, line_sep)
                self._append_to_outfile(processed_chunk, out_fname)
        else:
            reader = self._read_input(in_fname)
            n_processed_examples = 0
            for chunk in tqdm(reader, leave=False):

                if n_processed_examples + len(chunk) > n_examples:
                    chunk = chunk[: n_examples - n_processed_examples]

                processed_chunk = self.process(chunk, line_sep)
                self._append_to_outfile(processed_chunk, out_fname)

                n_processed_examples += len(chunk)
                if n_processed_examples >= n_examples:
                    break

        self.logger.info(f"Finished processing {in_fname}")
