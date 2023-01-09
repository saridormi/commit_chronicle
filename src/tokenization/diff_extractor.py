from typing import List, Optional

import pandas as pd
from tqdm import tqdm

from ..utils import BaseProcessor


class DiffExtractor(BaseProcessor):
    """This class is used to extract diffs from data to train tokenizer on."""

    def process(self, chunk: pd.DataFrame, line_sep: str, **kwargs) -> List[str]:  # type: ignore[override]
        chunk["diff"] = [line_sep.join([mod["diff"] for mod in commit]) + "\n" for commit in chunk["mods"].tolist()]
        return chunk["diff"].tolist()

    def extract_diffs(self, in_fname: str, out_fname: str, line_sep: str, n_examples: Optional[int] = None) -> None:
        """Extracts first `n_examples` diffs from input file and saves them to separate file.
        If `n_examples` is not given, extracts all diffs from input file.
        """
        self.logger.info(f"Starting processing {in_fname}")

        self._prepare_outfile(out_fname, add_data_format=False)

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
