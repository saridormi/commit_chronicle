from typing import Optional

import pandas as pd
from tqdm import tqdm

from ..utils import BaseProcessor


class DiffExtractor(BaseProcessor):
    def process(self, chunk: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return chunk["diff"].tolist()

    def extract_diffs(self, in_fname: str, out_fname: str, n_examples: Optional[int] = None):
        """Extracts first `n_examples` diffs from input file and saves them to separate file."""
        self.logger.info(f"Starting processing {in_fname}")

        self._prepare_outfile(out_fname, add_data_format=False)

        if not n_examples:
            reader = self._read_input(in_fname)
            for chunk in tqdm(reader, leave=False):
                processed_chunk = self.process(chunk)
                self._append_to_outfile(processed_chunk, out_fname)
        else:
            reader = self._read_input(in_fname)
            n_processed_examples = 0
            for chunk in tqdm(reader, leave=False):

                if n_processed_examples + len(chunk) > n_examples:
                    chunk = chunk[: n_examples - n_processed_examples]

                processed_chunk = self.process(chunk)
                self._append_to_outfile(processed_chunk, out_fname)

                n_processed_examples += len(chunk)
                if n_processed_examples > n_examples:
                    break

        self.logger.info(f"Finished processing {in_fname}")
