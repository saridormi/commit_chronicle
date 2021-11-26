import os
import jsonlines
import pandas as pd
from typing import List, Dict, Any, Optional


class FileProcessor:
    """
    This is a base class for writing & reading data, used for data collection and processing.

    Currently data is saved as jsonl.
    """

    def __init__(self, chunksize: int):
        self._chunksize = chunksize

    def _prepare_outfile(self, out_fname: str):
        """
        Do what might be required before saving to chosen output format.
        """
        open(os.path.join(out_fname), mode="w").close()

    def _append_to_outfile(self, data: List[Dict[str, Any]], out_fname: str):
        """
        Append current data chunk to chosen output format.
        """
        with jsonlines.open(out_fname, mode="a") as writer:
            writer.write_all(data)

    def _read_input(self, input_fname: str, compression: Optional[str] = None):
        """
        Read data in chunks according to chosen output format.
        """
        return pd.read_json(
            input_fname,
            chunksize=self._chunksize,
            orient="records",
            lines=True,
            compression=compression,
            convert_dates=False,
        )
