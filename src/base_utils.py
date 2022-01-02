import os
import logging
import jsonlines
import pandas as pd
from tqdm import tqdm
from typing import List, Dict, Any, Optional, Union


class BaseProcessor:
    """
    This is a base class for data collection and processing, provides methods for writing & reading data and for
    logging.

    Currently data is saved as jsonl.
    """

    def __init__(self, chunksize: int, n_workers: Optional[int] = None, logger_name: Optional[str] = None):
        self._chunksize = chunksize
        self._n_workers = n_workers
        self.logger = BaseProcessor._get_logger(logger_name)

    @staticmethod
    def _get_logger(name):
        """
        Workaround for logging with joblib (based on https://github.com/joblib/joblib/issues/1017)
        """
        logger = logging.getLogger(name)
        if len(logger.handlers) == 0:
            logger.setLevel(logging.INFO)
            sh = logging.StreamHandler()
            sh.setFormatter(logging.Formatter("%(asctime)s %(levelname)-8s %(message)s"))
            fh = logging.FileHandler(f"{name}.log", mode="a")
            fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)-8s %(message)s"))
            logger.addHandler(sh)
            logger.addHandler(fh)
        return logger

    def _prepare_outfile(self, out_fname: str):
        """
        Do what might be required before saving to chosen output format.
        (e.g. write header in case of csv files)
        """
        open(os.path.join(out_fname), mode="w").close()

    def _append_to_outfile(self, data: List[Union[str, Dict[str, Any]]], out_fname: str):
        """
        Append current data chunk to chosen output format.
        """
        with jsonlines.open(out_fname, mode="a") as writer:
            writer.write_all(data)

    def _read_input(self, input_fname: str, compression: Optional[str] = None, read_whole: Optional[bool] = None):
        """
        Read data according to chosen output format.
        """
        if read_whole:
            return pd.read_json(
                input_fname,
                orient="records",
                lines=True,
                compression=compression,
                convert_dates=False,
            )

        return pd.read_json(
            input_fname,
            chunksize=self._chunksize,
            orient="records",
            lines=True,
            compression=compression,
            convert_dates=False,
        )

    def prepare(self, in_fname: str, **kwargs):
        """
        This method might perform any necessary actions before chunk processing begins.

        Args:
            - in_fname: path to read input data from
        """
        pass

    def process(self, chunk: pd.DataFrame, **kwargs) -> Union[pd.DataFrame, List[str], Dict[str, Any]]:
        """
        This method should implement chunk processing logic.

        Args:
            - in_fname: path to read input data from
            - out_fname: path to save processed data to
        """
        raise NotImplementedError()

    def __call__(self, in_fname: str, out_fname: str, **kwargs):
        """
        This method iterates over input data in chunks, processes it in some way and saves results to separate file.

        Args:
            - in_fname: path to read input data from
            - out_fname: path to save processed data to
        """
        prepare_kwargs = {key.strip("prepare_"): value for key, value in kwargs.items() if key.startswith("prepare_")}
        process_kwargs = {key: value for key, value in kwargs.items() if not key.startswith("prepare_")}

        self.logger.info(f"Starting processing {in_fname}")

        self._prepare_outfile(out_fname)
        self.prepare(in_fname, **prepare_kwargs)

        reader = self._read_input(in_fname)
        for chunk in tqdm(reader, leave=False):
            processed_chunk = self.process(chunk, **process_kwargs)
            if isinstance(processed_chunk, pd.DataFrame):
                processed_chunk = processed_chunk.to_dict(orient="records")
            self._append_to_outfile(processed_chunk, out_fname)

        self.logger.info(f"Finished processing {in_fname}")
