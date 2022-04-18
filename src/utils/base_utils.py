import logging
from typing import List, Optional, Union

import dask.dataframe as dd
import jsonlines
import pandas as pd
from tqdm import tqdm


class BaseManager:
    """
    This is a base class for writing & reading data.
    """

    def prepare_outfile(self, out_fname: str, add_data_format: Optional[bool] = True) -> None:
        """
        Does what might be required before saving to chosen output format.
        (e.g. write header in case of csv files)
        """
        raise NotImplementedError()

    def append_to_outfile(self, data: pd.DataFrame, out_fname: str, add_data_format: Optional[bool] = True) -> None:
        """
        Appends current data chunk to chosen output format.
        """
        raise NotImplementedError()

    def read_input(self, in_fname: str, add_data_format: Optional[bool] = True, **kwargs):
        """
        Reads data according to chosen output format (accessing full dataset/reading in chunks).
        """
        raise NotImplementedError()

    def read_input_lazy(self, in_fname: str, add_data_format: Optional[bool] = True, **kwargs):
        """
        Reads data according to chosen output format (accessing full dataset in lazy fashion).
        """
        raise NotImplementedError()


class JsonlManager(BaseManager):
    """
    This is a class for writing & reading jsonl data.
    """

    def prepare_outfile(self, out_fname: str, add_data_format: Optional[bool] = True) -> None:
        """
        Clears target file.
        """
        if add_data_format:
            out_fname = f"{out_fname}.jsonl"

        open(out_fname, mode="w").close()

    def append_to_outfile(
        self,
        data: pd.DataFrame,
        out_fname: str,
        add_data_format: Optional[bool] = True,
    ) -> None:
        """
        Appends current data chunk.
        """
        data = data.to_dict(orient="records")

        if add_data_format:
            out_fname = f"{out_fname}.jsonl"

        with jsonlines.open(out_fname, mode="a") as writer:
            writer.write_all(data)

    def read_input(self, in_fname: str, add_data_format: Optional[bool] = True, **kwargs):
        """
        Reads jsonl data with pandas.
        """
        if add_data_format:
            in_fname = f"{in_fname}.jsonl"

        return pd.read_json(in_fname, orient="records", lines=True, convert_dates=False, **kwargs)

    def read_input_lazy(self, in_fname: str, add_data_format: Optional[bool] = True, **kwargs):
        """
        Reads jsonl data with dask.
        """
        if add_data_format:
            in_fname = f"{in_fname}.jsonl"

        return dd.read_json(in_fname, orient="records", lines=True, **kwargs)


class BaseProcessor:
    """
    This is a base class for data collection and processing, which provides methods for writing & reading data and
    logging.
    """

    def __init__(
        self,
        data_format: str,
        chunksize: Optional[int] = None,
        n_workers: Optional[int] = None,
        logger_name: Optional[str] = None,
    ):
        self._chunksize = chunksize if chunksize else 1000
        self._n_workers = n_workers if n_workers else 1
        self.logger = BaseProcessor._get_logger(logger_name)
        self.data_format = data_format

        if data_format == "jsonl":
            self._data_manager = JsonlManager()
        else:
            raise NotImplementedError("Current data format is not supported")

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

    def _prepare_outfile(self, out_fname: str, add_data_format: Optional[bool] = True) -> None:
        """
        Does what might be required before saving to chosen output format.
        """
        self._data_manager.prepare_outfile(out_fname, add_data_format=add_data_format)

    def _append_to_outfile(
        self,
        data: Union[pd.DataFrame, List[str]],
        out_fname: str,
        add_data_format: Optional[bool] = True,
    ) -> None:
        """
        Appends current data chunk to chosen output format.
        """
        if isinstance(data, pd.DataFrame):
            self._data_manager.append_to_outfile(data, out_fname, add_data_format=add_data_format)
        else:
            with open(out_fname, mode="a") as f:
                f.writelines(data)

    def _read_input(
        self,
        in_fname: str,
        read_whole: Optional[bool] = None,
        add_data_format: Optional[bool] = True,
        read_lazy: Optional[bool] = None,
        **kwargs,
    ):
        """
        Reads data according to chosen output format.
        """
        if read_lazy:
            return self._data_manager.read_input_lazy(in_fname, add_data_format=add_data_format, **kwargs)
        return self._data_manager.read_input(
            in_fname, add_data_format=add_data_format, chunksize=None if read_whole else self._chunksize, **kwargs
        )

    def prepare(self, in_fname: str, **kwargs) -> None:
        """
        Performs any necessary actions before data processing begins.

        Args:
            in_fname: Path to read input data from.
            **kwargs: Arbitrary keyword arguments.
        """
        pass

    def process(self, chunk: pd.DataFrame, **kwargs) -> Union[pd.DataFrame, List[str]]:
        """
        Implements chunk processing logic.

        Args:
            chunk: Small subset of original dataset.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            List of strings in cases when only one column is necessary for further processing. In other cases, `pd.DataFrame`.
        """
        raise NotImplementedError()

    def __call__(self, in_fname: str, out_fname: str, add_data_format: Optional[bool] = True, **kwargs) -> None:
        """
        Iterates over input data in chunks, processes it in some way and saves results to separate file.

        Args:
            in_fname: Path to read input data from.
            out_fname: Path to save processed data to.
            **kwargs: Arbitrary keyword arguments. Keyword arguments starting from prefix 'prepare_'
                will be passed to method that is called before data processing,
                all others - to method that processes each chunk.
        """
        prepare_kwargs = {key[len("prepare_") :]: value for key, value in kwargs.items() if key.startswith("prepare_")}
        process_kwargs = {key: value for key, value in kwargs.items() if not key.startswith("prepare_")}

        self.logger.info(f"Starting processing {in_fname}")

        self._prepare_outfile(out_fname, add_data_format=add_data_format)
        self.prepare(in_fname, **prepare_kwargs)

        reader = self._read_input(in_fname)
        for chunk in tqdm(reader, leave=False):
            processed_chunk = self.process(chunk, **process_kwargs)
            self._append_to_outfile(processed_chunk, out_fname, add_data_format=add_data_format)

        self.logger.info(f"Finished processing {in_fname}")
