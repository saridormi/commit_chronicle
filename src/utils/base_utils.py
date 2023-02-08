import gzip
import logging
import os
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union

import jsonlines
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm


class BaseManager(ABC):
    """
    This is a base class for writing & reading data.
    """

    @abstractmethod
    def zip_file(self, out_fname: str, add_data_format: Optional[bool] = True):
        pass

    @abstractmethod
    def prepare_outfile(self, out_fname: str, add_data_format: Optional[bool] = True) -> None:
        """
        Does what might be required before saving to chosen output format.
        (e.g. write header in case of csv files)
        """
        pass

    @abstractmethod
    def append_to_outfile(
        self, data: Union[pd.DataFrame, List[Dict], List[str]], out_fname: str, add_data_format: Optional[bool] = True
    ) -> None:
        """
        Appends current data chunk to chosen output format.
        """
        pass

    @abstractmethod
    def read_input(self, in_fname: str, add_data_format: Optional[bool] = True, **kwargs):
        """
        Reads data according to chosen output format (accessing full dataset/reading in chunks).
        """
        pass


class JsonlManager(BaseManager):
    """
    This is a class for writing & reading jsonl data.
    """

    def zip_file(self, out_fname: str, add_data_format: Optional[bool] = True):
        """
        Uses gzip to compress target file.
        """
        if add_data_format:
            out_fname += ".jsonl"
        with open(out_fname, "rb") as f_in, gzip.open(f"{out_fname}.gz", "wb") as f_out:
            f_out.writelines(f_in)
        os.remove(out_fname)

    def prepare_outfile(self, out_fname: str, add_data_format: Optional[bool] = True) -> None:
        """
        Clears target file.
        """
        if add_data_format:
            out_fname = f"{out_fname}.jsonl"

        open(out_fname, mode="w").close()

    def append_to_outfile(
        self,
        data: Union[pd.DataFrame, List[Dict], List[str]],
        out_fname: str,
        add_data_format: Optional[bool] = True,
    ) -> None:
        """
        Appends current data chunk.
        """
        if isinstance(data, list) and all(isinstance(d, str) for d in data):
            with open(out_fname, mode="a") as f:
                f.writelines(data)  # type: ignore[arg-type]
            return
        elif isinstance(data, pd.DataFrame):
            data = data.to_dict(orient="records")
        elif isinstance(data, list) and all(isinstance(d, dict) for d in data):
            pass
        else:
            raise ValueError(
                "Unknown data format: expected either pd.DataFrame, list of dictionaries or list of strings."
            )

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


def get_logger(logger_name):
    """
    Workaround for logging with joblib (based on https://github.com/joblib/joblib/issues/1017)
    """
    logger = logging.getLogger(logger_name)
    if len(logger.handlers) == 0:
        logger.setLevel(logging.INFO)
        sh = logging.StreamHandler()
        sh.setFormatter(logging.Formatter("%(asctime)s %(levelname)-8s %(message)s"))
        fh = logging.FileHandler(f"{logger_name}.log", mode="a")
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)-8s %(message)s"))
        logger.addHandler(sh)
        logger.addHandler(fh)
    return logger


class BaseProcessor(ABC):
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
        self._logger_name = logger_name
        self.data_format = data_format

        if data_format == "jsonl":
            self._data_manager = JsonlManager()
        else:
            raise NotImplementedError("Passed data format is not supported")

    @property
    def logger(self):
        return get_logger(self._logger_name)

    def _prepare_outfile(self, out_fname: str, add_data_format: Optional[bool] = True) -> None:
        """
        Does what might be required before saving to chosen output format.
        """
        self._data_manager.prepare_outfile(out_fname, add_data_format=add_data_format)

    def _append_to_outfile(
        self,
        data: Union[pd.DataFrame, List[Dict], List[str]],
        out_fname: str,
        add_data_format: Optional[bool] = True,
    ) -> None:
        """
        Appends current data chunk to chosen output format.
        """
        self._data_manager.append_to_outfile(data, out_fname, add_data_format=add_data_format)

    def _read_input(
        self,
        in_fname: str,
        add_data_format: bool = True,
        read_whole: Optional[bool] = None,
        **kwargs,
    ):
        """
        Reads data according to chosen output format.
        """
        return self._data_manager.read_input(
            in_fname, add_data_format=add_data_format, chunksize=None if read_whole else self._chunksize, **kwargs
        )

    @abstractmethod
    def _process_chunk(self, chunk: pd.DataFrame, repo: str, **kwargs) -> pd.DataFrame:
        """
        Implements single chunk processing logic.

        Args:
            chunk: Dataframe with commits, a subset of possibly huge file. Its size is defined by `_chunksize` field.
            repo: Name of current repository.
            **kwargs: Arbitrary keyword arguments.

        Note:
            `repo` is a directory name, so it's not a `org/name` structure from GitHub, but rather `org#repo`.

        Returns:
            Processed chunk.
        """
        pass

    def _process_repo(self, input_dir: str, output_dir: str, repo: str, part: str, **kwargs) -> None:
        """
        Implements single repo processing logic:
         * iterate over repo commits in chunks
         * do something meaningful to each chunk
         * append processed chunk to file in repo-specific output directory

        Args:
            input_dir: Path to root input directory with data.
            output_dir: Path to root output directory with data.
            repo: Name of current repository.
            part: Name of current dataset part. Will be passed to `_process_chunk` method.
            **kwargs: Arbitrary keyword arguments. Will be passed to `_process_chunk` method.
        """
        os.makedirs(os.path.join(output_dir, repo), exist_ok=True)
        out_fname = os.path.join(output_dir, repo, "commits")
        self._data_manager.prepare_outfile(out_fname)

        self.logger.info(f"[{repo}] Start processing")

        reader = self._data_manager.read_input(
            os.path.join(input_dir, repo, f"commits.{self.data_format}.gz"),
            compression="gzip",
            add_data_format=False,
            chunksize=self._chunksize,
        )

        for chunk in tqdm(reader, desc=f"Processing {repo}", leave=False):
            chunk = self._process_chunk(chunk, repo=repo, **kwargs)
            self._data_manager.append_to_outfile(chunk, out_fname)

        self.logger.debug(f"[{repo}] Zipping file")
        self._data_manager.zip_file(out_fname)
        self.logger.info(f"[{repo}] Finish processing")

    def __call__(self, input_dir: str, output_dir: str, part: str, **kwargs) -> None:
        """
        Iterates over input data, processes commits from each repository independently,
        saves results to separate directory.

        Args:
            input_dir: Path to root input directory with data.
            output_dir: Path to root output directory with data.
            part: Name of current dataset part. Will be passed to `_process_repo` method.
            **kwargs: Arbitrary keyword arguments. Will be passed to `_process_repo` method.
        Note:
            input_dir and output_dir should already include `part` folder. E.g. if we have root directory data and we want to
            process train, `input_dir` should be `data/train`
        """
        repos = sorted(os.listdir(input_dir))

        with Parallel(self._n_workers) as pool:
            pool(
                delayed(self._process_repo)(input_dir=input_dir, output_dir=output_dir, part=part, repo=repo, **kwargs)
                for repo in repos
            )
