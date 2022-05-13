import json
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd
from tqdm import tqdm

from ..utils import BaseProcessor


class FinalProcessor(BaseProcessor):
    """This class is used to perform several simple operations with dataset.

    It deletes `mods` field, drops examples with empty diffs, converts authors' personal information into unique ids
    and adds license type for each repository.

    Args:
        data_format: In which format mined data is saved.
        chunksize: Number of examples to proccess at once (data is read in chunks). Optional, default value is 1000.
        n_workers: Maximum number of concurrently running jobs. Optional, default value is 1 (sequential execution).
        logger_name: Name of logger for this class. Optional, default value is None.
    """

    def __init__(
        self,
        data_format: str,
        chunksize: Optional[int] = None,
        n_workers: Optional[int] = None,
        logger_name: Optional[str] = None,
    ):
        super().__init__(chunksize=chunksize, n_workers=n_workers, data_format=data_format, logger_name=logger_name)
        self._authors: Set[Tuple[str, str]] = set()
        self._authors_map: Dict[Tuple[str, str], int] = {}
        self._repo_license_map: Dict[str, str] = {}

    def _get_authors(self, in_fname: str, in_fnames: List[str]) -> None:
        """Builds an unique set of authors.

        Currently, all work is done when `train` part is given. For other parts, this method doesn't do anything.

        Args:
            in_fname: Path to current dataset part.
            in_fnames: List with paths to all dataset parts. When `train` part is given, authors will be gathered
                from these files.
        """
        if "train" in in_fname:
            for fname in in_fnames:
                if "train" in fname:
                    fname = in_fname
                reader = self._read_input(fname)
                for chunk in tqdm(reader, desc=f"Reading authors from {fname}", leave=False):
                    self._authors.update((tuple(author) for author in chunk["author"].tolist()))

            self._authors_map = {author: i for i, author in enumerate(self._authors)}

    def _get_licenses(self, license_in_fname: str) -> None:
        with open(license_in_fname, "r") as file:
            self._repo_license_map = json.load(file)

    def prepare(self, in_fname: str, license_in_fname: str, in_fnames: List[str], **kwargs) -> None:
        self._get_authors(in_fname=in_fname, in_fnames=in_fnames)
        self._get_licenses(license_in_fname=license_in_fname)

    def process(self, chunk: pd.DataFrame, **kwargs) -> pd.DataFrame:
        chunk = chunk.drop(columns="mods")
        chunk = chunk.loc[chunk["diff"].str.len() > 0]
        chunk["author"] = [self._authors_map[tuple(author)] for author in chunk["author"].tolist()]
        chunk["license"] = [self._repo_license_map[repo] for repo in chunk["repo"].tolist()]
        return chunk
