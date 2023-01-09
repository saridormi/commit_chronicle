import json
from typing import Dict, List, Optional, Set, Tuple

import jsonlines
import pandas as pd
from tqdm import tqdm

from ..utils import BaseProcessor


class MetadataProcessor(BaseProcessor):
    """This class is used to finalize metadata about each example that we want to include in public dataset.

    It converts authors' personal information into ids, drops bot authors
    and adds info about license type to each example.

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
        self._bot_authors: Set[Tuple[str, str]] = set()
        self._repo_license_map: Dict[str, str] = {}
        self._authors_repo_map: Dict[Tuple[str, str, str], int] = {}
        self._ids_history_position_map: Dict[int, int] = {}

    def _get_authors(self, in_fnames: List[str]) -> None:
        """Builds a set of authors from all given files.

        Args:
            in_fnames: List with paths to all dataset parts.
        """
        for fname in in_fnames:
            reader = self._read_input(fname)
            for chunk in tqdm(reader, desc=f"Reading authors from {fname}", leave=False):
                authors: List[List[str]] = chunk["author"].tolist()
                assert all(len(author) == 2 for author in authors)
                self._authors.update(tuple(author) for author in authors)  # type: ignore

        self.logger.info(f"Total number of authors: {len(self._authors)}")

    def _get_bot_authors(self, known_bots_fname: str) -> None:
        """Aggregates ids of bot authors. Two approaches to detect bots are used:

        * Open datasets from papers on boot detection BIMAN, BoDeGHa (BoDeGiC), BotHunter.

          - BoDeGHa and BotHunter provide GitHub usernames of bot accounts. We used it to query GitHub API and
            retrieve `name`, `login` and `email` fields.
          - BIMAN provides names and emails from bot commits.

        * Consider authors with names ending with either `bot` or `[bot]` suffixes bots.

        Args:
            known_bots_fname: Path to file with bot information. Expected to be stored as JSONLines.
        """
        assert self._authors

        authors_df = pd.DataFrame(
            [{"name": author[0], "login": author[0], "email": author[1]} for author in self._authors]
        )

        # search for our authors in open bots datasets
        # our format is (name, email) from commits, name is compared both against `name` and against `login` fields
        known_bots_df = pd.read_json(known_bots_fname, orient="records", lines=True)
        overlaps = []
        for col in ["name", "login", "email"]:
            overlap = authors_df.loc[(authors_df[col].str.len() > 0) & (authors_df[col].isin(known_bots_df[col]))]
            overlaps.append(overlap)
        overlaps = pd.concat(overlaps, axis=0, ignore_index=True)

        # find authors with names ending on `bot` or on `[bot]`
        suffixes = authors_df.loc[
            authors_df.name.str.lower().str.endswith("bot") | authors_df.name.str.lower().str.endswith("[bot]")
        ]

        bot_authors = pd.concat([overlaps, suffixes], axis=0, ignore_index=True)
        self._bot_authors = set((row["name"], row["email"]) for _, row in bot_authors.iterrows())

    def _get_authors_mapping(self, authors_map_fname: str) -> None:
        """Reads (author name, author email, repository) <-> (unique id) mapping from given file.

        Args:
            authors_map_fname: Path to file with (author name, author email, repository) <-> (unique id) mapping.
             Expected to be stored as JSON Lines, where each row contains keys "name", "email", "repo", "id".
        """
        with jsonlines.open(authors_map_fname, "r") as reader:
            d: List[Dict[str, str]] = [line for line in reader]
        self._authors_repo_map = {(author["name"], author["email"], author["repo"]): int(author["id"]) for author in d}

    def _get_repo_license_mapping(self, licenses_fname: str) -> None:
        """Reads repository <-> license mapping from given file.

        Args:
            licenses_fname: Path to file with repository <-> license mapping. Expected to be stored as JSON.
        """
        with open(licenses_fname, "r") as file:
            self._repo_license_map = json.load(file)

    def _get_history_positions(self, in_fname: str) -> None:
        """Groups dataset by authors and aggregates position in author's history for each example.

        Args:
            in_fname: Path to input file.

        Note:
            Assumes that commits from each author are already in correct chronological order,
            which is true for default PyDriller configuration.
        """
        assert self._authors_repo_map
        data = []

        if not in_fname.endswith(".jsonl") and "." not in in_fname:
            in_fname += ".jsonl"
        with jsonlines.open(in_fname, "r") as reader:
            for row in tqdm(reader, desc=f"Iterating over {in_fname} to construct history"):
                # an issue related with name disambiguation: there was a case when developer's personal account was
                # considered the same as some bot from the same repo, so including bot authors at this points led to
                # incorrect "pos_in_history" construction ><
                if (row["author"][0], row["author"][1]) in self._bot_authors:
                    continue
                data.append(
                    {
                        "author": self._authors_repo_map[(row["author"][0], row["author"][1], row["repo"])],
                        "id": int(row["id"]),
                    }
                )

        df = pd.DataFrame(data)
        df["pos_in_history"] = df.groupby("author").cumcount()
        self._ids_history_position_map = {row["id"]: row["pos_in_history"] for _, row in df.iterrows()}

    def prepare(  # type: ignore[override]
        self,
        in_fname: str,
        in_fnames: List[str],
        authors_map_fname: str,
        known_bots_fname: str,
        licenses_fname: str,
        is_ready: bool = False,
        *args,
        **kwargs,
    ) -> None:
        if not is_ready:
            self._get_authors(in_fnames=in_fnames)
            self._get_bot_authors(known_bots_fname=known_bots_fname)
            self._get_authors_mapping(authors_map_fname=authors_map_fname)
            self._get_repo_license_mapping(licenses_fname=licenses_fname)
        self._get_history_positions(in_fname=in_fname)

    def process(self, chunk: pd.DataFrame, **kwargs) -> pd.DataFrame:
        # add license info
        chunk["license"] = [self._repo_license_map[repo] for repo in chunk["repo"].tolist()]
        chunk["author"] = chunk["author"].apply(tuple)
        # drop examples from bot authors
        chunk = chunk.loc[~chunk.author.isin(self._bot_authors)].copy()
        # convert authors to ids
        chunk["temp"] = [(author[0], author[1], repo) for author, repo in zip(chunk["author"], chunk["repo"])]
        chunk["author"] = chunk["temp"].map(self._authors_repo_map)
        # add position in author's history to each example
        chunk["pos_in_history"] = chunk["id"].map(self._ids_history_position_map)
        return chunk.drop(columns="temp")