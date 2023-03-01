import os
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import jsonlines
import pandas as pd
from tqdm import tqdm

from ..utils import BaseProcessor


class MetadataProcessor(BaseProcessor):
    """This class is used to finalize metadata about each example that we want to include in public dataset.

    It converts authors' personal information into ids, drops bot authors
    and adds info about license and programming language to each example.

    Args:
        data_format: In which format mined data is saved.
        chunksize: Number of examples to proccess at once (data is read in chunks). Optional, default value is 1000.
        n_workers: Maximum number of concurrently running jobs. Optional, default value is 1 (sequential execution).
        logger_name: Name of logger for this class. Optional, default value is None.
    """

    def __init__(
        self,
        ids_to_commits_map: Dict[int, Dict[str, str]],
        data_format: str,
        chunksize: Optional[int] = None,
        n_workers: Optional[int] = None,
        logger_name: Optional[str] = None,
    ):
        super().__init__(chunksize=chunksize, n_workers=n_workers, data_format=data_format, logger_name=logger_name)

        self._ids_to_commits_map: Dict[int, Dict[str, str]] = ids_to_commits_map
        self._short_examples_to_drop: Dict[str, Set[str]] = defaultdict(set)

        self._authors: Dict[str, Set[Tuple[str, str]]] = defaultdict(set)
        self._train_authors_to_drop: Set[Tuple[str, str]] = set()
        self._bot_authors: Set[Tuple[str, str]] = set()
        self._repo_license_map: Dict[str, str] = {}
        self._repo_language_map: Dict[str, str] = {}
        self._authors_repo_map: Dict[Tuple[str, str, str], int] = {}

    def _get_short_examples(self, root_dir: str, parts: List[str]) -> None:
        """
        Builds a set of extremely short examples (<= 1 unique token).

        SourcererCC can't process these examples, and we don't want them present in a dataset.

        Args:
            diff_fname: Path to diffs file for SourcererCC.
            message_fname: Path to messages file for SourcererCC.
        """
        for data_type in ["diffs", "messages"]:
            for part in parts:
                fname = os.path.join(root_dir, f"{part}_{data_type}.txt")
                self.logger.info(f"Iterating over {fname} to get ids of short examples")
                with open(fname, "r") as f:
                    for line in f:
                        part_idx, idx, num_tokens, num_unique_tokens = (int(i) for i in line.strip().split(",")[:4])
                        if num_unique_tokens <= 1:
                            commit: Dict[str, str] = self._ids_to_commits_map[idx]
                            self._short_examples_to_drop[commit["repo"]].add(commit["hash"])
        self.logger.info(
            f"Number of short examples to drop: {sum(len(self._short_examples_to_drop[key]) for key in self._short_examples_to_drop)}"
        )

    def _get_authors(self, input_dir: str, parts: List[str]) -> None:
        """Builds a set of authors from all given files.

        Args:
            input_dirs: List with paths to all dataset parts.
        """
        for part in parts:
            repos = sorted(os.listdir(os.path.join(input_dir, part)))
            for repo in tqdm(repos, desc=f"Reading authors from {part}"):
                reader = self._data_manager.read_input(
                    os.path.join(input_dir, part, repo, f"commits.{self.data_format}.gz"),
                    compression="gzip",
                    add_data_format=False,
                    chunksize=self._chunksize,
                )

                for chunk in tqdm(reader, desc=f"Reading authors from {repo}", leave=False):
                    authors: List[List[str]] = chunk["author"].tolist()
                    assert all(len(author) == 2 for author in authors)
                    self._authors[part].update(tuple(author) for author in authors)  # type: ignore

            self.logger.info(f"Number of authors in {part}: {len(self._authors[part])}")
        self._train_authors_to_drop = self._authors["train"] & (self._authors["test"] | self._authors["val"])
        self.logger.info(
            f"Number of authors that are present both in train and val/test: {len(self._train_authors_to_drop)}"
        )

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

        all_authors: List[Dict[str, str]] = []
        for part in ["train", "val", "test"]:
            all_authors.extend(
                {"name": author[0], "login": author[0], "email": author[1]} for author in self._authors[part]
            )
        authors_df = pd.DataFrame(all_authors)

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
        self.logger.info(f"Number of bot authors: {len(self._bot_authors)}")

    def _get_authors_mapping(self, authors_map_fname: str) -> None:
        """Reads (author name, author email, repository) <-> (unique id) mapping from given file
        (in our case, it was obtained by name disambiguation).

        Args:
            authors_map_fname: Path to file with (author name, author email, repository) <-> (unique id) mapping.
             Expected to be stored as JSON Lines, where each row contains keys "name", "email", "repo", "id".
        """
        with jsonlines.open(authors_map_fname, "r") as reader:
            d: List[Dict[str, str]] = [line for line in reader]
        self._authors_repo_map = {(author["name"], author["email"], author["repo"]): int(author["id"]) for author in d}

    def _get_repo_metadata(self, repos_metadata_fname: str) -> None:
        """Reads repositories metadata from given file.

        Args:
            repos_metadata_fname: Path to file with repositories metadata (results from GitHub Search tool).
        """
        df = pd.read_json(repos_metadata_fname, orient="records", lines=True)
        self._repo_language_map = {row["name"]: row["mainLanguage"] for _, row in df.iterrows()}
        self._repo_license_map = {row["name"]: row["license"] for _, row in df.iterrows()}

    def prepare(
        self,
        input_dir: str,
        parts: List[str],
        authors_map_fname: str,
        known_bots_fname: str,
        repos_metadata_fname: str,
        deduplication_raw_dir: str,
    ) -> None:
        """
        Runs all necessary preprocessing to drop examples later.

        Args:
            input_dir: Path to root input directory with data.
            parts: List of dataset parts' names.
            authors_map_fname: Path to file with author <-> unique id mapping.
            known_bots_fname: Path to file with combination of open bots datasets.
            repos_metadata_fname: Path to file with repositories metadata.
            deduplication_raw_dir: Path to root directory with preprocessed files for SourcererCC.
        """
        self._get_authors(input_dir=input_dir, parts=parts)
        self._get_bot_authors(known_bots_fname=known_bots_fname)
        self._get_authors_mapping(authors_map_fname=authors_map_fname)
        self._get_repo_metadata(repos_metadata_fname=repos_metadata_fname)
        self._get_short_examples(root_dir=deduplication_raw_dir, parts=parts)

    def _process_chunk(self, chunk: pd.DataFrame, repo: str, part_id: int = -1, **kwargs) -> pd.DataFrame:
        if part_id == -1:
            raise ValueError("Please, pass correct part_id")
        # drop short examples
        chunk = chunk.loc[[cur_hash not in self._short_examples_to_drop[repo] for cur_hash in chunk.hash]].copy()
        # add metadata about repositories
        chunk["language"] = self._repo_language_map[repo.replace("#", "/")]
        chunk["license"] = self._repo_license_map[repo.replace("#", "/")]
        # drop examples from bot authors
        chunk["author"] = chunk["author"].apply(tuple)
        chunk = chunk.loc[~chunk.author.isin(self._bot_authors)].copy()
        # drop examples from train authors that are present in val/test
        if part_id == 1:
            chunk = chunk.loc[~chunk.author.isin(self._train_authors_to_drop)].copy()
        # convert authors to ids
        chunk["author"] = [self._authors_repo_map[(author[0], author[1], repo)] for author in chunk.author]
        return chunk
