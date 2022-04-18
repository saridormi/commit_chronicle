import os
from typing import List, Optional, Set

import jsonlines
import pandas as pd
from tqdm import tqdm

from ..utils import BaseProcessor


class PostDeduplicationProcessor(BaseProcessor):
    """This class is used to drop duplicates found by code clones detection tool SourcererCC.

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
        self._train_full_clones: Set[str] = set()
        self._ids_to_drop: Set[int] = set()

    def _extract_metadata(self, in_path: str, deduplication_dir: str, parts: List[str]) -> None:
        """Saves commits metadata (author, timestamp, repo, hash) from main dataset files to separate files.

        Args:
            in_path: Path to folder where input data is stored.
            parts: List of all parts in input dataset.
            deduplication_dir: Path to folder where files with found clones are stored.
        """

        full_out_fname = os.path.join(deduplication_dir, "metadata")
        self._prepare_outfile(full_out_fname)

        for i, part in enumerate(parts):
            self.logger.info(f"Extracting metadata from {part}")

            part_out_fname = os.path.join(deduplication_dir, f"{part}_metadata")
            self._prepare_outfile(part_out_fname)

            reader = self._read_input(os.path.join(in_path, part))

            for chunk in tqdm(reader, desc=f"Iterating over {part} to extract metadata"):
                chunk["project_id"] = i + 1
                self._append_to_outfile(
                    chunk[["project_id", "id", "author", "date", "hash", "repo"]],
                    part_out_fname,
                )
                self._append_to_outfile(
                    chunk[["project_id", "id", "author", "date", "hash", "repo"]],
                    full_out_fname,
                )

    def _add_metadata(self, in_fname: str, out_fname: str, deduplication_dir: str):
        """Adds metadata to each pair of clones.

        Initially clones are created in a format `project_id1,sample_id1,project_id2,sample_id2`, we add metadata about
            each example for further use.
        """
        self.logger.info(f"Adding metadata to {in_fname}")
        df = self._read_input(os.path.join(deduplication_dir, "metadata"), read_whole=True).sort_values(
            by=["project_id", "id"]
        )
        df.sort_index(axis=1, inplace=True)
        sorted_cols = {col: i for i, col in enumerate(df.columns.tolist())}

        # fast indexing on SourcererCC ids
        indexes = {}
        for idx, row in tqdm(df.iterrows()):
            indexes[(row["project_id"], row["id"])] = idx
        data = df.to_numpy()

        # clear target file
        open(out_fname, "w").close()

        metadata = []

        with open(in_fname, "r") as file:
            for i, line in tqdm(enumerate(file), desc=f"Iterating over {in_fname} to add metadata"):
                pr_1, s_1, pr_2, s_2 = (int(j) for j in line.strip().split(","))
                ex1 = data[indexes[(pr_1, s_1)]]
                ex2 = data[indexes[(pr_2, s_2)]]

                metadata.append(
                    {
                        "part_id1": ex1[sorted_cols["project_id"]],
                        "id1": ex1[sorted_cols["id"]],
                        "repo1": ex1[sorted_cols["repo"]],
                        "hash1": ex1[sorted_cols["hash"]],
                        "part_id2": ex2[sorted_cols["project_id"]],
                        "id2": ex2[sorted_cols["id"]],
                        "repo2": ex2[sorted_cols["repo"]],
                        "hash2": ex2[sorted_cols["hash"]],
                    }
                )

                if i % self._chunksize == 0:
                    with jsonlines.open(out_fname, mode="a") as writer:
                        writer.write_all(metadata)
                    metadata = []

        if len(metadata) > 0:
            with jsonlines.open(out_fname, mode="a") as writer:
                writer.write_all(metadata)

    def _get_full_clones(self, msg_clones_fname: str, diff_clones_fname: str, out_fname: str):
        """Builds a set of ids of examples from train which are completely identical to examples from train/val/test
            (both diffs and messages are the same).

        Args:
            msg_clones_fname: Path to file with clones in terms of messages.
            diff_clones_fname: Path to file with clones in terms of diffs.
            out_fname: Path to save resulting full clones.
        """
        # get train clones by messages
        train_msgs_clones = set()
        with jsonlines.open(msg_clones_fname, "r") as reader:
            for line in tqdm(reader, desc="Reading message clones"):
                if line["part_id1"] == 1 and line["part_id2"] != 1:
                    train_msgs_clones.add(
                        f"{line['part_id1']},{line['id1']},{line['repo1']},{line['hash1']},{line['part_id2']},{line['id2']},{line['repo2']},{line['hash2']}\n"
                    )
                elif line["part_id2"] == 1 and line["part_id1"] != 1:
                    train_msgs_clones.add(
                        f"{line['part_id2']},{line['id2']},{line['repo2']},{line['hash2']},{line['part_id1']},{line['id1']},{line['repo1']},{line['hash1']}\n"
                    )

        # get train clones by diffs
        train_diffs_clones = set()
        with jsonlines.open(diff_clones_fname, "r") as reader:
            for line in tqdm(reader, desc="Reading diff clones"):
                if line["part_id1"] == 1 and line["part_id2"] != 1:
                    train_diffs_clones.add(
                        f"{line['part_id1']},{line['id1']},{line['repo1']},{line['hash1']},{line['part_id2']},{line['id2']},{line['repo2']},{line['hash2']}\n"
                    )
                elif line["part_id2"] == 1 and line["part_id1"] != 1:
                    train_diffs_clones.add(
                        f"{line['part_id2']},{line['id2']},{line['repo2']},{line['hash2']},{line['part_id1']},{line['id1']},{line['repo1']},{line['hash1']}\n"
                    )

        self._train_full_clones = train_msgs_clones.intersection(train_diffs_clones)

        with open(out_fname, "w") as file:
            file.writelines(list(self._train_full_clones))

        self._ids_to_drop = set(int(pair.split(",")[1]) for pair in self._train_full_clones)
        self.logger.info(f"Got {len(self._ids_to_drop)} clones ids to drop")

    def prepare(
        self,
        in_fname: str,
        in_path: str,
        parts: List[str],
        msg_clones_fname: str,
        diff_clones_fname: str,
        deduplication_dir: str,
        is_ready: Optional[bool] = False,
        **kwargs,
    ) -> None:
        """Prepares a set of ids of fully identical entries between train and validation/test.

        During this process, metadata is extracted from input dataset and added to clones ids.

        Args:
            in_fname: Path to specific input file.
            in_path: Path to root folder with input data.
            parts: List of all parts in input dataset.
            msg_clones_fname: Path to file with clones in terms of messages.
            diff_clones_fname: Path to file with clones in terms of diffs.
            deduplication_dir: Path to folder where files with found clones are stored.
            is_ready: A flag to indicate cases when clones ids are already built. When it is set to True,
                this method doesn't do anything.
        """
        if is_ready:
            return

        self._extract_metadata(in_path, deduplication_dir, parts)

        self._add_metadata(
            in_fname=os.path.join(deduplication_dir, msg_clones_fname),
            out_fname=os.path.join(deduplication_dir, f"{msg_clones_fname.split('.')[0]}_metadata.txt"),
            deduplication_dir=deduplication_dir,
        )

        self._add_metadata(
            in_fname=os.path.join(deduplication_dir, diff_clones_fname),
            out_fname=os.path.join(deduplication_dir, f"{diff_clones_fname.split('.')[0]}_metadata.txt"),
            deduplication_dir=deduplication_dir,
        )

        self._get_full_clones(
            msg_clones_fname=os.path.join(deduplication_dir, f"{msg_clones_fname.split('.')[0]}_metadata.txt"),
            diff_clones_fname=os.path.join(deduplication_dir, f"{diff_clones_fname.split('.')[0]}_metadata.txt"),
            out_fname=os.path.join(deduplication_dir, "full_clones_metadata.txt"),
        )

    def process(self, chunk: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return chunk.loc[~chunk["id"].isin(self._ids_to_drop)]
