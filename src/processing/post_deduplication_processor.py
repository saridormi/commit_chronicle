from typing import Dict, List, Optional, Set, Tuple

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
        self._ids_to_drop: Set[int] = set()

    def _get_outer_clones(self, clones_fname: str) -> List[Dict[str, List]]:
        """Processes clones coming from different dataset parts.

        Args:
            clones_fname: Part to file to read clones from.

        Returns:
            A list where each clone group is represented as dictionary with `ex1` and `ex2` keys. `ex2` is an
             example from val/test part and `ex1` is a set of its clones from train.
        """
        clones_df = pd.read_csv(clones_fname, header=None, names=["part_id1", "id1", "part_id2", "id2"])
        outer_df = clones_df.loc[(clones_df["part_id1"] == 1) & (clones_df["part_id2"].isin([2, 4]))].copy()
        outer_df["ex1"] = list(zip(outer_df.part_id1, outer_df.id1))
        outer_df["ex2"] = list(zip(outer_df.part_id2, outer_df.id2))
        outer_df = outer_df.groupby("ex2").agg(ex1=("ex1", list)).reset_index()
        return outer_df[["ex1", "ex2"]].to_dict(orient="records")

    def _get_inner_clones(self, clones_fname: str, part_id: Optional[int] = 1) -> List[Set[Tuple[int, int]]]:
        """Processes clones coming from the same dataset part.

        Args:
            clones_fname: Part to file to read clones from.
            part_id: Which dataset part to process (default value is 1, corresponding to train).

        Returns:
            A list where each clone group is represented as set of (part_id, id) tuples. Clone groups are disjoint,
             each example appears only in one clone group.
        """
        clones_df = pd.read_csv(clones_fname, header=None, names=["part_id1", "id1", "part_id2", "id2"])
        inner_df = clones_df.loc[(clones_df["part_id1"] == part_id) & (clones_df["part_id2"] == part_id)].copy()
        inner_df["ex1"] = list(zip(inner_df.part_id1, inner_df.id1))
        inner_df["ex2"] = list(zip(inner_df.part_id2, inner_df.id2))
        inner_df = inner_df.groupby("ex1").agg(ex2=("ex2", set)).sort_index(ascending=False)
        new_clones: Dict[Tuple[int, int], Set[Tuple[int, int]]] = {}
        for x, x_clones in tqdm(inner_df["ex2"].iteritems(), total=inner_df.shape[0], desc=f"Processing inner clones"):
            x_clones.add(x)
            if any(x in new_clones[key] for key in new_clones):
                continue
            new_clones[x] = x_clones

        return [new_clones[key] for key in new_clones]

    def _get_outer_ids_to_drop(self, msg_clones_fname: str, diff_clones_fname: str) -> None:
        """Aggregates ids of train examples that are duplicate to val/test examples either in terms of messages or
        in terms of diffs.

        Args:
            msg_clones_fname: Part to file to read message clones from.
            diff_clones_fname: Part to file to read diff clones from.
        """
        # get train clones by messages and by diffs
        msg_clones = self._get_outer_clones(msg_clones_fname)
        diff_clones = self._get_outer_clones(diff_clones_fname)

        # drop all message clones from train
        for group in msg_clones:
            self._ids_to_drop.update([ex[1] for ex in group["ex1"]])
        # drop all diffs clones from train
        for group in diff_clones:
            self._ids_to_drop.update([ex[1] for ex in group["ex1"]])

    def _get_inner_ids_to_drop(self, msg_clones_fname: str, diff_clones_fname: str):
        """Aggregates ids of duplicated examples inside specific dataset part (e.g. train).

        Args:
            msg_clones_fname: Path to file with clones in terms of messages.
            diff_clones_fname: Path to file with clones in terms of diffs.
            out_fname: Path to save resulting full clones.
        """
        # get train clones by messages and by diffs
        msg_clones = self._get_inner_clones(msg_clones_fname, part_id=1)
        diff_clones = self._get_inner_clones(diff_clones_fname, part_id=1)

        # aggregate full train clones
        full_clones = []
        for cur_msg_clones in tqdm(msg_clones, total=len(msg_clones), desc="Processing full clones"):
            for cur_diff_clones in diff_clones:
                cur_full_clones = cur_msg_clones & cur_diff_clones
                if len(cur_full_clones) > 1:
                    full_clones.append(list(cur_full_clones))

        # keep only 1 example from each full clone group
        for group in full_clones:
            self._ids_to_drop.update([ex[1] for ex in group[1:]])

    def prepare(
        self,
        in_fname: str,
        msg_clones_fname: str,
        diff_clones_fname: str,
        is_ready: Optional[bool] = False,
        **kwargs,
    ) -> None:
        """Prepares a set of clones ids.

        Args:
            in_fname: Path to specific input file.
            msg_clones_fname: Path to file with clones in terms of messages.
            diff_clones_fname: Path to file with clones in terms of diffs.
            is_ready: A flag to indicate cases when clones ids are already built. When it is set to True,
                this method doesn't do anything.
        """
        if is_ready:
            self.logger.info(f"Got {len(self._ids_to_drop)} ids to drop")
            return

        self._get_inner_ids_to_drop(msg_clones_fname=msg_clones_fname, diff_clones_fname=diff_clones_fname)
        self._get_outer_ids_to_drop(msg_clones_fname=msg_clones_fname, diff_clones_fname=diff_clones_fname)
        self.logger.info(f"Got {len(self._ids_to_drop)} ids to drop")

    def process(self, chunk: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return chunk.loc[~chunk["id"].isin(self._ids_to_drop)]
