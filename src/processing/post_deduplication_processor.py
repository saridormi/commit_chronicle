import os
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Set, Tuple

import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from ..utils import BaseProcessor, CloneGroup


class PostDeduplicationProcessor(BaseProcessor):
    """This class is used to drop duplicates found by code clones detection tool SourcererCC.

    Args:
        data_format: In which format mined data is saved.
        ids_to_commits_map: Mapping from surrogate ids to real commits (commit is represented as dict with keys `repo`, `hash`).
        chunksize: Number of examples to process at once (data is read in chunks). Optional, default value is 1000.
        n_workers: Maximum number of concurrently running jobs. Optional, default value is 1 (sequential execution).
        logger_name: Name of logger for this class. Optional, default value is None.
    """

    def __init__(
        self,
        data_format: str,
        ids_to_commits_map: Dict[int, Dict[str, str]],
        chunksize: Optional[int] = None,
        n_workers: Optional[int] = None,
        logger_name: Optional[str] = None,
    ):
        super().__init__(chunksize=chunksize, n_workers=n_workers, data_format=data_format, logger_name=logger_name)
        self._ids_to_commits_map = ids_to_commits_map
        self._inner_clones_to_drop: Dict[str, Set[str]] = defaultdict(set)
        self._outer_clones_to_drop: Dict[str, Set[str]] = defaultdict(set)

    def _get_outer_clones_exact_hash(
        self, clones_fname: str, inner_part_id: int, outer_part_ids: Sequence[int]
    ) -> List[CloneGroup]:
        """Processes clones between different dataset parts. One of the parts is "inner", and the other parts are "outer".
        Later, we want to drop clones only from the "inner" part.

        The primary use-case of this method is to find clones between train and val/test – we want to drop them
        from train.

        This version expects clones to be already aggregated into disjoint clone groups and
        stored in JSONLines with key "clones". It is supposed to be used with ExactHashProcessor output.

        Args:
            clones_fname: Part to file to read clones from.
            inner_part_id: "Inner" dataset part: we intend to drop clones from this part.
            outer_part_ids: Sequence of "outer" dataset parts: we do not intend to drop clones from these parts.

        Returns:
            A list of CloneGroup. In this case, clone groups are divided into a root and clones,
              where root is the "outer" example and clones are its "inner" clones.
        """
        self.logger.info(f"Processing outer clones from {clones_fname}")
        clones = pd.read_json(clones_fname, orient="records", lines=True)
        outer_clone_groups = []
        for clone_group in tqdm(clones.clones, desc=f"Processing outer clones for {inner_part_id}"):
            inner_clones = set([tuple(clone) for clone in clone_group if clone[0] == inner_part_id])
            outer_clones = [tuple(clone) for clone in clone_group if clone[0] in outer_part_ids]
            for outer_clone in outer_clones:
                outer_clone_groups.append(CloneGroup(clone_root=outer_clone, clones=inner_clones))  # type: ignore[arg-type]
        return outer_clone_groups

    def _get_outer_clones(
        self, clones_fname: str, inner_part_id: int, outer_part_ids: Sequence[int]
    ) -> List[CloneGroup]:
        """Processes clones between different dataset parts. One of the parts is "inner", and the other parts are "outer".
        Later, we want to drop clones only from the "inner" part.

        The primary use-case of this method is to find clones between train and val/test – we want to drop them
        from train.

        This version expects clones to be stored in text file with pairs in format "{part_id1},{id1},{part_id2},{id2}".
        It is supposed to be used with SourcererCC output. Note that it won't work for large-scale deduplication
        (e.g. 700M clone pairs), as it loads everything into memory.

        Args:
            clones_fname: Part to file to read clones from.
            inner_part_id: "Inner" dataset part: we intend to drop clones from this part.
            outer_part_ids: Sequence of "outer" dataset parts: we do not intend to drop clones from these parts.

        Returns:
            A list of CloneGroup. In this case, clone groups are divided into a root and clones,
              where root is the "outer" example and clones are its "inner" clones.
        """
        clones_df = pd.read_csv(clones_fname, header=None, names=["part_id1", "id1", "part_id2", "id2"])

        # limit to inner/outer pairs (or outer/inner, they are also present!)
        outer_df = clones_df.loc[
            (clones_df["part_id1"] == inner_part_id) & (clones_df["part_id2"].isin(outer_part_ids))
            | ((clones_df["part_id1"].isin(outer_part_ids)) & (clones_df["part_id2"] == inner_part_id))
        ].copy()

        # make sure that inner part is always id1
        def swap_ids(row):
            if row["part_id1"] != inner_part_id:
                row["part_id1"], row["part_id2"] = row["part_id2"], row["part_id1"]
                row["id1"], row["id2"] = row["id2"], row["id1"]
            return row

        outer_df = outer_df.apply(swap_ids, axis=1)
        outer_df["inner_example"] = list(zip(outer_df.part_id1, outer_df.id1))
        outer_df["outer_example"] = list(zip(outer_df.part_id2, outer_df.id2))
        outer_df = outer_df.groupby("outer_example").agg(inner_examples=("inner_example", set)).reset_index()
        return [
            CloneGroup(clone_root=outer_ex, clones=inner_examples)
            for inner_examples, outer_ex in zip(outer_df.inner_examples, outer_df.outer_example)
        ]

    def _get_inner_clones_identical_exact_hash(self, clones_fname: str, part_id: int) -> List[CloneGroup]:
        """Processes clones coming from the same dataset part. This method is used only for 100% clones, and it relies
        on the assumption that relation "being a clone" is transitive.

        This version expects clones to be already aggregated into disjoint clone groups and
        stored in JSONLines with key "clones". It is supposed to be used with ExactHashProcessor output.

        Args:
            clones_fname: Part to file to read clones from.
            part_id: Which dataset part to process.
        """
        self.logger.info(f"Processing inner clones from {clones_fname} (part {part_id})")
        clones_df = pd.read_json(clones_fname, orient="records", lines=True)
        clone_groups: List[Set[Tuple[int, int]]] = [
            set([tuple(example) for example in clone_group if example[0] == part_id])  # type: ignore[misc]
            for clone_group in clones_df.clones.tolist()
        ]
        return [CloneGroup(clone_root=None, clones=clone_group) for clone_group in clone_groups if len(clone_group) > 1]

    def _get_inner_clones_identical(self, clones_fname: str, part_id: int) -> List[CloneGroup]:
        """Processes clones coming from the same dataset part. This method is used only for 100% clones, and it relies
        on the assumption that relation "being a clone" is transitive.

        This version expects clones to be stored in text file with pairs in format "{part_id1},{id1},{part_id2},{id2}".
        It is supposed to be used with SourcererCC output. Note that it won't work for large-scale deduplication
        (e.g. 700M clone pairs), as it loads everything into memory.

        Args:
            clones_fname: Part to file to read clones from.
            part_id: Which dataset part to process.

        Returns:
            A list of CloneGroup. In this case, clone groups do not have a root, because being a 100% clone is an
              equivalence relation and these groups are basically equivalence classes. Also, clone groups are disjoint,
              each example appears only in one clone group.
        """
        clones_df = pd.read_csv(clones_fname, header=None, names=["part_id1", "id1", "part_id2", "id2"])
        # limit to inner pairs
        inner_df = clones_df.loc[(clones_df["part_id1"] == part_id) & (clones_df["part_id2"] == part_id)].copy()
        inner_df["ex1"] = list(zip(inner_df.part_id1, inner_df.id1))
        inner_df["ex2"] = list(zip(inner_df.part_id2, inner_df.id2))
        # group by first elements of the pairs
        # initially, for each ex1 it aggregates only clones with smaller ids
        # so we loop over the pairs again and unite current set of clones with clones with bigger ids
        inner_df = inner_df.groupby("ex1").agg(ex2=("ex2", set)).sort_index(ascending=False)
        new_clones: Dict[Tuple[int, int], Set[Tuple[int, int]]] = {}
        for x, x_clones in tqdm(inner_df["ex2"].iteritems(), total=inner_df.shape[0], desc="Processing inner clones"):
            flag = False
            for key in new_clones:
                if x in new_clones[key] or x_clones & new_clones[key]:
                    if flag:
                        raise ValueError(f"{x} is present in more than one clone group!")
                    new_clones[key] |= x_clones
                    flag = True
            if not flag:
                new_clones[x] = x_clones

        for key in new_clones:
            new_clones[key].add(key)

        return [CloneGroup(clone_root=None, clones=new_clones[key]) for key in new_clones]

    def _get_full_inner_clones_identical(
        self, msg_clones: List[CloneGroup], diff_clones: List[CloneGroup]
    ) -> List[CloneGroup]:
        """Aggregates a list of full clones (both in terms of diffs and a messages). This method is used only
        for 100% clones, and it relies on the assumption that relation "being a clone" is transitive.

        Args:
            msg_clones: A list of clones in terms of messages.
            diff_clones: A list of clones in terms of diffs.

        Returns:
            A list of CloneGroup. In this case, clone groups do not have a root, because being a 100% clone is an
              equivalence relation and these groups are basically equivalence classes. Also, clone groups are disjoint,
              each example appears only in one clone group.
        """
        full_clones = []
        for cur_msg_clones in tqdm(msg_clones, total=len(msg_clones), desc="Aggregating full inner clones"):
            for cur_diff_clones in diff_clones:
                cur_full_clones = cur_msg_clones.clones & cur_diff_clones.clones
                if len(cur_full_clones) > 1:
                    full_clones.append(CloneGroup(clone_root=None, clones=cur_full_clones))
        return full_clones

    def _get_inner_clones_similar(self, clones_fname: str, part_id: int) -> List[CloneGroup]:
        """Processes clones coming from the same dataset part. This method is used for similar clones, and it doesn't
        rely on the assumption that relation "being a clone" is transitive.

        This version expects clones to be stored in text file with pairs in format "{part_id1},{id1},{part_id2},{id2}".
        It is supposed to be used with SourcererCC output. Note that it won't work for large-scale deduplication
        (e.g. 700M clone pairs), as it loads everything into memory.

        Args:
            clones_fname: Part to file to read clones from.
            part_id: Which dataset part to process.

        Returns:
            A list of CloneGroup. In this case, clone groups are divided into a root and clones, where clones are the
             clones of this root. Also, clone groups are NOT disjoint.
        """
        clones_df = pd.read_csv(clones_fname, header=None, names=["part_id1", "id1", "part_id2", "id2"])

        # limit to inner pairs
        inner_df = clones_df.loc[(clones_df["part_id1"] == part_id) & (clones_df["part_id2"] == part_id)].copy()
        inner_df["ex1"] = list(zip(inner_df.part_id1, inner_df.id1))
        inner_df["ex2"] = list(zip(inner_df.part_id2, inner_df.id2))

        d = inner_df.groupby("ex1").agg(clones_group=("ex2", set)).sort_index(ascending=True)
        new_clones: Dict[Tuple[int, int], Set[Tuple[int, int]]] = {}
        for x, x_clones in tqdm(
            d["clones_group"].iteritems(), total=d.shape[0], desc="Processing inner clones (step 1)"
        ):
            for key in new_clones:
                if key in x_clones:
                    new_clones[key].add(x)
            new_clones[x] = x_clones

        # if any example was only present as "ex2", its clone group won't appear, e.g: "1,3, 1,2\n1,5, 1,2"
        d = inner_df.groupby("ex2").agg(clones_group=("ex1", set)).sort_index(ascending=True)
        for x, x_clones in tqdm(
            d["clones_group"].iteritems(), total=d.shape[0], desc="Processing inner clones (step 2)"
        ):
            if x not in new_clones:
                new_clones[x] = x_clones
        return [CloneGroup(clone_root=key, clones=new_clones[key]) for key in new_clones]

    def _get_full_inner_clones_similar(
        self, msg_clones: List[CloneGroup], diff_clones: List[CloneGroup]
    ) -> List[CloneGroup]:
        """Aggregates a list of full clones (both in terms of diffs and a messages). This method is used
        for similar clones, and it doesn't rely on the assumption that relation "being a clone" is transitive.

        Args:
            msg_clones: A list of clones in terms of messages.
            diff_clones: A list of clones in terms of diffs.

        Returns:
            A list of CloneGroup. In this case, clone groups are divided into a root and clones, where clones are the
             clones of this root. Also, clone groups are NOT disjoint.
        """

        def get_full_clones(cur_msg_clones, cur_diff_clones):
            if cur_diff_clones.clone_root == cur_msg_clones.clone_root:
                cur_full_clones = cur_msg_clones.clones & cur_diff_clones.clones
                if len(cur_full_clones) > 0:
                    return CloneGroup(clone_root=cur_diff_clones.clone_root, clones=cur_full_clones)
            return None

        self.logger.info("Aggregating full clones")

        with Parallel(self._n_workers) as pool:
            full_clones = pool(
                delayed(get_full_clones)(cur_msg_clones, cur_diff_clones)
                for cur_msg_clones in msg_clones
                for cur_diff_clones in diff_clones
            )
        return [g for g in full_clones if g]

    def _get_outer_ids_to_drop(
        self,
        msg_clones_fname: str,
        diff_clones_fname: str,
        inner_part_id: int,
        outer_part_ids: Sequence[int],
        use_exact_hash: bool,
    ) -> None:
        """Aggregates ids of "inner" part (e.g. train) examples that are duplicate to "outer" parts (e.g. val/test) examples
        either in terms of messages or in terms of diffs.

        Args:
            msg_clones_fname: Part to file to read message clones from.
            diff_clones_fname: Part to file to read diff clones from.
            inner_part_id: "Inner" dataset part: we intend to drop clones from this part.
            outer_part_ids: Sequence of "outer" dataset parts: we do not intend to drop clones from these parts.
            use_exact_hash: True to use logic for ExactHashProcessor output format, False - for SourcererCC.
        """
        # get outer clones by messages and by diffs
        if use_exact_hash:
            msg_clones = self._get_outer_clones_exact_hash(
                msg_clones_fname, inner_part_id=inner_part_id, outer_part_ids=outer_part_ids
            )
            diff_clones = self._get_outer_clones_exact_hash(
                diff_clones_fname, inner_part_id=inner_part_id, outer_part_ids=outer_part_ids
            )
        else:
            msg_clones = self._get_outer_clones(
                msg_clones_fname, inner_part_id=inner_part_id, outer_part_ids=outer_part_ids
            )
            diff_clones = self._get_outer_clones(
                diff_clones_fname, inner_part_id=inner_part_id, outer_part_ids=outer_part_ids
            )

        # drop all message clones from inner part
        for group in msg_clones:
            ids_to_drop = group.get_ids_to_drop(include_root=False)
            for idx in ids_to_drop:
                commit = self._ids_to_commits_map[idx]
                self._outer_clones_to_drop[commit["repo"]].add(commit["hash"])

        # drop all diffs clones from inner part
        for group in diff_clones:
            ids_to_drop = group.get_ids_to_drop(include_root=False)
            for idx in ids_to_drop:
                commit = self._ids_to_commits_map[idx]
                self._outer_clones_to_drop[commit["repo"]].add(commit["hash"])

    def _get_inner_ids_to_drop(
        self,
        msg_clones_fname: str,
        diff_clones_fname: str,
        inner_part_id: int,
        only_full_inner_clones: bool,
        identical_clones: bool,
        use_exact_hash: bool,
    ) -> None:
        """Aggregates ids of duplicated examples inside specific dataset part (e.g. train).

        Args:
            msg_clones_fname: Path to file with clones in terms of messages.
            diff_clones_fname: Path to file with clones in terms of diffs.
            inner_part_id: Current part id, should be the same as used in `PreDeduplicationProcessor` for this specific part.
            only_full_inner_clones: True to drop only full clones, False to also drop partial clones
                (clones only in terms of diffs or messages).
            identical_clones: True to use logic for 100% clones and False to use logic for similar clones.
            use_exact_hash: True to use logic for ExactHashProcessor output format, False - for SourcererCC.
        """
        # obtain inner clones
        if identical_clones:
            if use_exact_hash:
                msg_clones = self._get_inner_clones_identical_exact_hash(msg_clones_fname, part_id=inner_part_id)
                diff_clones = self._get_inner_clones_identical_exact_hash(diff_clones_fname, part_id=inner_part_id)
            else:
                msg_clones = self._get_inner_clones_identical(msg_clones_fname, part_id=inner_part_id)
                diff_clones = self._get_inner_clones_identical(diff_clones_fname, part_id=inner_part_id)
        else:
            msg_clones = self._get_inner_clones_similar(msg_clones_fname, part_id=inner_part_id)
            diff_clones = self._get_inner_clones_similar(diff_clones_fname, part_id=inner_part_id)

        if only_full_inner_clones:
            if identical_clones:
                full_clones = self._get_full_inner_clones_identical(msg_clones=msg_clones, diff_clones=diff_clones)
            else:
                full_clones = self._get_full_inner_clones_similar(msg_clones=msg_clones, diff_clones=diff_clones)
            for group in full_clones:
                ids_to_drop = group.get_ids_to_drop()
                for idx in ids_to_drop:
                    commit = self._ids_to_commits_map[idx]
                    self._inner_clones_to_drop[commit["repo"]].add(commit["hash"])
        else:
            for group in msg_clones:
                ids_to_drop = group.get_ids_to_drop()
                for idx in ids_to_drop:
                    commit = self._ids_to_commits_map[idx]
                    self._inner_clones_to_drop[commit["repo"]].add(commit["hash"])

            for group in diff_clones:
                ids_to_drop = group.get_ids_to_drop()
                for idx in ids_to_drop:
                    commit = self._ids_to_commits_map[idx]
                    self._inner_clones_to_drop[commit["repo"]].add(commit["hash"])

    def clones_report(self):
        self.logger.info("===== Clones Report =====")
        self.logger.info(
            f"Will drop {sum(len(self._outer_clones_to_drop[key]) for key in self._outer_clones_to_drop)} outer clones\n"
        )
        self.logger.info(
            f"Will drop {sum(len(self._inner_clones_to_drop[key]) for key in self._inner_clones_to_drop)} inner clones\n"
        )

    def prepare(  # type: ignore[override]
        self,
        inner_part_id: int,
        outer_part_ids: Sequence[int],
        msg_clones_fname: str,
        diff_clones_fname: str,
        process_inner_clones: bool,
        process_outer_clones: bool,
        only_full_inner_clones: bool,
        identical_clones: bool,
        use_exact_hash: bool,
        is_ready: bool = False,
        **kwargs,
    ) -> None:
        """Prepares a set of clones ids to drop.

        Note:
          For `inner_part_id` and `outer_part_ids` ids should be the same as
          used in `PreDeduplicationProcessor` for each part.

        Args:
            in_fname: Path to specific input file.
            inner_part_id: "Inner" part id.
            outer_part_ids: A sequence of parts that will be considered as "outer" when searching for outer clones.
            msg_clones_fname: Path to file with clones in terms of messages.
            diff_clones_fname: Path to file with clones in terms of diffs.
            process_inner_clones: True to process inner clones (clones inside given dataset part), False otherwise.
            process_outer_clones: True to process outer clones (clones between given dataset parts), False otherwise.
            only_full_inner_clones: True to drop only full clones, False to also drop partial clones
                (clones only in terms of diffs or messages).
            identical_clones: True to use logic for 100% clones and False to use logic for similar clones.
            use_exact_hash: True to use logic for ExactHashProcessor output format, False - for SourcererCC.
            is_ready: A flag to indicate that clones ids are already built and are stored in `self._ids_to_drop`.
                When it is set to True, this method doesn't do anything.
        """
        if is_ready:
            self.clones_report()
            return

        if process_inner_clones:
            self._get_inner_ids_to_drop(
                msg_clones_fname=msg_clones_fname,
                diff_clones_fname=diff_clones_fname,
                inner_part_id=inner_part_id,
                only_full_inner_clones=only_full_inner_clones,
                identical_clones=identical_clones,
                use_exact_hash=use_exact_hash,
            )
        if process_outer_clones:
            self._get_outer_ids_to_drop(
                msg_clones_fname=msg_clones_fname,
                diff_clones_fname=diff_clones_fname,
                inner_part_id=inner_part_id,
                outer_part_ids=outer_part_ids,
                use_exact_hash=use_exact_hash,
            )

        self.clones_report()

    def _process_chunk(self, chunk: pd.DataFrame, repo: str, **kwargs) -> pd.DataFrame:
        return chunk.loc[
            [
                cur_hash not in self._inner_clones_to_drop[repo] and cur_hash not in self._outer_clones_to_drop[repo]
                for cur_hash in chunk.hash
            ]
        ]
