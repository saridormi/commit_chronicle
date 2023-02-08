import re
from typing import Dict, List

import pandas as pd

from ..utils import BaseProcessor


class DiffProcessor(BaseProcessor):
    """This class is used to delete undesirable patterns from diffs."""

    @staticmethod
    def _process_diff(diff: str, line_sep: str) -> str:
        """Processes a single diff (for a single file modification).

        Currently, it includes the following:
            * removing @@ ... @@ line – unnecessary git stuff
            * squeeze several whitespace sequences into one

        Args:
            diff: Input diff.
            line_sep: Newline separator that should be used in processed diff.

        Returns:
            Processed diff.
        """
        diff_lines = diff.split("\n")
        processed_lines = []

        for line in diff_lines:
            line = line.strip()
            if not line:
                continue

            if line.startswith("@@") and line.endswith("@@"):
                continue

            processed_lines.append(line)

        processed_diff = line_sep.join(processed_lines + [""])
        # squeeze several whitespace sequences into one (do not consider \n)
        processed_diff = re.sub(r"[^\S\n]+", " ", processed_diff)
        return processed_diff

    def _process_mods(self, mods: List[Dict[str, str]], line_sep: str) -> List[Dict[str, str]]:
        """
        Processes diffs in all modifications from single commit.

        Args:
            mods: A list of modifications from current commit.
            line_sep: Newline separator that should be used in processed diff.

        Returns:
            A list of modifications with processed diffs.
        """
        filtered_mods = []
        for mod in mods:
            if isinstance(mod["diff"], str) and mod["diff"].isascii():
                mod["diff"] = DiffProcessor._process_diff(mod["diff"], line_sep=line_sep)
                filtered_mods.append(mod)
        return filtered_mods

    def _process_chunk(self, chunk: pd.DataFrame, line_sep: str = "\n", **kwargs) -> pd.DataFrame:  # type: ignore[override]
        filtered_mods = [self._process_mods(cur_mods, line_sep) for cur_mods in chunk.mods]
        chunk["mods"] = filtered_mods
        chunk["diff_len"] = [sum(len(mod["diff"]) for mod in mods) for mods in chunk["mods"]]
        return chunk.loc[chunk.diff_len > 0].drop(columns="diff_len")
