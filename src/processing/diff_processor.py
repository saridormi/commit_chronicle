import re
from typing import Dict, List

import pandas as pd
from joblib import Parallel, delayed

from ..utils import BaseProcessor


class DiffProcessor(BaseProcessor):
    """This class is used to delete undesirable patterns from diffs."""

    @staticmethod
    def _filter_diff(diff: str) -> str:
        """Filters single diff string.

        Currently, it includes the following:
            * removing some unnecessary git stuff (e.g. @@ ... @@)
            * removing non-changed lines
            * removing extra `\t` and `\r` symbols
        """
        diff_lines = diff.split("\n")
        processed_lines = []

        for line in diff_lines:
            line = line.strip()

            if line.startswith("-") and len(line) > 1:
                # lines that were removed
                # example: - version='2.0.2'
                processed_lines.append(line)

            elif line.startswith("+") and len(line) > 1:
                # lines that were added
                # example: + version='2.0.2'
                processed_lines.append(line)

            elif line.startswith("Binary files") and line.endswith("differ"):
                # example: Binary files <filename1> and <filename2> differ
                processed_lines.append(line)

        processed_diff = "\n".join(processed_lines + [""])
        # squeeze several whitespace sequences into one (do now consider \n)
        processed_diff = re.sub("[^\S\n]+", " ", processed_diff)
        return processed_diff

    def _filter_mods(self, mods: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Filters all modifications from single commit.
        """
        filtered_mods = []
        for mod in mods:
            if isinstance(mod["diff"], str) and mod["diff"].isascii():
                mod["diff"] = DiffProcessor._filter_diff(mod["diff"])
                filtered_mods.append(mod)
        return filtered_mods

    def process(self, chunk: pd.DataFrame, **kwargs) -> pd.DataFrame:
        with Parallel(self._n_workers) as pool:
            filtered_mods = pool(delayed(self._filter_mods)(mods) for _, mods in chunk["mods"].items())

        chunk["mods"] = filtered_mods
        chunk["diff_len"] = [sum(len(mod["diff"]) for mod in mods) for mods in chunk["mods"]]
        chunk = chunk.loc[chunk.diff_len > 0].drop(columns="diff_len")
        return chunk
