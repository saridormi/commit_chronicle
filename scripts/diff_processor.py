import argparse
import re
import pandas as pd
from typing import List, Tuple, Dict
from joblib import Parallel, delayed


class DiffProcessor:
    """
    Class to remove unchanged lines from diffs.
    """

    @staticmethod
    def _process_single_diff(idx: int, diff: str) -> Tuple[int, str]:
        """
        This method preprocessed single diff string.
        Currently _preprocessing for diffs includes the following:
            - removing some unnecessary special info
            - removing non-changed lines
        """
        diff_lines = diff.split("\n")
        processed_lines = []

        for line in diff_lines:
            if len(line) == 0:
                # remove empty lines
                continue

            elif line.startswith("new") or line.startswith("deleted"):
                # line in git diff when file was created or deleted
                # example: new file mode <mode> <filename> / deleted file mode <mode> <filename>
                processed_lines.append(line)

            elif line.startswith("rename") or line.startswith("copy"):
                # lines in git diff when file was renamed or copied
                # example 1: rename from <old_filename>, rename to <new_filename>
                # example 2: copy from <old_filename>, copy to <new_filename>
                processed_lines.append(line)

            elif (line.startswith("-") or line.startswith("+")) and len(line.split()) > 1:
                # lines that were removed/added
                # example: - version='2.0.2', -version='2.0.2'
                # example: + version='2.0.2', +version='2.0.2
                processed_lines.append(line)

            elif (
                line.startswith("index")
                or line.startswith("similarity index")
                or (line.startswith("@@") and line.endswith("@@"))
            ):
                # some special info that we are not interested in
                # example 1: index 0000000..3f26e45
                # example 2: similarity index 100%
                # example 3: @@ -0,0 +1,192 @@
                continue

            elif line.startswith("Binary files") and line.endswith("differ"):
                # example: Binary files <file1> and <file2> differ
                processed_lines.append(line)

            elif len(line.split()) == 1:
                # filename header in case of file modification and maybe other rare cases that won't hurt too much
                # example: <filename>
                processed_lines.append(line)

        processed_diff = "\n".join(processed_lines)
        processed_diff = re.sub(r" +", " ", processed_diff)
        return idx, processed_diff

    @staticmethod
    def _process_diffs(ids: List[int], diffs: List[str]) -> Dict[int, str]:
        with Parallel(16) as pool:
            diff_res = pool(delayed(DiffProcessor._process_single_diff)(idx, diff) for idx, diff in zip(ids, diffs))
        return {idx: diff for idx, diff in diff_res}

    @staticmethod
    def process(input_filename: str, output_filename: str):
        df = pd.read_csv(input_filename)
        df["diff"] = pd.Series(DiffProcessor._process_diffs(df["id"].tolist(), df["diff"].tolist()))
        df = df.dropna()
        df[["id", "author", "date", "hash", "message", "diff", "repo"]].to_csv(
            output_filename, index=False, header=True
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="This script removes unchanged lines from diffs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input_filename", type=str, default="../filtered_commits.csv", help="path to .csv file with data"
    )
    parser.add_argument(
        "--output_filename",
        type=str,
        default="../commits_drop_unchanged.csv",
        help="path to save .csv file with filtered commits",
    )
    args = parser.parse_args()

    DiffProcessor.process(input_filename=args.input_filename, output_filename=args.output_filename)
