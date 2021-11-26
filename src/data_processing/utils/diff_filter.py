import csv
import re
import pandas as pd
from tqdm import tqdm


class DiffFilter:
    """
    Class to remove unchanged lines from diffs.
    """

    def __call__(self, diff: str) -> str:
        """
        This method preprocessed single diff string.
        Currently _preprocessing for diffs includes the following:
            - removing some unnecessary special info
            - removing non-changed lines
        """
        diff_lines = diff.split("\n")
        processed_lines = []

        for line in diff_lines:
            if line.startswith("@@"):
                line = line.split("@@")[-1]
            line = line.strip()

            if line.startswith("new file"):
                # line in git diff when file was created
                # example: new file <filename>
                processed_lines.append(line)

            elif line.startswith("deleted file"):
                # line in git diff when file was created or deleted
                # example: deleted file <filename>
                processed_lines.append(line)

            elif line.startswith("rename"):
                # lines in git diff when file was renamed
                # example: rename from <old_filename>, rename to <new_filename>
                processed_lines.append(line)

            elif line.startswith("copy"):
                # lines in git diff when file was copied
                # example: copy from <old_filename>, copy to <new_filename>
                processed_lines.append(line)

            elif line.startswith("-") and len(line.replace(" ", "").strip("-")) > 1:
                # lines that were removed
                # example: - version='2.0.2', -version='2.0.2'
                processed_lines.append(line)

            elif line.startswith("+") and len(line.replace(" ", "").strip("+")) > 1:
                # lines that were added
                # example: + version='2.0.2', +version='2.0.2
                processed_lines.append(line)

            elif line.startswith("Binary files") and line.endswith("differ"):
                # example: Binary files <filename1> and <filename2> differ
                processed_lines.append(line)

            elif len(line) > 1 and len(line.split()) == 1 and re.match("^[\w\.\/]+$", line):
                # filename header
                processed_lines.append(line)

        processed_diff = "\n".join(processed_lines)
        processed_diff = re.sub(r" +", " ", processed_diff)
        return processed_diff

    @staticmethod
    def filter(input_filename: str, output_filename: str, chunksize: int):
        fieldnames = ["id", "author", "date", "hash", "message", "diff", "repo"]
        with open(output_filename, "w") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

        filter = DiffFilter()
        reader = pd.read_csv(input_filename, chunksize=chunksize)
        for chunk in tqdm(reader, desc=f"Filtering diffs from {input_filename}", leave=False):
            filtered_diffs = []
            for _, diff in chunk["diff"].items():
                if isinstance(diff, str) and diff.isascii():
                    filtered_diffs.append(filter(diff))
                else:
                    filtered_diffs.append("")
            chunk["diff"] = filtered_diffs
            chunk = chunk.loc[chunk["diff"].str.len() > 0]
            chunk[["id", "author", "date", "hash", "message", "diff", "repo"]].to_csv(
                output_filename, mode="a", index=False, header=False
            )
