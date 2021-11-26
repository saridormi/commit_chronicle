import csv
import re
import logging
import pandas as pd
from typing import List
import pygments
from pygments import lex
from pygments.lexers import guess_lexer_for_filename, TextLexer
from joblib import delayed, Parallel


class Lexer:
    def __init__(self, sep_token: str):
        self._fname_pattern = "^(?:\/?[\w]+\/)+?(?:[\w\.]*)?\n|^(?:\.[\w]+?)\n|^(?:[\w]+?\.[\w]+?)\n"
        self._rename_pattern = "^rename from .*?\n|rename to .*?\n"
        self._copy_pattern = "^copy from .*?\n|copy to .*?\n"
        self._new_pattern = "^new file .*?\n"
        self._delete_pattern = "^deleted file .*?\n"
        self._pattern = (
            "("
            + "|".join(
                [self._rename_pattern, self._copy_pattern, self._new_pattern, self._delete_pattern, self._fname_pattern]
            )
            + ")"
        )
        self._sep = sep_token

    def _lex(self, id: int, fname: str, diff: str) -> List[str]:
        """
        Find appropriate lexer based on diff and filename and return resulting lexemes
        in case if pygments doesn't have appropriate lexer or decides to use TextLexer (which doesn't do anything),
        just split tokens by whitespaces
        """

        try:
            lexer = guess_lexer_for_filename(fname, diff)
            if not isinstance(lexer, TextLexer):
                lexemes = lex(diff, lexer)
                return [lexeme[1] for lexeme in lexemes]
            else:
                logging.warning(f"TextLexer chosen for `{fname}` (id: {id})")
                return diff.split()
        except pygments.util.ClassNotFound:
            logging.warning(f"no lexer found for `{fname}` (id: {id})")
            return diff.split()

    def _split(self, id: int, diff: str) -> str:
        """
        Single diff might contain changes of several files on different languages,
        we have to process changes for each file separately.

        Split diff by changed files via regular expressions and tokenize each sub-diff
        with appropriate lexer
        """
        rows = [line for line in re.split(self._pattern, diff, flags=re.MULTILINE) if line != ""]
        i = 0
        tokens = []
        while i < len(rows):
            row = rows[i]
            fname = None
            if len(row.split("\n")[:-1]) == 1 and (
                row.startswith("new file")
                or row.startswith("deleted file")
                or row.startswith("rename to")
                or row.startswith("copy to")
            ):
                fname = row.split(" ")[-1]
            elif len(row.split("\n")[:-1]) == 1 and re.match(self._fname_pattern, row):
                fname = row

            if fname:
                tokens.append(fname)
                if i + 1 < len(rows):
                    tokens.extend(self._lex(id, fname.strip(), rows[i + 1]))
                    i += 1
            else:
                tokens.extend(row.split())

            i += 1
        return self._sep.join(tokens)

    def __call__(
        self,
        input_filename: str,
        output_filename: str,
        chunksize: int,
        save_diffs: bool = False,
        diff_filename: str = None,
    ):
        fieldnames = ["id", "author", "date", "hash", "message", "diff", "repo"]
        with open(output_filename, "w") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
        if save_diffs:
            open(diff_filename, "w").close()

        reader = pd.read_csv(input_filename, chunksize=chunksize)
        for chunk in reader:
            """
            with Parallel(4) as pool:
                diffs = pool(
                    delayed(self._split)(row["id"], row["diff"])
                    for _, row in chunk[["id", "diff"]].iterrows()
                )

            """
            diffs = []
            for _, row in chunk[["id", "diff"]].iterrows():
                diffs.append(self._split(row["id"], row["diff"]))

            chunk["diff"] = diffs

            chunk.to_csv(output_filename, mode="a", index=None, header=None)
            if save_diffs:
                chunk["diff"].to_csv(diff_filename, mode="a", index=None, header=None)
