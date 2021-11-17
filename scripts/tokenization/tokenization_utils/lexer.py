import re
import logging
import pandas as pd
import pygments.util
from typing import List
from pygments import lex
from pygments.lexers import guess_lexer_for_filename, TextLexer


class Lexer:
    def __init__(self, sep_token: str):
        self._fname_pattern = "^<FNAME>.*?\n"
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

    def _lex(self, id: int, fname: str, string: str) -> List[str]:
        """
        Find appropriate lexer based on diff and filename and return resulting lexemes
        in case if pygments decides to use TextLexer (which doesn't do anything), just split tokens by whitespaces
        """
        try:
            lexer = guess_lexer_for_filename(fname, string)
            if not isinstance(lexer, TextLexer):
                return [token[1] for token in lex(string, lexer)]
            else:
                logging.warning(f"TextLexer chosen for `{fname}` (id: {id})")
                return string.split()
        except pygments.util.ClassNotFound:
            logging.warning(f"no lexer found for `{fname}` (id: {id})")
            return string.split()

    def lex(self, id: int, diff: str) -> str:
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
            if (
                row.startswith("new file")
                or row.startswith("deleted file")
                or row.startswith("rename to")
                or row.startswith("copy to")
            ):
                fname = row.split(" ")[-1]
            elif row.startswith("<FNAME>"):
                fname = row.strip("<FNAME>")
            else:
                tokens.extend(row.split())

            if fname:
                tokens.append(fname)
                if i + 1 < len(rows) and not (
                    rows[i + 1].startswith("new file")
                    or rows[i + 1].startswith("deleted file")
                    or rows[i + 1].startswith("rename to")
                    or rows[i + 1].startswith("copy to")
                    or rows[i + 1].startswith("<FNAME>")
                ):
                    tokens.extend(self._lex(id, fname.strip(), rows[i + 1]))
                    i += 1

            i += 1
        return self._sep.join(tokens)

    def __call__(self, input_filename: str, output_filename: str, save_diffs: bool = False, diff_filename: str = None):
        df = pd.read_csv(input_filename)
        diffs = []
        for id, diff in df["diff"].items():
            diffs.append(self.lex(id, diff))
        df["diff"] = diffs

        df.to_csv(output_filename, index=None)
        if save_diffs:
            assert diff_filename
            df["diff"].to_csv(diff_filename, index=None)
