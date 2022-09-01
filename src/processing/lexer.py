import copy
import json
import os
import re
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from pygments import lex
from pygments.lexers import TextLexer, guess_lexer_for_filename
from pygments.token import (
    Comment,
    Error,
    Keyword,
    Literal,
    Operator,
    Punctuation,
    Whitespace,
    _TokenType,
)
from pygments.util import ClassNotFound
from tqdm import tqdm

from ..utils import BaseProcessor


class Lexer(BaseProcessor):
    """This class is used to lex diffs when it is possible.

    It calculates percentiles on tokens' lengths and drops very long tokens.

    It also saves lexed data in a format required for pre-tokenization stage:
    lexemes are separated by additional space characters.

    Args:
        upper_percentile: Percentile to use as an upper bound (should be in (0, 1) range).
        data_format: In which format mined data is saved.
        chunksize: Number of examples to process at once (data is read in chunks). Optional, default value is 1000.
        n_workers: Maximum number of concurrently running jobs. Optional, default value is 1 (sequential execution).
        logger_name: Name of logger for this class. Optional, default value is None.
    """

    def __init__(
        self,
        upper_percentile: float,
        line_sep: str,
        data_format: str,
        chunksize: Optional[int] = None,
        n_workers: Optional[int] = None,
        logger_name: Optional[str] = None,
    ):
        super().__init__(chunksize=chunksize, n_workers=n_workers, data_format=data_format, logger_name=logger_name)

        self._upper_percentile = upper_percentile
        self._line_sep = line_sep
        self._percentiles: Dict[float, float] = {}

        # TODO: these examples make pygments hang ;( currently they are manually skipped
        # (note: they all contain some gsql, might be related to https://github.com/pygments/pygments/pull/2006)
        self._examples_to_skip = [1731725, 1731749, 1731755, 1731759, 1732004]

    def _is_lexeme_allowed(self, lexeme: Tuple[_TokenType, str]) -> bool:
        """Checks if current lexeme is allowed.
        Lexeme is allowed if:
        * it is a docstring literal
        * it is a comment
        * it is shorter than `self._upper_percentile`
        """
        if lexeme[0] == Literal.String.Doc or lexeme[0] in Comment:
            return True

        return len(lexeme[1]) <= self._percentiles[self._upper_percentile]

    def _lex_diff(self, id: int, fname: str, diff: str) -> Iterable[Tuple[_TokenType, str]]:
        """Finds appropriate lexer based on diff and filename and returns resulting lexemes.

        In case `pygments` doesn't have appropriate lexer or decides to use `TextLexer` (which doesn't do anything),
        tokens are simply split by whitespaces.
        """
        try:
            lexer = guess_lexer_for_filename(fname, diff)
            if not isinstance(lexer, TextLexer):
                yield from lex(diff, lexer)
            else:
                yield from ((Error.DefaultLexer, token) for token in diff.split())
        except ClassNotFound:
            self.logger.warning(f"No lexer found for `{fname}` (id: {id})")
            yield from ((Error.DefaultLexer, token) for token in diff.split())

    def _lex_commit_mods(self, cur_id: int, cur_mods: List[Dict[str, str]]) -> List[Dict[str, Union[str, List[str]]]]:
        """Iterates over all modifications in current commit and tokenizes each of them."""

        for mod in cur_mods:
            if mod["change_type"] == "UNKNOWN":
                continue
            if mod["change_type"] == "DELETE":
                fname = mod["old_path"]
            else:
                fname = mod["new_path"]

            # drop literals with lengths more than upper percentile
            diff = [
                lexeme[1] if self._is_lexeme_allowed(lexeme) else "[LONG]"
                for lexeme in self._lex_diff(cur_id, fname, mod["diff"])
            ]

            if len(diff) == 1 and diff[0].strip() == "[LONG]":
                self.logger.error(f"Whole diff as a single token! id: {cur_id}, fname: {fname}")
                diff = [lexeme[1] for lexeme in self._lex_diff(cur_id, fname, mod["diff"])]

            mod["diff"] = diff

        return cur_mods

    def _get_literals_len_mods(self, cur_id: int, cur_mods: List[Dict[str, str]]) -> List[int]:
        """Iterates over all modifications in current commit,
        tokenizes each of them and returns length of each literal.
        """
        literals_len = []

        for mod in cur_mods:
            if mod["change_type"] == "UNKNOWN":
                continue

            if mod["change_type"] == "DELETE":
                fname = mod["old_path"]
            else:
                fname = mod["new_path"]

            mod_tokenized = self._lex_diff(cur_id, fname, mod["diff"])

            literals_len.extend(
                [
                    len(lexeme[1])
                    for lexeme in mod_tokenized
                    if lexeme[0] != Error.DefaultLexer
                    and lexeme[0] not in Punctuation
                    and lexeme[0] not in Operator
                    and lexeme[0] not in Whitespace
                    and lexeme[0] not in Keyword
                    and len(lexeme[1]) > 1
                ]
            )

        return literals_len

    def _get_literals_len(self, in_fname: str, literals_len_dir: str) -> None:
        """Tokenizes diffs with appropriate lexers and saves lengths of tokens marked as literals.

        Args:
            in_fname: Path to read input data from.
            literals_len_dir: Path to directory to save literals lengths to.
        """
        self.logger.info(f"Starting processing literals in {in_fname}")

        open(os.path.join(literals_len_dir, "literals_len.txt"), "w", encoding="utf-8").close()

        reader = self._read_input(in_fname)
        for chunk in tqdm(reader, desc=f"Tokenizing {in_fname}", leave=False):
            chunk = chunk.loc[~chunk["id"].isin(self._examples_to_skip)]

            with Parallel(self._n_workers) as pool:
                res = pool(
                    delayed(self._get_literals_len_mods)(item["id"], item["mods"])
                    for _, item in chunk[["id", "mods"]].iterrows()
                )
            with open(os.path.join(literals_len_dir, "literals_len.txt"), "a", encoding="utf-8") as file:
                for lines in res:
                    file.writelines([f"{line}\n" for line in lines])

        self.logger.info(f"Finished processing literals in {in_fname}")

    def _get_percentiles(self, literals_len_dir: str) -> None:
        """Calculates percentiles of literals lengths from diffs.

        Args:
            literals_len_dir: Path to directory to read precomputed literals lengths from.
        """
        with open(os.path.join(literals_len_dir, "literals_len.txt"), "r") as file:
            literals_lens = [int(line.strip()) for line in file]

        for q in [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]:
            self._percentiles[q] = np.quantile(literals_lens, q)

        with open(os.path.join(literals_len_dir, "literals.json"), "w") as file:
            json.dump(self._percentiles, file)

    def prepare(self, in_fname: str, literals_len_dir: str, percentile_dir: Optional[str] = None, **kwargs) -> None:
        """Runs lexers on diffs and removes literals with lengths more than percentiles.

        Args:
            in_fname: Path to read input data from.
            literals_len_dir: Path to save supplementary information like # of tokens for each example and percentiles.
            percentile_dir: Path to directory with already computed percentiles. Optional. Use-case: dropping outliers
               from val/test by percentiles calculated on train.
        """
        if percentile_dir:
            # read precomputed percentiles
            with open(os.path.join(percentile_dir, "literals.json"), "r") as file:
                self._percentiles = json.load(file, object_hook=lambda d: {float(k): v for k, v in d.items()})
        else:
            # compute percentiles
            self._get_literals_len(in_fname=in_fname, literals_len_dir=literals_len_dir)
            self._get_percentiles(literals_len_dir=literals_len_dir)

    def process(self, chunk: pd.DataFrame, **kwargs) -> pd.DataFrame:
        with Parallel(self._n_workers) as pool:
            mods = pool(
                delayed(self._lex_commit_mods)(cur_id=item["id"], cur_mods=item["mods"])
                for _, item in chunk[["id", "mods"]].iterrows()
            )

        mods_sep = copy.deepcopy(mods)
        for i, example in enumerate(mods_sep):
            for j, mod_sep in enumerate(example):
                diff = self._line_sep.join(
                    [re.sub(" +", " ", line.strip()) for line in " ".join(mod_sep["diff"]).split("\n")]
                )
                mods_sep[i][j]["diff"] = diff
        chunk["mods_sep"] = mods_sep

        for i, example in enumerate(mods):
            for j, mod in enumerate(example):
                mods[i][j]["diff"] = "".join(mod["diff"]).replace("\n", self._line_sep)
        chunk["mods"] = mods

        return chunk

    def __call__(self, in_fname: str, out_fname: str, delimiter_out_fname: str, **kwargs) -> None:
        """Iterates over input data in chunks, lexes it and saves results to separate file.

        In this particular case, there are two versions of output files:

        * data without long literals
            (this version can be tokenized with other tokenizers)
        * data without long literals + all tokens are separated with additional spaces
            (this version can be tokenized with our custom tokenizer)

        Args:
            in_fname: Path to read input data from.
            out_fname: Path to save processed data to.
            delimiter_out_fname: Path to save processed data with special delimiter to.
            **kwargs: Arbitrary keyword arguments. Keyword arguments starting from prefix 'prepare_'
                will be passed to method that is called before data processing,
                all others - to method that processes each chunk.
        """
        prepare_kwargs = {key[len("prepare_") :]: value for key, value in kwargs.items() if key.startswith("prepare_")}
        process_kwargs = {key: value for key, value in kwargs.items() if not key.startswith("prepare_")}

        self.logger.info(f"Starting processing {in_fname}")

        self._prepare_outfile(out_fname)
        self._prepare_outfile(delimiter_out_fname)
        self.prepare(in_fname, **prepare_kwargs)

        reader = self._read_input(in_fname)
        for chunk in tqdm(reader, leave=False):
            processed_chunk = self.process(chunk.loc[~chunk["id"].isin(self._examples_to_skip)], **process_kwargs)
            self._append_to_outfile(processed_chunk.drop(columns=["mods_sep"]), out_fname)
            self._append_to_outfile(
                processed_chunk.drop(columns=["mods"]).rename(columns={"mods_sep": "mods"}), delimiter_out_fname
            )
        self.logger.info(f"Finished processing {in_fname}")
