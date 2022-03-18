import os
import json
import numpy as np
import pandas as pd
from typing import List, Iterable, Optional, Dict, Tuple
from tqdm import tqdm
from joblib import Parallel, delayed
from pygments import lex
from pygments.token import _TokenType, Literal, Text
from pygments.util import ClassNotFound
from pygments.lexers import guess_lexer_for_filename, TextLexer
from ..base_utils import BaseProcessor


class Lexer(BaseProcessor):
    """
    This class finds appropriate lexer for each file diff.
    It is used as pre-tokenizer for our custom BPE tokenizer.
    """

    def __init__(
        self,
        upper_percentile: float,
        sep_token: str,
        chunksize: int,
        data_format: str,
        n_workers: Optional[int] = None,
        logger_name: Optional[str] = None,
    ):
        super().__init__(chunksize=chunksize, logger_name=logger_name, n_workers=n_workers, data_format=data_format)
        self._sep = sep_token
        self._upper_percentile = upper_percentile
        self._percentiles: Dict[float, float] = {}

        # TODO: these examples make pygments hang ;( currently they are manually skipped
        # (note: they all contain some gsql, might be related to https://github.com/pygments/pygments/pull/2006)
        self._examples_to_skip = [1731725, 1731749, 1731755, 1731759, 1732004]

    def _lex_diff(self, id: int, fname: str, diff: str) -> Iterable[Tuple[_TokenType, str]]:
        """
        This method finds appropriate lexer based on diff and filename and returns resulting lexemes.

        In case pygments doesn't have appropriate lexer or decides to use TextLexer (which doesn't do anything),
        tokens are simply split by spaces.
        """
        try:
            lexer = guess_lexer_for_filename(fname, diff)
            if not isinstance(lexer, TextLexer):
                yield from lex(diff, lexer)
            else:
                self.logger.warning(f"TextLexer chosen for `{fname}` (id: {id})")
                yield from ((Text, token) for token in diff.split())
        except ClassNotFound:
            self.logger.warning(f"No lexer found for `{fname}` (id: {id})")
            yield from ((Text, token) for token in diff.split())

    def _lex_commit_mods(self, cur_id: int, cur_mods: List[Dict[str, str]]) -> str:
        """
        This method iterates over all modifications in current commit and tokenizes each of them.
        """
        tokens: List[str] = []

        for mod in cur_mods:
            if mod["change_type"] == "UNKNOWN":
                continue
            if mod["change_type"] == "ADD":
                file_diff = f"new file {mod['new_path']}\n"
                fname = mod["new_path"]
            elif mod["change_type"] == "DELETE":
                file_diff = f"deleted file {mod['old_path']}\n"
                fname = mod["old_path"]
            elif mod["change_type"] == "RENAME":
                file_diff = f"rename from {mod['old_path']}\nrename to {mod['new_path']}\n"
                fname = mod["new_path"]
            elif mod["change_type"] == "COPY":
                file_diff = f"copy from {mod['old_path']}\ncopy to {mod['new_path']}\n"
                fname = mod["new_path"]
            else:
                file_diff = f"{mod['new_path']}\n"
                fname = mod["new_path"]

            mod_tokenized = self._lex_diff(cur_id, fname, mod["diff"])
            tokens.extend((token.strip() for token in file_diff.split()))
            tokens.extend(
                (
                    lexeme[1].strip()
                    for lexeme in mod_tokenized
                    if lexeme[0] not in Literal
                    or (lexeme[0] in Literal and len(lexeme[1]) > self._percentiles[self._upper_percentile])
                )
            )

        return self._sep.join(tokens)

    def _get_literals_len_mods(self, cur_id: int, cur_mods: List[Dict[str, str]]) -> List[int]:
        """
        This method iterates over all modifications in current commit, tokenizes each of them and returns # symbols
        in each literal.
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
                [len(lexeme[1]) for lexeme in mod_tokenized if lexeme[0] in Literal and len(lexeme[1]) > 1]
            )

        return literals_len

    def _get_literals_len(self, in_fname: str, literals_len_dir: str):
        """
        This method tokenizes diffs with appropriate lexers and saves lengths of tokens marked as literals.

        Args:
            - in_fname: path to read input data from
            - literals_len_dir: path to directory to save literals lengths to
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

    def _get_percentiles(self, literals_len_dir: str):
        """
        This method calculates 1%, 5%, 90%, 95%, 99% percentiles of literals lengths from diffs.

        Args:
            - literals_len_dir: path to directory to read precomputed literals lengths from
        """
        with open(os.path.join(literals_len_dir, "literals_len.txt"), "r") as file:
            literals_lens = [int(line.strip()) for line in file]

        for q in [0.01, 0.05, 0.9, 0.95, 0.99]:
            self._percentiles[q] = np.quantile(literals_lens, q)

        with open(os.path.join(literals_len_dir, "literals.json"), "w") as file:
            json.dump(self._percentiles, file)

    def prepare(
        self,
        in_fname: str,
        literals_len_dir: str,
        percentile_dir: Optional[str] = None,
    ):
        """
        This method tokenizes diffs and messages and calculates percentiles for literals lengths.

         Args:
             - in_fname: path to read input data from
             - literals_len_dir: path to save supplementary information like # of tokens for each example and percentiles
             - percentile_dir: (optional) path to directory with already computed percentiles; might be useful for
                dropping outliers from val/test by using percentiles from train, which has much more examples
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
            res = pool(
                delayed(self._lex_commit_mods)(cur_id=item["id"], cur_mods=item["mods"])
                for _, item in chunk[["id", "mods"]].iterrows()
            )

        chunk["mods"] = res
        return chunk

    def __call__(self, in_fname: str, out_fname: str, diffs_out_fname: str, **kwargs):
        """
        This method iterates over input data in chunks, processes it in some way and saves results to separate file.

        Args:
            - in_fname: path to read input data from
            - out_fname: path to save processed data to
            - diffs_out_fname: path to save diffs
        """
        prepare_kwargs = {key[len("prepare_") :]: value for key, value in kwargs.items() if key.startswith("prepare_")}
        process_kwargs = {key: value for key, value in kwargs.items() if not key.startswith("prepare_")}

        self.logger.info(f"Starting processing {in_fname}")

        self._prepare_outfile(out_fname)
        self._prepare_outfile(diffs_out_fname)
        self.prepare(in_fname, **prepare_kwargs)

        reader = self._read_input(in_fname)
        for chunk in tqdm(reader, leave=False):
            processed_chunk = self.process(chunk.loc[~chunk["id"].isin(self._examples_to_skip)], **process_kwargs)
            self._append_to_outfile(processed_chunk["mods"].tolist(), diffs_out_fname)
            self._append_to_outfile(processed_chunk, out_fname)

        self.logger.info(f"Finished processing {in_fname}")
