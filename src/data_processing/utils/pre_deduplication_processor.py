import re
import hashlib
import logging
import pandas as pd
from typing import List, Dict, Tuple, Union
from collections import Counter
from tqdm import tqdm
from joblib import Parallel, delayed


class PreDeduplicationProcessor:
    """ """

    def __init__(self, project_id: int):
        self._separators = r'[;.\[\]\(\)\~!\-\+\&\*/%<>\^\|\?\{\}=\#,"\\\:\$\'`@ +\n\r\t]'
        self._project_id = project_id

    def _get_diff_from_mods(self, mods: List[Dict[str, str]]) -> str:
        """


        We don't want to consider filenames when running duplicates search on diffs.
        """
        commit_diff = []
        for mod in mods:
            if mod["change_type"] == "UNKNOWN":
                continue

            commit_diff.append(mod["diff"])

        return " ".join(commit_diff)

    def _hash_string(self, x: str) -> str:
        hash = hashlib.md5()
        hash.update(x.encode("utf-8"))
        return hash.hexdigest()

    def _split_by_several_separators(self, x: str) -> List[str]:
        return [y.strip() for y in re.split(self._separators, x) if y]

    def _preprocess_single_example(
        self, id: int, example: Union[str, List[Dict[str, str]]], diff_mode: bool
    ) -> Tuple[str, int, int]:
        """
        This method does the following:
        1) Simple preprocessing:
          - diff: remove filenames and '@@ -0,0 +1 @@'-like git stuff
          - message: cast to lowercase
        2) Converting to following format:
          'token_hash@#@token1@@::@@frequency,token2@@::@@frequency,...'
        3) Calculating total # tokens and unique # tokens
        """
        data_col = "diff" if diff_mode else "message"
        # diff preprocessing
        if diff_mode:
            try:
                example = self._get_diff_from_mods(example)
                example = re.sub("\@\@ .*? \@\@\n", "", example)
            except TypeError as e:
                logging.error(f"[{data_col}] {id} produced TypeError {e}")
                example = str(example)
        # message preprocessing
        if not diff_mode:
            try:
                example = example.lower()
            except AttributeError as e:
                logging.error(f"[{data_col}] {id} produced AttributeError {e}")
                example = str(example)

        c = Counter(self._split_by_several_separators(example))
        tokens_enc = self._hash_string(example) + "@#@" + ",".join(f"{token}@@::@@{freq}" for token, freq in c.items())
        total_n_tokens = sum(c.values())
        unique_n_tokens = len(c)
        return tokens_enc, total_n_tokens, unique_n_tokens

    def preprocess_single_example(self, item: str, id: int, diff_mode: bool):
        if type(id) != int:
            try:
                id = int(id)
            except ValueError:
                logging.error(f"`id` is expected to be `int`, got `{type(id)} instead ({id})`")
                return ""

        tokens_enc, total_n_tokens, unique_n_tokens = self._preprocess_single_example(
            example=item, id=id, diff_mode=diff_mode
        )
        return f"{self._project_id},{id},{total_n_tokens},{unique_n_tokens},{tokens_enc}\n"

    def preprocess(self, input_filename: str, output_filename: str, chunksize: int, diff_mode: bool):
        """
        Processes each example in 'input_filename' (iterating in chunks of 'chunksize') to format
        'project_id,file_id,total_tokens,unique_tokens,token_hash@#@token1@@::@@frequency,token2@@::@@frequency,...'
        and saves result to 'n_tokens_dir'
        'diff_mode' = True -> processes diffs (in 'diff' column),
        otherwise -> processes messages (in 'message' column)
        """
        data_col = "diff" if diff_mode else "message"
        # make sure to clear target file
        open(output_filename, "w", encoding="utf-8").close()

        logging.info(f"[{data_col}] Starting processing")

        reader = pd.read_csv(input_filename, chunksize=chunksize)
        for chunk in tqdm(reader):

            with Parallel(8) as pool:
                res = pool(
                    delayed(self.preprocess_single_example)(item=item[data_col], id=item["id"], diff_mode=diff_mode)
                    for _, item in chunk[["id", data_col]].iterrows()
                )

            with open(output_filename, "a", encoding="utf-8") as target:
                target.writelines(res)

        logging.info(f"[{data_col}] Finished processing")
