import re
import argparse
import os
import hashlib
import logging
import gzip
import pandas as pd
from typing import List, Tuple
from collections import Counter
from tqdm import tqdm
from joblib import Parallel, delayed

logging.basicConfig(
    filename="../logs/deduplication.log",
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)


class DeduplicationPreprocessor:
    def __init__(self):
        self._separators = r'[;.\[\]\(\)\~!\-\+\&\*/%<>\^\|\?\{\}=\#,"\\\:\$\'`@ +\n\r\t]'

    def _remove_filenames(self, x: str) -> str:
        """
        We don't want to consider filenames when running duplicates search on diffs.

        This method removes all filename patterns:
        * path/to/file.py                              - when file is modified
        (this case is parsed with more complex and error-prone regex)
        * rename from filename1 \n rename to filename2 - when file is renamed
        * copy from filename1 \n copy to filename2     - when file is copied
        * new file filename                            - when file is created
        * deleted file filename                        - when file is deleted
        """
        x = re.sub("^(\/?[\w\-_]+\/)*?([\w\-_]+\.[\w\-_]+?)*?\n", "", x)
        x = re.sub("rename from .*?\n.*?\n", "", x)
        x = re.sub("copy from .*?\n.*?\n", "", x)
        x = re.sub("new file .*?\n", "", x)
        x = re.sub("deleted file .*?\n", "", x)
        return x

    def hash_string(self, x: str) -> str:
        hash = hashlib.md5()
        hash.update(x.encode("utf-8"))
        return hash.hexdigest()

    def split_by_several_separators(self, x: str) -> List[str]:
        return [y.strip() for y in re.split(self._separators, x) if y]

    def _preprocess_single_example(self, example: str, id: int, diff_mode: bool) -> Tuple[str, int, int]:
        """
        1) Does super simple preprocessing:
          * diff: remove filenames and '@@ -0,0 +1 @@'-like git stuff
          * message: cast to lowercase
        2) Processes resulting string to following format:
          'token_hash@#@token1@@::@@frequency,token2@@::@@frequency,...'
        3) Calculates total # tokens and unique # tokens
        """
        data_col = "diff" if diff_mode else "message"
        # diff preprocessing
        if diff_mode:
            try:
                example = self._remove_filenames(example)
                example = re.sub("\@\@ .*? \@\@\n", "", example)
            except TypeError:
                logging.error(f"[{data_col}] {id}: `{example}` is not a string")
                example = str(example)
        # message preprocessing
        if not diff_mode:
            try:
                example = example.lower()
            except AttributeError:
                logging.error(f"[{data_col}] {id}: `{example}` is not a string")
                example = str(example)

        c = Counter(self.split_by_several_separators(example))
        tokens_enc = self.hash_string(example) + "@#@" + ",".join(f"{token}@@::@@{freq}" for token, freq in c.items())
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
        return f"1,{id},{total_n_tokens},{unique_n_tokens},{tokens_enc}\n"

    def preprocess(self, csv_filename: str, chunksize: int, output_dir: str, diff_mode: bool):
        """
        Processes each example in 'csv_filename' (iterating in chunks of 'chunksize') to format
        'project_id,file_id,total_tokens,unique_tokens,token_hash@#@token1@@::@@frequency,token2@@::@@frequency,...'
        and saves result to 'output_dir'
        'diff_mode' = True -> processes diffs (in 'diff' column),
        otherwise -> processes messages (in 'message' column)
        """
        data_col = "diff" if diff_mode else "message"
        # make sure to clear target file
        os.makedirs(output_dir, exist_ok=True)
        open(os.path.join(output_dir, f"res_{data_col}.txt"), "w", encoding="utf-8").close()

        logging.info(f"[{data_col}] Starting processing")

        reader = pd.read_csv(csv_filename, chunksize=chunksize, index_col="id")
        for chunk in tqdm(reader, total=2846334 // chunksize + 1):

            with Parallel(8) as pool:
                res = pool(
                    delayed(self.preprocess_single_example)(item=item, id=id, diff_mode=diff_mode)
                    for id, item in chunk[data_col].items()
                )

            with open(os.path.join(output_dir, f"res_{data_col}.txt"), "a", encoding="utf-8") as target:
                target.writelines(res)

        logging.info(f"[{data_col}] Compressing with gzip")

        with open(os.path.join(output_dir, f"res_{data_col}.txt"), "rb") as f_in, gzip.open(
            os.path.join(output_dir, f"res_{data_col}.txt.gz"), "wb"
        ) as f_out:
            f_out.writelines(f_in)
        os.remove(os.path.join(output_dir, f"res_{data_col}.txt"))

        logging.info(f"[{data_col}] Finished processing")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="This script processes .csv file with data into necessary format for deduplication",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="../deduplication",
        help="path to directory to save processed data",
    )
    parser.add_argument("--csv_filename", type=str, default="../commits_fxd.csv", help="path to .csv file with data")
    parser.add_argument("--diff_mode", type=bool, help="`True` to process diffs and `False` to process messages")
    parser.add_argument("--chunksize", type=int, default=1000, help="# of examples to process at one step")
    args = parser.parse_args()

    dp = DeduplicationPreprocessor()
    dp.preprocess(
        csv_filename=args.csv_filename, chunksize=args.chunksize, output_dir=args.output_dir, diff_mode=args.diff_mode
    )
