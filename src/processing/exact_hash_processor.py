import hashlib
import json
import os
from typing import Any, Dict, List, Literal, Optional

import jsonlines
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, StringType, StructField, StructType
from tqdm import tqdm

from ..utils import JsonlManager, get_logger


class ExactHashProcessor:
    """This class is used to find all pairs of clones in terms of md5 hash of tokens.

    It builds off files preprocessed for SourcererCC.
    """

    def __init__(
        self,
        data_format: str,
        chunksize: int = 1000,
        logger_name: Optional[str] = None,
        n_workers: Optional[int] = None,
    ):
        if data_format == "jsonl":
            self._data_manager = JsonlManager()
        else:
            raise ValueError("Given data format is not supported.")
        self.data_format = data_format
        self._chunksize = chunksize
        self._logger_name = logger_name
        self._n_workers = n_workers
        self._spark = (
            SparkSession.builder.master("local[*]")
            .config("spark.driver.memory", "18g")
            .appName("CloneSearch")
            .getOrCreate()
        )

    @property
    def logger(self):
        return get_logger(self._logger_name)

    def _hash_string(self, x: str) -> str:
        """Obtains hash of given string."""
        hash = hashlib.md5()
        hash.update(x.encode("utf-8"))
        return hash.hexdigest()

    def _preprocess_files(self, input_path: str, hash_path: str, use_tokens_hash: bool) -> None:
        """This method obtains hashes from examples preprocessed to SourcererCC format.

        Args:
            input_path: Path to file with examples preprocessed to SourcererCC format.
            hash_path: Path to output file with hashes (format: <part_id>,<id>,<hash>).
            use_tokens_hash: True to calculate hash of dict of tokens, False to reuse hash of the whole string.
        """
        self.logger.info(f"Start processing hashes")

        open(hash_path, "w").close()

        current_examples: List[str] = []
        with open(input_path, "r") as in_f:
            for line in tqdm(in_f, desc=f"Calculating hashes for {input_path}"):

                if len(current_examples) > self._chunksize:
                    with open(hash_path, "a") as out_f:
                        out_f.writelines(current_examples)
                        current_examples = []

                info, tokens = line.strip().split("@#@")
                part_idx, idx, num_tokens, num_unique_tokens, hash = info.split(",")

                if not int(num_unique_tokens):
                    self.logger.info(f"({part_idx}, {idx}) Has 0 tokens")
                    continue
                if use_tokens_hash:
                    tokens = {token.split("@@::@@")[0]: int(token.split("@@::@@")[1]) for token in tokens.split(",")}
                    hash = self._hash_string(json.dumps(tokens, sort_keys=True))
                current_examples.append(f"{part_idx},{idx},{hash}\n")

        if len(current_examples) > 0:
            with open(hash_path, "a") as out_f:
                out_f.writelines(current_examples)
        self.logger.info(f"Finish processing hashes")

    def _aggregate_spark_results(self, root_dir: str, output_path: str) -> None:
        open(output_path, "w").close()

        for fname in os.listdir(root_dir):
            if fname.startswith("part") and fname.endswith("json"):
                chunk: List[Dict[str, Any]] = []

                with jsonlines.open(os.path.join(root_dir, fname), "r") as reader:
                    for line in reader:
                        if len(chunk) > self._chunksize:
                            with jsonlines.open(output_path, "a") as writer:
                                writer.write_all(chunk)
                            chunk = []

                        chunk.append({"hash": line["hash"], "clones": line["collect_list(array(part_idx, idx))"]})

                if len(chunk) > 0:
                    with jsonlines.open(output_path, "a") as writer:
                        writer.write_all(chunk)

    def _calculate_clones(self, input_path: str, output_root_dir: str) -> None:
        self.logger.info(f"Start processing clones from {input_path}")

        schema = StructType(
            [
                StructField("part_idx", IntegerType(), True),
                StructField("idx", IntegerType(), True),
                StructField("hash", StringType(), True),
            ]
        )
        df = self._spark.read.csv(input_path, header=False, schema=schema)

        df_grouped = (
            df.groupby("hash")
            .agg(F.collect_list(F.array("part_idx", "idx")))
            .where(F.size("collect_list(array(part_idx, idx))") > 1)
        )
        df_grouped.write.json(os.path.join(output_root_dir, "results"))

        self._aggregate_spark_results(
            root_dir=os.path.join(output_root_dir, "results"),
            output_path=os.path.join(output_root_dir, "results.jsonl"),
        )

        self.logger.info(f"Finish processing clones from {input_path}")

    def __call__(
        self,
        deduplication_root: str,
        parts: List[str],
        data_type: Literal["diffs", "messages"],
        use_tokens_hash: bool = True,
        use_cache: bool = False,
    ):
        assert parts[0] == "train"

        os.makedirs(os.path.join(deduplication_root, "hash"), exist_ok=True)
        os.makedirs(os.path.join(deduplication_root, "results", "exact_hash", data_type), exist_ok=True)

        if use_tokens_hash:
            hash_path = os.path.join(deduplication_root, "hash", f"res_{data_type}_tokens.txt")
        else:
            hash_path = os.path.join(deduplication_root, "hash", f"res_{data_type}_str.txt")
        if not use_cache:
            self._preprocess_files(
                input_path=os.path.join(deduplication_root, "raw", f"res_{data_type}.txt"),
                hash_path=hash_path,
                use_tokens_hash=use_tokens_hash,
            )
        self._calculate_clones(
            input_path=hash_path,
            output_root_dir=os.path.join(
                deduplication_root, "results" + ("_tokens" if use_tokens_hash else "_str"), "exact_hash", data_type
            ),
        )
