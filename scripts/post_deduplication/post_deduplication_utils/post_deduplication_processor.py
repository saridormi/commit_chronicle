import csv
import os
import logging
import pandas as pd
from tqdm import tqdm
from typing import Optional


class PostDeduplicationProcessor:
    def __init__(self, raw_data_dir: str, deduplication_dir: str, chunksize: int = 5000):
        self._raw_data_dir = raw_data_dir
        self._deduplication_dir = deduplication_dir
        self._chunksize = chunksize
        self._train_full_clones = None

    def _extract_metadata(self):
        """Extract metadata (author, timestamp, repo, hash) for each commit."""
        for i, part in enumerate(["train", "val", "test", "val_original", "test_original"]):
            if f"{part}_metadata.csv" not in os.listdir(self._deduplication_dir):
                logging.info(f"Extracting metadata from {part}")
                output_filename = os.path.join(self._deduplication_dir, f"{part}_metadata.csv")
                fieldnames = ["project_id", "id", "author", "date", "hash", "repo"]
                with open(output_filename, "w") as _:
                    pass
                reader = pd.read_csv(os.path.join(self._raw_data_dir, f"{part}_no_outliers.csv"),
                                     chunksize=self._chunksize)
                for chunk in tqdm(reader, desc=f"Iterating over {part} to extract metadata"):
                    chunk["project_id"] = i + 1
                    chunk[fieldnames].to_csv(output_filename, mode="a", index=False, header=False)

        with open(os.path.join(self._deduplication_dir, "metadata.csv"), "w") as outfile:
            for part in ["train", "val", "test", "val_original", "test_original"]:
                with open(os.path.join(self._deduplication_dir, f"{part}_metadata.csv"), "r") as infile:
                    for line in infile:
                        outfile.write(line)

    def _add_metadata(self, input_filename: str, output_filename: str):
        """
        For each pair of clones from `input_filename`, find corresponding metadata and save it to `output_filename`.
        """
        logging.info(f"Adding metadata to {input_filename}")
        df = pd.read_csv(os.path.join(self._deduplication_dir, "metadata.csv"), header=None,
                         names=["project_id", "id", "author", "date", "hash", "repo"]).sort_values(by=["project_id", "id"])
        indexes = {}
        for idx, row in tqdm(df.iterrows()):
            indexes[(row["project_id"], row["id"])] = idx
        data = df.to_numpy()

        fieldnames = ["part_id1", "id1", "repo1", "hash1", "part_id2", "id2", "repo2", "hash2"]
        with open(output_filename, "w") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

        metadata = []

        with open(input_filename, "r") as file:
            for i, line in tqdm(enumerate(file), desc=f"Iterating over {input_filename} to add metadata"):
                pr_1, s_1, pr_2, s_2 = (int(j) for j in line.strip().split(","))
                ex1 = data[indexes[(pr_1, s_1)]]
                ex2 = data[indexes[(pr_2, s_2)]]

                metadata.append({"part_id1": ex1[0],
                                 "id1": ex1[1],
                                 "repo1": ex1[5],
                                 "hash1": ex1[4],
                                 "part_id2": ex2[0],
                                 "id2": ex2[1],
                                 "repo2": ex2[5],
                                 "hash2": ex2[4],
                                 })

                if i % self._chunksize == 0:
                    with open(output_filename, "a") as csvfile:
                        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                        writer.writerows(metadata)
                        metadata = []

        if len(metadata) > 0:
            with open(output_filename, "a") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writerows(metadata)

    def _get_full_clones(self, msg_filename: str, diff_filename: str):
        """
        Get ids of examples from train which are duplicates to some examples from train/val/test in terms of
        both messages and diffs.
        """
        # get train clones by messages
        train_msgs_clones = set()
        with open(msg_filename, "r") as file:
            for line in tqdm(file, desc="Reading message clones"):
                # expected format: ["part_id1", "id1", "repo1", "hash1", "part_id2", "id2", "repo2", "hash2"]
                pr_1, s_1, repo1, hash1, pr_2, s_2, repo2, hash2 = line.strip().split(",")
                if pr_1 == "1":
                    train_msgs_clones.add(f"{pr_1},{s_1},{repo1},{hash1},{pr_2},{s_2},{repo2},{hash2}\n")
                elif pr_1 != "1" and pr_2 == "1":
                    train_msgs_clones.add(f"{pr_2},{s_2},{repo2},{hash2},{pr_1},{s_1},{repo1},{hash1}\n")

        # get train clones by diffs
        train_diffs_clones = set()
        with open(diff_filename, "r") as file:
            for line in tqdm(file, desc="Reading diff clones"):
                # expected format: ["part_id1", "id1", "repo1", "hash1", "part_id2", "id2", "repo2", "hash2"]
                pr_1, s_1, repo1, hash1, pr_2, s_2, repo2, hash2 = line.strip().split(",")
                if pr_1 == "1":
                    train_diffs_clones.add(f"{pr_1},{s_1},{repo1},{hash1},{pr_2},{s_2},{repo2},{hash2}\n")
                elif pr_1 != "1" and pr_2 == "1":
                    train_diffs_clones.add(f"{pr_2},{s_2},{repo2},{hash2},{pr_1},{s_1},{repo1},{hash1}\n")

        self._train_full_clones = train_msgs_clones.intersection(train_diffs_clones)
        fieldnames = ["part_id1", "id1", "repo1", "hash1", "part_id2", "id2", "repo2", "hash2"]
        with open(os.path.join(self._deduplication_dir, "full_clones.csv"), "w") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

        with open(os.path.join(self._deduplication_dir, "full_clones.csv"), "a") as csvfile:
            for row in self._train_full_clones:
                csvfile.write(row)

    def _drop_duplicates(self):
        logging.info(f"Got {len(self._train_full_clones)} ids to drop")
        ids_to_drop = set(int(pair.split(",")[1]) for pair in self._train_full_clones)

        input_filename = os.path.join(self._raw_data_dir, f"train_no_outliers.csv")
        output_filename = os.path.join(self._raw_data_dir, f"train_no_duplicates.csv")

        fieldnames = ["id", "author", "date", "hash", "message", "diff", "repo"]
        with open(output_filename, "w") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

        reader = pd.read_csv(input_filename, chunksize=self._chunksize)
        n_dropped = 0
        for chunk in tqdm(reader, desc="Iterating over train data to drop duplicates"):
            n_before_drop = len(chunk)
            dedup_chunk = chunk.loc[~chunk["id"].isin(ids_to_drop)]
            n_dropped += n_before_drop - len(dedup_chunk)

            dedup_chunk[["id", "author", "date", "hash", "message", "diff", "repo"]].to_csv(
                output_filename, mode="a", index=False, header=False
            )
            logging.info(
                f"{(n_before_drop - len(dedup_chunk)) / n_before_drop * 100:.2f}% duplicates in last chunk"
                f" ({n_dropped} duplicates total)"
            )

    def __call__(self, msg_filename: str, diff_filename: str,
                 raw_msg_filename: Optional[str] = None, raw_diff_filename: Optional[str] = None):
        if not os.path.isfile(os.path.join(self._deduplication_dir, "metadata.csv")):
            self._extract_metadata()

        if msg_filename not in os.listdir(self._deduplication_dir):
            self._add_metadata(input_filename=os.path.join(self._deduplication_dir, raw_msg_filename),
                               output_filename=os.path.join(self._deduplication_dir, msg_filename))
        if diff_filename not in os.listdir(self._deduplication_dir):
            self._add_metadata(input_filename=os.path.join(self._deduplication_dir, raw_diff_filename),
                               output_filename=os.path.join(self._deduplication_dir, diff_filename))

        self._get_full_clones(msg_filename=os.path.join(self._deduplication_dir, msg_filename),
                              diff_filename=os.path.join(self._deduplication_dir, diff_filename))
        self._drop_duplicates()
