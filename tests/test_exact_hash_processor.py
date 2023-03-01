import gzip
import hashlib
import itertools
import json
import os

import jsonlines
import pytest

from src.processing import ExactHashProcessor, PreDeduplicationProcessor


@pytest.fixture
def default_processors():
    return PreDeduplicationProcessor(special_tokens=[], data_format="jsonl"), ExactHashProcessor(
        data_format="jsonl", chunksize=1
    )


def preprocess_files_sourc(tmp_path, default_processors):
    pred_processor, _ = default_processors

    for i in range(10):
        os.makedirs(f"{tmp_path}/data/train/repo{i}", exist_ok=True)
        with jsonlines.open(f"{tmp_path}/data/train/repo{i}/commits.jsonl", "w") as writer:
            writer.write(
                {
                    "hash": f"hash{i}",
                    "mods": [
                        {
                            "change_type": "MODIFY",
                            "old_path": "fname",
                            "new_path": "fname",
                            "diff": f"diff{i % 2}",
                        }
                    ],
                    "message": f"message{i % 5}",
                }
            )
        with open(f"{tmp_path}/data/train/repo{i}/commits.jsonl", "rb") as f_in, gzip.open(
            f"{tmp_path}/data/train/repo{i}/commits.jsonl.gz", "wb"
        ) as f_out:
            f_out.writelines(f_in)
        os.remove(f"{tmp_path}/data/train/repo{i}/commits.jsonl")
    assert set(os.listdir(f"{tmp_path}/data/train")) == {f"repo{i}" for i in range(10)}

    for i in range(10):
        os.makedirs(f"{tmp_path}/data/test/repo{i + 100}", exist_ok=True)
        with jsonlines.open(f"{tmp_path}/data/test/repo{i + 100}/commits.jsonl", "w") as writer:
            writer.write(
                {
                    "hash": f"hash{i + 100}",
                    "mods": [
                        {
                            "change_type": "MODIFY",
                            "old_path": "fname",
                            "new_path": "fname",
                            "diff": f"diff{i % 2}",
                        }
                    ],
                    "message": f"message{i % 5}",
                }
            )
        with open(f"{tmp_path}/data/test/repo{i + 100}/commits.jsonl", "rb") as f_in, gzip.open(
            f"{tmp_path}/data/test/repo{i + 100}/commits.jsonl.gz", "wb"
        ) as f_out:
            f_out.writelines(f_in)
        os.remove(f"{tmp_path}/data/test/repo{i + 100}/commits.jsonl")
    assert set(os.listdir(f"{tmp_path}/data/test")) == {f"repo{i + 100}" for i in range(10)}

    os.makedirs(f"{tmp_path}/pred_processor_output", exist_ok=True)
    for i, part in enumerate(["train", "test"]):
        pred_processor(
            input_dir=f"{tmp_path}/data/{part}",
            diff_fname=f"{tmp_path}/pred_processor_output/{part}_diffs.txt",
            message_fname=f"{tmp_path}/pred_processor_output/{part}_messages.txt",
            part=part,
            project_id=i + 1,
        )
    for part in ["train", "test"]:
        for data_type in ["messages", "diffs"]:
            with open(f"{tmp_path}/pred_processor_output/{part}_{data_type}.txt", "r") as in_f:
                with open(f"{tmp_path}/pred_processor_output/res_{data_type}.txt", "a") as out_f:
                    out_f.writelines(line for line in in_f)


def preprocess_files_tokens(tmp_path, default_processors):
    preprocess_files_sourc(tmp_path, default_processors)
    _, exact_hash_processor = default_processors
    os.makedirs(str(tmp_path / "exact_hash_processor_output"), exist_ok=True)
    exact_hash_processor._preprocess_files(
        input_path=f"{tmp_path}/pred_processor_output/res_messages.txt",
        hash_path=f"{tmp_path}/exact_hash_processor_output/res_messages.txt",
        use_tokens_hash=True,
    )
    exact_hash_processor._preprocess_files(
        input_path=f"{tmp_path}/pred_processor_output/res_diffs.txt",
        hash_path=f"{tmp_path}/exact_hash_processor_output/res_diffs.txt",
        use_tokens_hash=True,
    )


def test_preprocess_files(tmp_path, default_processors):
    preprocess_files_tokens(tmp_path, default_processors)

    for data_type in ["diffs", "messages"]:
        with open(f"{tmp_path}/exact_hash_processor_output/res_{data_type}.txt", "r") as tokens_f:
            with open(f"{tmp_path}/pred_processor_output/res_{data_type}.txt", "r") as sourc_f:
                for tokens_line, sourc_line in zip(tokens_f, sourc_f):
                    sourc_info, sourc_tokens = sourc_line.strip().split("@#@")
                    sourc_proj_idx, sourc_idx, _, _, sourc_hash = sourc_info.split(",")
                    tokens_proj_idx, tokens_idx, tokens_hash = tokens_line.strip().split(",")

                    assert sourc_proj_idx == tokens_proj_idx
                    assert sourc_idx == tokens_idx

                    h = hashlib.md5()
                    h.update(json.dumps({sourc_tokens.split("@@::@@")[0]: 1}, sort_keys=True).encode("utf-8"))
                    assert tokens_hash == h.hexdigest(), "Hash should be calculated from tokens"


def test_calculate_clones(tmp_path, default_processors):
    preprocess_files_tokens(tmp_path, default_processors)
    _, exact_hash_processor = default_processors

    for data_type in ["diffs", "messages"]:
        output_root_dir = os.path.join(tmp_path, "exact_hash_processor_output", data_type)
        os.makedirs(output_root_dir, exist_ok=True)
        exact_hash_processor._calculate_clones(
            input_path=os.path.join(tmp_path, "exact_hash_processor_output", f"res_{data_type}.txt"),
            output_root_dir=output_root_dir,
        )

        with jsonlines.open(os.path.join(output_root_dir, "results.jsonl"), "r") as reader:
            groups = [line for line in reader]

        if data_type == "diffs":
            assert len(groups) == 2

            hashes = {}
            for i in range(2):
                h = hashlib.md5()
                h.update(json.dumps({f"diff{i}": 1}, sort_keys=True).encode("utf-8"))
                hashes[h.hexdigest()] = i

            for group in groups:
                remainder = hashes[group["hash"]]
                assert group["clones"] == [[1, i] for i in range(10) if i % 2 == remainder] + [
                    [2, i + 10] for i in range(10) if i % 2 == remainder
                ]

        elif data_type == "messages":
            assert len(groups) == 5

            hashes = {}
            for i in range(5):
                h = hashlib.md5()
                h.update(json.dumps({f"message{i}": 1}, sort_keys=True).encode("utf-8"))
                hashes[h.hexdigest()] = i

            for group in groups:
                remainder = hashes[group["hash"]]
                assert group["clones"] == [[1, i] for i in range(10) if i % 5 == remainder] + [
                    [2, i + 10] for i in range(10) if i % 5 == remainder
                ]
