import gzip
import json
import os

import jsonlines
import numpy as np

from src.processing import OutliersProcessor


def test_get_stats_mods():
    processor = OutliersProcessor(
        lower_percentile=None,  # irrelevant to this test
        upper_percentile=None,  # irrelevant to this test
        data_format="jsonl",
    )
    mods = [{"change_type": "MODIFY", "new_path": "sample_fname", "old_path": "sample_fname", "diff": "sample diff"}]

    assert processor._get_stats_mods("repo", "hash", mods) == {
        "repo": "repo",
        "hash": "hash",
        "num_tokens": 3,
        "num_chars": len("sample_fname\n") + len("sample diff"),
        "num_mods": 1,
    }
    longer_mods = []
    for _ in range(5):
        longer_mods.append(mods[0])

    assert processor._get_stats_mods("repo", "hash", longer_mods) == {
        "repo": "repo",
        "hash": "hash",
        "num_tokens": 5 * 3,
        "num_chars": 5 * (len("sample_fname\n") + len("sample diff")),
        "num_mods": 5,
    }

    strange_mods = [
        {"change_type": "MODIFY", "new_path": "sample_fname", "old_path": "sample_fname", "diff": None},
        {"change_type": "MODIFY", "new_path": "sample_fname", "old_path": "sample_fname", "diff": "sample diff"},
    ]

    assert processor._get_stats_mods("repo", "hash", strange_mods) == {
        "repo": "repo",
        "hash": "hash",
        "num_tokens": None,
        "num_chars": None,
        "num_mods": None,
    }

    strange_mods = [
        {"change_type": "MODIFY", "new_path": "sample_fname", "old_path": "sample_fname", "diff": "sample diff"},
        {"change_type": "MODIFY", "new_path": "sample_fname", "old_path": "sample_fname", "diff": np.nan},
    ]

    assert processor._get_stats_mods("repo", "hash", strange_mods) == {
        "repo": "repo",
        "hash": "hash",
        "num_tokens": None,
        "num_chars": None,
        "num_mods": None,
    }


def test_get_stats_msg():
    processor = OutliersProcessor(
        lower_percentile=None,  # irrelevant to this test
        upper_percentile=None,  # irrelevant to this test
        data_format="jsonl",
        chunksize=1,
    )

    assert processor._get_stats_msg("repo", "hash", "Random message") == {
        "repo": "repo",
        "hash": "hash",
        "num_tokens": 2,
        "num_chars": len("Random message"),
    }

    assert processor._get_stats_msg("repo", "hash", None) == {
        "repo": "repo",
        "hash": "hash",
        "num_tokens": None,
        "num_chars": None,
    }

    assert processor._get_stats_msg("repo", "hash", np.nan) == {
        "repo": "repo",
        "hash": "hash",
        "num_tokens": None,
        "num_chars": None,
    }


def test_get_stats(tmp_path):
    os.makedirs(f"{tmp_path}/stats")
    os.makedirs(f"{tmp_path}/data/repo")
    processor = OutliersProcessor(
        lower_percentile=None,  # irrelevant to this test
        upper_percentile=None,  # irrelevant to this test
        data_format="jsonl",
        chunksize=1,
    )

    with jsonlines.open(f"{tmp_path}/data/repo/commits.jsonl", "w") as writer:
        writer.write_all(
            [
                {
                    "hash": "hash1",
                    "message": "sample message",
                    "mods": [
                        {
                            "change_type": "MODIFY",
                            "new_path": "sample_fname",
                            "old_path": "sample_fname",
                            "diff": "sample diff",
                        }
                    ],
                },
                {
                    "hash": "hash2",
                    "message": "sample message",
                    "mods": [
                        {
                            "change_type": "MODIFY",
                            "new_path": "sample_fname",
                            "old_path": "sample_fname",
                            "diff": None,
                        }
                    ],
                },
                {
                    "hash": "hash3",
                    "message": None,
                    "mods": [
                        {
                            "change_type": "MODIFY",
                            "new_path": "sample_fname",
                            "old_path": "sample_fname",
                            "diff": "sample diff",
                        }
                    ],
                },
                {
                    "hash": "hash4",
                    "message": None,
                    "mods": [
                        {
                            "change_type": "MODIFY",
                            "new_path": "sample_fname",
                            "old_path": "sample_fname",
                            "diff": None,
                        }
                    ],
                },
                {
                    "hash": "hash5",
                    "message": "another example of message\nthis time longer",
                    "mods": [
                        {
                            "change_type": "MODIFY",
                            "new_path": "sample_fname",
                            "old_path": "sample_fname",
                            "diff": "-another example of diff\n+this time longer\n",
                        }
                    ],
                },
            ]
        )
    with open(f"{tmp_path}/data/repo/commits.jsonl", "rb") as f_in, gzip.open(
        f"{tmp_path}/data/repo/commits.jsonl.gz", "wb"
    ) as f_out:
        f_out.writelines(f_in)

    processor._get_stats(input_dir=f"{tmp_path}/data", stats_dir=f"{tmp_path}/stats")
    with jsonlines.open(os.path.join(f"{tmp_path}/stats", "stats_diff.jsonl"), "r") as reader:
        stats_diff = [line for line in reader]
    assert stats_diff[0] == {
        "repo": "repo",
        "hash": "hash1",
        "num_tokens": 3,
        "num_chars": len("sample_fname\n") + len("sample diff"),
        "num_mods": 1,
    }
    assert stats_diff[1] == {"repo": "repo", "hash": "hash2", "num_tokens": None, "num_chars": None, "num_mods": None}
    assert stats_diff[2] == {
        "repo": "repo",
        "hash": "hash3",
        "num_tokens": 3,
        "num_chars": len("sample_fname\n") + len("sample diff"),
        "num_mods": 1,
    }
    assert stats_diff[3] == {"repo": "repo", "hash": "hash4", "num_tokens": None, "num_chars": None, "num_mods": None}
    assert stats_diff[4] == {
        "repo": "repo",
        "hash": "hash5",
        "num_tokens": 8,
        "num_chars": len("sample_fname\n") + len("-another example of diff\n+this time longer\n"),
        "num_mods": 1,
    }

    with jsonlines.open(os.path.join(f"{tmp_path}/stats", "stats_message.jsonl"), "r") as reader:
        stats_message = [line for line in reader]
    assert stats_message[0] == {"repo": "repo", "hash": "hash1", "num_tokens": 2, "num_chars": len("sample message")}
    assert stats_message[1] == {"repo": "repo", "hash": "hash2", "num_tokens": 2, "num_chars": len("sample message")}
    assert stats_message[2] == {"repo": "repo", "hash": "hash3", "num_tokens": None, "num_chars": None}
    assert stats_message[3] == {"repo": "repo", "hash": "hash4", "num_tokens": None, "num_chars": None}
    assert stats_message[4] == {
        "repo": "repo",
        "hash": "hash5",
        "num_tokens": 7,
        "num_chars": len("another example of message\nthis time longer"),
    }


def test_get_percentiles(tmp_path):
    os.makedirs(f"{tmp_path}/stats")
    processor = OutliersProcessor(
        lower_percentile=None,  # irrelevant to this test
        upper_percentile=None,  # irrelevant to this test
        data_format="jsonl",
        chunksize=1,
    )
    with jsonlines.open(f"{tmp_path}/stats/stats_diff.jsonl", "w") as writer:
        writer.write_all(
            [
                {"repo": "repo", "hash": f"hash{i}", "num_tokens": i + 1, "num_chars": i + 1, "num_mods": i + 1}
                for i in range(100)
            ]
            + [{"repo": "repo", "hash": "hash101", "num_tokens": None, "num_chars": None, "num_mods": None}]
        )

    with jsonlines.open(f"{tmp_path}/stats/stats_message.jsonl", "w") as writer:
        writer.write_all(
            [{"repo": "repo", "hash": f"hash{i}", "num_tokens": i + 100, "num_chars": i + 100} for i in range(100)]
            + [{"repo": "repo", "hash": "hash101", "num_tokens": None, "num_chars": None}]
        )

    processor._get_percentiles(stats_dir=f"{tmp_path}/stats")

    for key in ["num_tokens", "num_chars", "num_mods"]:
        for p in [5, 95]:
            assert processor._diff_percentiles[p][key] == np.percentile([i + 1 for i in range(100)], p)

    for key in ["num_tokens", "num_chars"]:
        for p in [5, 95]:
            assert processor._message_percentiles[p][key] == np.percentile([i + 100 for i in range(100)], p)


def test_read_percentiles(tmp_path):
    os.makedirs(f"{tmp_path}/stats")
    processor = OutliersProcessor(
        lower_percentile=None,  # irrelevant to this test
        upper_percentile=None,  # irrelevant to this test
        data_format="jsonl",
    )
    with open(f"{tmp_path}/stats/diff.json", "w") as f:
        json.dump(
            {
                p: {
                    key: np.percentile([i + 100 for i in range(100)], p)
                    for key in ["num_tokens", "num_chars", "num_mods"]
                }
                for p in [5, 95]
            },
            f,
        )

    with open(f"{tmp_path}/stats/message.json", "w") as f:
        json.dump(
            {
                p: {key: np.percentile([i + 1 for i in range(100)], p) for key in ["num_tokens", "num_chars"]}
                for p in [5, 95]
            },
            f,
        )

    processor._read_percentiles(percentile_dir=f"{tmp_path}/stats")

    for key in ["num_tokens", "num_chars", "num_mods"]:
        for p in [5, 95]:
            assert processor._diff_percentiles[p][key] == np.percentile([i + 100 for i in range(100)], p)

    for key in ["num_tokens", "num_chars"]:
        for p in [5, 95]:
            assert processor._message_percentiles[p][key] == np.percentile([i + 1 for i in range(100)], p)


def test_get_ids_to_drop(tmp_path):
    os.makedirs(f"{tmp_path}/stats")
    processor = OutliersProcessor(
        lower_percentile=5,
        upper_percentile=95,
        data_format="jsonl",
    )

    processor._diff_percentiles = {
        5: {"num_tokens": 2, "num_chars": 3, "num_mods": 1},
        95: {"num_tokens": 5, "num_chars": 5, "num_mods": 4},
    }

    processor._message_percentiles = {
        5: {"num_tokens": 2, "num_chars": 3},
        95: {"num_tokens": 4, "num_chars": 5},
    }

    with jsonlines.open(f"{tmp_path}/stats/stats_diff.jsonl", "w") as writer:
        writer.write_all(
            [
                {"repo": "repo", "hash": f"hash{i}", "num_tokens": i + 1, "num_chars": i + 1, "num_mods": i + 1}
                for i in range(5)
            ]
            + [{"repo": "repo", "hash": f"hash101", "num_tokens": None, "num_chars": None, "num_mods": None}]
        )

    with jsonlines.open(f"{tmp_path}/stats/stats_message.jsonl", "w") as writer:
        writer.write_all(
            [
                {"repo": "repo", "hash": f"hash{i}", "num_tokens": i - 5 + 1, "num_chars": i - 5 + 1}
                for i in range(5, 10)
            ]
            + [{"repo": "repo", "hash": f"hash102", "num_tokens": None, "num_chars": None}]
        )

    processor._get_ids_to_drop(stats_dir=f"{tmp_path}/stats")

    assert processor._commits_to_drop == {"repo": {f"hash{i}" for i in [0, 1, 4, 5, 6, 9, 101, 102]}}
