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

    assert processor._get_stats_mods(0, mods) == {
        "id": 0,
        "num_tokens": 3,
        "num_chars": len("sample_fname\n") + len("sample diff"),
        "num_mods": 1,
    }
    longer_mods = []
    for _ in range(5):
        longer_mods.append(mods[0])

    assert processor._get_stats_mods(0, longer_mods) == {
        "id": 0,
        "num_tokens": 5 * 3,
        "num_chars": 5 * (len("sample_fname\n") + len("sample diff")),
        "num_mods": 5,
    }

    strange_mods = [
        {"change_type": "MODIFY", "new_path": "sample_fname", "old_path": "sample_fname", "diff": None},
        {"change_type": "MODIFY", "new_path": "sample_fname", "old_path": "sample_fname", "diff": "sample diff"},
    ]

    assert processor._get_stats_mods(0, strange_mods) == {
        "id": 0,
        "num_tokens": None,
        "num_chars": None,
        "num_mods": None,
    }

    strange_mods = [
        {"change_type": "MODIFY", "new_path": "sample_fname", "old_path": "sample_fname", "diff": "sample diff"},
        {"change_type": "MODIFY", "new_path": "sample_fname", "old_path": "sample_fname", "diff": np.nan},
    ]

    assert processor._get_stats_mods(0, strange_mods) == {
        "id": 0,
        "num_tokens": None,
        "num_chars": None,
        "num_mods": None,
    }


def test_get_stats_msg():
    processor = OutliersProcessor(
        lower_percentile=None,  # irrelevant to this test
        upper_percentile=None,  # irrelevant to this test
        data_format="jsonl",
    )

    assert processor._get_stats_msg(0, "Random message") == {
        "id": 0,
        "num_tokens": 2,
        "num_chars": len("Random message"),
    }

    assert processor._get_stats_msg(0, None) == {"id": 0, "num_tokens": None, "num_chars": None}

    assert processor._get_stats_msg(0, np.nan) == {"id": 0, "num_tokens": None, "num_chars": None}


def test_get_stats(tmp_path):
    os.makedirs(f"{tmp_path}/stats")
    processor = OutliersProcessor(
        lower_percentile=None,  # irrelevant to this test
        upper_percentile=None,  # irrelevant to this test
        data_format="jsonl",
    )

    with jsonlines.open(f"{tmp_path}/input.jsonl", "w") as writer:
        writer.write_all(
            [
                {
                    "id": 0,
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
                    "id": 1,
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
                    "id": 2,
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
                    "id": 3,
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
                    "id": 4,
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

    processor._get_stats(in_fname=f"{tmp_path}/input", stats_dir=f"{tmp_path}/stats")
    with jsonlines.open(os.path.join(f"{tmp_path}/stats", "stats_diff.jsonl"), "r") as reader:
        stats_diff = [line for line in reader]
    assert stats_diff[0] == {
        "id": 0,
        "num_tokens": 3,
        "num_chars": len("sample_fname\n") + len("sample diff"),
        "num_mods": 1,
    }
    assert stats_diff[1] == {"id": 1, "num_tokens": None, "num_chars": None, "num_mods": None}
    assert stats_diff[2] == {
        "id": 2,
        "num_tokens": 3,
        "num_chars": len("sample_fname\n") + len("sample diff"),
        "num_mods": 1,
    }
    assert stats_diff[3] == {"id": 3, "num_tokens": None, "num_chars": None, "num_mods": None}
    assert stats_diff[4] == {
        "id": 4,
        "num_tokens": 8,
        "num_chars": len("sample_fname\n") + len("-another example of diff\n+this time longer\n"),
        "num_mods": 1,
    }

    with jsonlines.open(os.path.join(f"{tmp_path}/stats", "stats_message.jsonl"), "r") as reader:
        stats_message = [line for line in reader]
    assert stats_message[0] == {"id": 0, "num_tokens": 2, "num_chars": len("sample message")}
    assert stats_message[1] == {"id": 1, "num_tokens": 2, "num_chars": len("sample message")}
    assert stats_message[2] == {"id": 2, "num_tokens": None, "num_chars": None}
    assert stats_message[3] == {"id": 3, "num_tokens": None, "num_chars": None}
    assert stats_message[4] == {
        "id": 4,
        "num_tokens": 7,
        "num_chars": len("another example of message\nthis time longer"),
    }


def test_get_percentiles(tmp_path):
    os.makedirs(f"{tmp_path}/stats")
    processor = OutliersProcessor(
        lower_percentile=None,  # irrelevant to this test
        upper_percentile=None,  # irrelevant to this test
        data_format="jsonl",
    )
    with jsonlines.open(f"{tmp_path}/stats/stats_diff.jsonl", "w") as writer:
        writer.write_all(
            [{"id": i, "num_tokens": i + 1, "num_chars": i + 1, "num_mods": i + 1} for i in range(100)]
            + [{"id": 101, "num_tokens": None, "num_chars": None, "num_mods": None}]
        )

    with jsonlines.open(f"{tmp_path}/stats/stats_message.jsonl", "w") as writer:
        writer.write_all(
            [{"id": i, "num_tokens": i + 100, "num_chars": i + 100} for i in range(100)]
            + [{"id": 101, "num_tokens": None, "num_chars": None}]
        )

    processor._get_percentiles(stats_dir=f"{tmp_path}/stats")

    for key in ["num_tokens", "num_chars", "num_mods"]:
        for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
            assert processor._diff_percentiles[p][key] == np.percentile([i + 1 for i in range(100)], p)

    for key in ["num_tokens", "num_chars"]:
        for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
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
                for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]
            },
            f,
        )

    with open(f"{tmp_path}/stats/message.json", "w") as f:
        json.dump(
            {
                p: {key: np.percentile([i + 1 for i in range(100)], p) for key in ["num_tokens", "num_chars"]}
                for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]
            },
            f,
        )

    processor._read_percentiles(percentile_dir=f"{tmp_path}/stats")

    for key in ["num_tokens", "num_chars", "num_mods"]:
        for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
            assert processor._diff_percentiles[p][key] == np.percentile([i + 100 for i in range(100)], p)

    for key in ["num_tokens", "num_chars"]:
        for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
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
            [{"id": i, "num_tokens": i + 1, "num_chars": i + 1, "num_mods": i + 1} for i in range(5)]
            + [{"id": 101, "num_tokens": None, "num_chars": None, "num_mods": None}]
        )

    with jsonlines.open(f"{tmp_path}/stats/stats_message.jsonl", "w") as writer:
        writer.write_all(
            [{"id": i, "num_tokens": i - 5 + 1, "num_chars": i - 5 + 1} for i in range(5, 10)]
            + [{"id": 102, "num_tokens": None, "num_chars": None}]
        )

    processor._get_ids_to_drop(stats_dir=f"{tmp_path}/stats")

    assert processor._ids_to_drop == {
        0,  # have < 5% percentile tokens in diffs
        1,  # have < 5% percentile characters in diffs
        4,  # have > 95% percentile modified files in diffs
        5,  # have < 5% percentile tokens in messages
        6,  # have < 5% percentile characters in messages
        9,  # have > 95% percentile tokens in messages
        101,
        102,  # have None values
    }
