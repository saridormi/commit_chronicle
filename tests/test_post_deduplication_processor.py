import pytest

from src.processing import PostDeduplicationProcessor
from src.utils import CloneGroup


@pytest.fixture
def default_processor():
    return PostDeduplicationProcessor(
        data_format="jsonl", ids_to_commits_map={i: {"repo": f"repo{i}", "hash": f"hash{i}"} for i in range(1000)}
    )


def test_get_outer_clones(default_processor, tmp_path):
    clones = [
        "1,2, 2,1",
        "1,3, 2,2",
        "1,4, 2,2",
        "1,4, 2,1",
        "1,6, 2,5",
        "2,2, 1,1",
        "2,3, 1,2",
        "2,5, 1,4",
        "2,5, 2,6",
        "1,3, 1,4",
    ]
    clones_fname = tmp_path / "clones.txt"
    with open(clones_fname, "w") as f:
        f.writelines(line + "\n" for line in clones)
    outer_clones = default_processor._get_outer_clones(str(clones_fname), inner_part_id=1, outer_part_ids=[2])
    assert outer_clones == [
        CloneGroup(clone_root=(2, 1), clones={(1, 2), (1, 4)}),
        CloneGroup(clone_root=(2, 2), clones={(1, 1), (1, 3), (1, 4)}),
        CloneGroup(clone_root=(2, 3), clones={(1, 2)}),
        CloneGroup(clone_root=(2, 5), clones={(1, 4), (1, 6)}),
    ]


def test_get_inner_clones_identical(default_processor, tmp_path):
    clones = [
        "1,2, 1,1",
        "1,3, 1,1",
        "1,3, 1,2",
        "1,5, 1,1",
        "1,5, 1,2",
        "1,5, 1,3",
        "1,7, 1,6",
        "1,8, 1,6",
        "1,8, 1,7",
        "2,9, 2,5",
        "1,9, 2,6",
    ]
    clones_fname = tmp_path / "clones.txt"
    with open(clones_fname, "w") as f:
        f.writelines(line + "\n" for line in clones)
    inner_clones = default_processor._get_inner_clones_identical(str(clones_fname), part_id=1)
    assert inner_clones == [
        CloneGroup(clone_root=None, clones={(1, 6), (1, 7), (1, 8)}),
        CloneGroup(clone_root=None, clones={(1, 1), (1, 2), (1, 3), (1, 5)}),
    ]


def test_get_inner_clones_similar(default_processor, tmp_path):
    clones = [
        "1,3, 1,1",
        "1,3, 1,2",
        "1,5, 1,1",
        "1,5, 1,2",
        "1,5, 1,3",
        "1,7, 1,6",
        "1,8, 1,6",
        "1,8, 1,7",
        "1,9, 1,8",
        "2,9, 2,5",
        "1,9, 2,6",
    ]
    clones_fname = tmp_path / "clones.txt"
    with open(clones_fname, "w") as f:
        f.writelines(line + "\n" for line in clones)
    inner_clones = default_processor._get_inner_clones_similar(str(clones_fname), part_id=1)
    assert sorted(inner_clones, key=lambda x: x.clone_root[1]) == [
        CloneGroup(clone_root=(1, 1), clones={(1, 3), (1, 5)}),
        CloneGroup(clone_root=(1, 2), clones={(1, 3), (1, 5)}),
        CloneGroup(clone_root=(1, 3), clones={(1, 1), (1, 2), (1, 5)}),
        CloneGroup(clone_root=(1, 5), clones={(1, 1), (1, 2), (1, 3)}),
        CloneGroup(clone_root=(1, 6), clones={(1, 7), (1, 8)}),
        CloneGroup(clone_root=(1, 7), clones={(1, 6), (1, 8)}),
        CloneGroup(clone_root=(1, 8), clones={(1, 6), (1, 7), (1, 9)}),
        CloneGroup(clone_root=(1, 9), clones={(1, 8)}),
    ]


def test_get_full_inner_clones_identical(default_processor):
    msg_clones = [
        CloneGroup(clone_root=None, clones={(1, 1), (1, 2), (1, 3)}),
        CloneGroup(clone_root=None, clones={(1, 5), (1, 7)}),
        CloneGroup(clone_root=None, clones={(1, 15), (1, 17)}),
    ]
    diff_clones = [
        CloneGroup(clone_root=None, clones={(1, 1), (1, 2)}),
        CloneGroup(clone_root=None, clones={(1, 5), (1, 9), (1, 10)}),
        CloneGroup(clone_root=None, clones={(1, 8), (1, 11), (1, 12)}),
    ]

    full_clones = default_processor._get_full_inner_clones_identical(msg_clones, diff_clones)

    assert full_clones == [CloneGroup(clone_root=None, clones={(1, 1), (1, 2)})]


def test_get_full_inner_clones_similar(default_processor):
    msg_clones = [
        CloneGroup(clone_root=(1, 1), clones={(1, 5), (1, 6)}),
        CloneGroup(clone_root=(1, 2), clones={(1, 5)}),
        CloneGroup(clone_root=(1, 3), clones={(1, 5)}),
        CloneGroup(clone_root=(1, 4), clones={(1, 6)}),
        CloneGroup(clone_root=(1, 5), clones={(1, 1), (1, 2), (1, 3)}),
        CloneGroup(clone_root=(1, 6), clones={(1, 1), (1, 4), (1, 12)}),
        CloneGroup(clone_root=(1, 12), clones={(1, 6)}),
    ]
    diff_clones = [
        CloneGroup(clone_root=(1, 1), clones={(1, 5), (1, 6)}),
        CloneGroup(clone_root=(1, 3), clones={(1, 17)}),
        CloneGroup(clone_root=(1, 5), clones={(1, 1)}),
        CloneGroup(clone_root=(1, 6), clones={(1, 1)}),
        CloneGroup(clone_root=(1, 17), clones={(1, 3)}),
    ]

    full_clones = default_processor._get_full_inner_clones_similar(msg_clones, diff_clones)

    assert full_clones == [
        CloneGroup(clone_root=(1, 1), clones={(1, 5), (1, 6)}),
        CloneGroup(clone_root=(1, 5), clones={(1, 1)}),
        CloneGroup(clone_root=(1, 6), clones={(1, 1)}),
    ]


def test_get_outer_ids_to_drop(default_processor, tmp_path):
    msg_clones = ["1,2, 2,1", "1,3, 2,2", "1,4, 2,2", "1,4, 2,1", "1,7, 2,1"]
    msg_clones_fname = tmp_path / "msg_clones.txt"
    with open(msg_clones_fname, "w") as f:
        f.writelines(line + "\n" for line in msg_clones)
    msg_outer_clones = default_processor._get_outer_clones(str(msg_clones_fname), inner_part_id=1, outer_part_ids=[2])
    assert msg_outer_clones == [
        CloneGroup(clone_root=(2, 1), clones={(1, 2), (1, 4), (1, 7)}),
        CloneGroup(clone_root=(2, 2), clones={(1, 3), (1, 4)}),
    ]

    diff_clones = ["1,2, 2,1", "1,3, 2,2", "1,4, 2,2", "1,5, 2,3", "1,6, 2,3"]
    diff_clones_fname = tmp_path / "diff_clones.txt"
    with open(diff_clones_fname, "w") as f:
        f.writelines(line + "\n" for line in diff_clones)
    diff_outer_clones = default_processor._get_outer_clones(str(diff_clones_fname), inner_part_id=1, outer_part_ids=[2])
    assert diff_outer_clones == [
        CloneGroup(clone_root=(2, 1), clones={(1, 2)}),
        CloneGroup(clone_root=(2, 2), clones={(1, 3), (1, 4)}),
        CloneGroup(clone_root=(2, 3), clones={(1, 5), (1, 6)}),
    ]

    default_processor._get_outer_ids_to_drop(
        diff_clones_fname=str(diff_clones_fname),
        msg_clones_fname=str(msg_clones_fname),
        inner_part_id=1,
        outer_part_ids=[2],
    )

    assert default_processor._outer_clones_to_drop == {f"repo{i}": {f"hash{i}"} for i in [2, 3, 4, 5, 6, 7]}


def test_get_inner_ids_to_drop_identical(default_processor, tmp_path):
    msg_clones = [
        "1,2, 1,1",
        "1,3, 1,1",
        "1,3, 1,2",
        "1,5, 1,1",
        "1,5, 1,2",
        "1,5, 1,3",
        "1,7, 1,6",
        "1,8, 1,6",
        "1,8, 1,7",
        "2,9, 2,5",
        "1,9, 2,6",
    ]
    msg_clones_fname = tmp_path / "msg_clones.txt"
    with open(msg_clones_fname, "w") as f:
        f.writelines(line + "\n" for line in msg_clones)
    msg_inner_clones = default_processor._get_inner_clones_identical(str(msg_clones_fname), part_id=1)
    assert msg_inner_clones == [
        CloneGroup(clone_root=None, clones={(1, 6), (1, 7), (1, 8)}),
        CloneGroup(clone_root=None, clones={(1, 1), (1, 2), (1, 3), (1, 5)}),
    ]

    diff_clones = ["1,10, 1,9", "1,11, 1,10"]
    diff_clones_fname = tmp_path / "diff_clones.txt"
    with open(diff_clones_fname, "w") as f:
        f.writelines(line + "\n" for line in diff_clones)
    diff_inner_clones = default_processor._get_inner_clones_identical(str(diff_clones_fname), part_id=1)
    assert diff_inner_clones == [CloneGroup(clone_root=None, clones={(1, 9), (1, 10), (1, 11)})]

    default_processor._get_inner_ids_to_drop(
        msg_clones_fname=str(msg_clones_fname),
        diff_clones_fname=str(diff_clones_fname),
        inner_part_id=1,
        only_full_inner_clones=False,
        identical_clones=True,
    )

    assert default_processor._inner_clones_to_drop == {
        f"repo{i}": {f"hash{i}"}
        for i in [
            1,
            2,
            3,
            5,
            6,
            7,
            8,  # message clones
            9,
            10,
            11,  # diff clones
        ]
    }


def test_get_inner_ids_to_drop_identical_only_full(default_processor, tmp_path):
    msg_clones = [
        "1,2, 1,1",
        "1,3, 1,1",
        "1,3, 1,2",
        "1,5, 1,1",
        "1,5, 1,2",
        "1,5, 1,3",
        "1,7, 1,6",
        "1,8, 1,6",
        "1,8, 1,7",
        "2,9, 2,5",
        "1,9, 2,6",
    ]
    msg_clones_fname = tmp_path / "msg_clones.txt"
    with open(msg_clones_fname, "w") as f:
        f.writelines(line + "\n" for line in msg_clones)
    # sanity check: msg clones should be as expected
    msg_inner_clones = default_processor._get_inner_clones_identical(str(msg_clones_fname), part_id=1)
    assert msg_inner_clones == [
        CloneGroup(clone_root=None, clones={(1, 6), (1, 7), (1, 8)}),
        CloneGroup(clone_root=None, clones={(1, 1), (1, 2), (1, 3), (1, 5)}),
    ]

    diff_clones = ["1,3,  1,2", "1,10, 1,9", "1,11, 1,10"]
    diff_clones_fname = tmp_path / "diff_clones.txt"
    with open(diff_clones_fname, "w") as f:
        f.writelines(line + "\n" for line in diff_clones)
    # sanity check: diff clones should be as expected
    diff_inner_clones = default_processor._get_inner_clones_identical(str(diff_clones_fname), part_id=1)
    assert diff_inner_clones == [
        CloneGroup(clone_root=None, clones={(1, 9), (1, 10), (1, 11)}),
        CloneGroup(clone_root=None, clones={(1, 2), (1, 3)}),
    ]

    default_processor._get_inner_ids_to_drop(
        msg_clones_fname=str(msg_clones_fname),
        diff_clones_fname=str(diff_clones_fname),
        inner_part_id=1,
        only_full_inner_clones=True,
        identical_clones=True,
    )

    assert default_processor._inner_clones_to_drop == {"repo2": {"hash2"}, "repo3": {"hash3"}}


def test_get_inner_ids_to_drop_identical_only_full_no_full_clones(default_processor, tmp_path):
    msg_clones = [
        "1,2, 1,1",
        "1,3, 1,1",
        "1,3, 1,2",
        "1,5, 1,1",
        "1,5, 1,2",
        "1,5, 1,3",
        "1,7, 1,6",
        "1,8, 1,6",
        "1,8, 1,7",
        "2,9, 2,5",
        "1,9, 2,6",
    ]
    msg_clones_fname = tmp_path / "msg_clones.txt"
    with open(msg_clones_fname, "w") as f:
        f.writelines(line + "\n" for line in msg_clones)
    msg_inner_clones = default_processor._get_inner_clones_identical(str(msg_clones_fname), part_id=1)
    assert msg_inner_clones == [
        CloneGroup(clone_root=None, clones={(1, 6), (1, 7), (1, 8)}),
        CloneGroup(clone_root=None, clones={(1, 1), (1, 2), (1, 3), (1, 5)}),
    ]

    diff_clones = ["1,10, 1,9", "1,11, 1,10"]
    diff_clones_fname = tmp_path / "diff_clones.txt"
    with open(diff_clones_fname, "w") as f:
        f.writelines(line + "\n" for line in diff_clones)
    diff_inner_clones = default_processor._get_inner_clones_identical(str(diff_clones_fname), part_id=1)
    assert diff_inner_clones == [CloneGroup(clone_root=None, clones={(1, 9), (1, 10), (1, 11)})]

    default_processor._get_inner_ids_to_drop(
        msg_clones_fname=str(msg_clones_fname),
        diff_clones_fname=str(diff_clones_fname),
        inner_part_id=1,
        only_full_inner_clones=True,
        identical_clones=True,
    )

    assert default_processor._inner_clones_to_drop == {}


def test_get_inner_ids_to_drop_similar(default_processor, tmp_path):
    msg_clones = ["1,3, 1,2", "1,5, 1,2", "1,5, 1,3", "1,7, 1,6", "1,8, 1,7", "2,9, 2,5", "1,9, 2,6"]
    msg_clones_fname = tmp_path / "msg_clones.txt"
    with open(msg_clones_fname, "w") as f:
        f.writelines(line + "\n" for line in msg_clones)
    msg_inner_clones = default_processor._get_inner_clones_similar(str(msg_clones_fname), part_id=1)
    assert sorted(msg_inner_clones, key=lambda x: x.clone_root[1]) == [
        CloneGroup(clone_root=(1, 2), clones={(1, 3), (1, 5)}),
        CloneGroup(clone_root=(1, 3), clones={(1, 2), (1, 5)}),
        CloneGroup(clone_root=(1, 5), clones={(1, 2), (1, 3)}),
        CloneGroup(clone_root=(1, 6), clones={(1, 7)}),
        CloneGroup(clone_root=(1, 7), clones={(1, 6), (1, 8)}),
        CloneGroup(clone_root=(1, 8), clones={(1, 7)}),
    ]

    diff_clones = ["1,10, 1,9", "1,11, 1,10"]
    diff_clones_fname = tmp_path / "diff_clones.txt"
    with open(diff_clones_fname, "w") as f:
        f.writelines(line + "\n" for line in diff_clones)
    diff_inner_clones = default_processor._get_inner_clones_similar(str(diff_clones_fname), part_id=1)
    assert sorted(diff_inner_clones, key=lambda x: x.clone_root[1]) == [
        CloneGroup(clone_root=(1, 9), clones={(1, 10)}),
        CloneGroup(clone_root=(1, 10), clones={(1, 9), (1, 11)}),
        CloneGroup(clone_root=(1, 11), clones={(1, 10)}),
    ]

    default_processor._get_inner_ids_to_drop(
        msg_clones_fname=str(msg_clones_fname),
        diff_clones_fname=str(diff_clones_fname),
        inner_part_id=1,
        only_full_inner_clones=False,
        identical_clones=False,
    )

    assert default_processor._inner_clones_to_drop == {f"repo{i}": {f"hash{i}"} for i in [2, 3, 5, 6, 7, 8, 9, 10, 11]}


def test_get_inner_ids_to_drop_similar_only_full(default_processor, tmp_path):
    msg_clones = ["1,3, 1,2", "1,5, 1,2", "1,5, 1,3", "1,7, 1,6", "1,8, 1,7", "2,9, 2,5", "1,9, 2,6"]
    msg_clones_fname = tmp_path / "msg_clones.txt"
    with open(msg_clones_fname, "w") as f:
        f.writelines(line + "\n" for line in msg_clones)
    msg_inner_clones = default_processor._get_inner_clones_similar(str(msg_clones_fname), part_id=1)
    assert sorted(msg_inner_clones, key=lambda x: x.clone_root[1]) == [
        CloneGroup(clone_root=(1, 2), clones={(1, 3), (1, 5)}),
        CloneGroup(clone_root=(1, 3), clones={(1, 2), (1, 5)}),
        CloneGroup(clone_root=(1, 5), clones={(1, 2), (1, 3)}),
        CloneGroup(clone_root=(1, 6), clones={(1, 7)}),
        CloneGroup(clone_root=(1, 7), clones={(1, 6), (1, 8)}),
        CloneGroup(clone_root=(1, 8), clones={(1, 7)}),
    ]

    diff_clones = ["1,3, 1,2", "1,5, 1,3", "1,9, 1,6", "2,9, 2,5", "1,9, 2,6"]
    diff_clones_fname = tmp_path / "diff_clones.txt"
    with open(diff_clones_fname, "w") as f:
        f.writelines(line + "\n" for line in diff_clones)
    diff_inner_clones = default_processor._get_inner_clones_similar(str(diff_clones_fname), part_id=1)
    assert sorted(diff_inner_clones, key=lambda x: x.clone_root[1]) == [
        CloneGroup(clone_root=(1, 2), clones={(1, 3)}),
        CloneGroup(clone_root=(1, 3), clones={(1, 2), (1, 5)}),
        CloneGroup(clone_root=(1, 5), clones={(1, 3)}),
        CloneGroup(clone_root=(1, 6), clones={(1, 9)}),
        CloneGroup(clone_root=(1, 9), clones={(1, 6)}),
    ]

    default_processor._get_inner_ids_to_drop(
        msg_clones_fname=str(msg_clones_fname),
        diff_clones_fname=str(diff_clones_fname),
        inner_part_id=1,
        only_full_inner_clones=True,
        identical_clones=False,
    )

    assert default_processor._inner_clones_to_drop == {f"repo{i}": {f"hash{i}"} for i in [2, 3, 5]}


def test_get_inner_ids_to_drop_similar_only_full_no_full_clones(default_processor, tmp_path):
    msg_clones = ["1,3, 1,2", "1,5, 1,2", "1,5, 1,3", "1,7, 1,6", "1,8, 1,7", "2,9, 2,5", "1,9, 2,6"]
    msg_clones_fname = tmp_path / "msg_clones.txt"
    with open(msg_clones_fname, "w") as f:
        f.writelines(line + "\n" for line in msg_clones)
    msg_inner_clones = default_processor._get_inner_clones_similar(str(msg_clones_fname), part_id=1)
    assert sorted(msg_inner_clones, key=lambda x: x.clone_root[1]) == [
        CloneGroup(clone_root=(1, 2), clones={(1, 3), (1, 5)}),
        CloneGroup(clone_root=(1, 3), clones={(1, 2), (1, 5)}),
        CloneGroup(clone_root=(1, 5), clones={(1, 2), (1, 3)}),
        CloneGroup(clone_root=(1, 6), clones={(1, 7)}),
        CloneGroup(clone_root=(1, 7), clones={(1, 6), (1, 8)}),
        CloneGroup(clone_root=(1, 8), clones={(1, 7)}),
    ]

    diff_clones = ["1,10, 1,9", "1,11, 1,10"]
    diff_clones_fname = tmp_path / "diff_clones.txt"
    with open(diff_clones_fname, "w") as f:
        f.writelines(line + "\n" for line in diff_clones)
    diff_inner_clones = default_processor._get_inner_clones_similar(str(diff_clones_fname), part_id=1)
    assert sorted(diff_inner_clones, key=lambda x: x.clone_root[1]) == [
        CloneGroup(clone_root=(1, 9), clones={(1, 10)}),
        CloneGroup(clone_root=(1, 10), clones={(1, 9), (1, 11)}),
        CloneGroup(clone_root=(1, 11), clones={(1, 10)}),
    ]

    default_processor._get_inner_ids_to_drop(
        msg_clones_fname=str(msg_clones_fname),
        diff_clones_fname=str(diff_clones_fname),
        inner_part_id=1,
        only_full_inner_clones=True,
        identical_clones=False,
    )

    assert default_processor._inner_clones_to_drop == {}
