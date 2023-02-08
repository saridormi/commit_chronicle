import gzip
import json
import logging
from datetime import datetime

from src.collection import RepoProcessor


def test_my_repo(tmp_path, caplog):
    caplog.set_level(logging.INFO)

    processor = RepoProcessor(
        temp_clone_dir=str(tmp_path),
        output_dir=str(tmp_path),
        chunksize=1,
        data_format="jsonl",
        max_lines=10,
    )

    processor.process_repo(
        repo_name="saridormi#commits_dataset",
        repo_url="https://github.com/saridormi/commits_dataset",
        only_no_merge=True,
        skip_whitespaces=True,
        since=datetime.strptime("01-01-2017", "%d-%m-%Y"),
        to=datetime.strptime("07-02-2023", "%d-%m-%Y"),
    )

    commits = []
    with gzip.open(f"{tmp_path}/saridormi#commits_dataset/commits.jsonl.gz", "rb") as f:
        for line in f:
            commits.append(json.loads(line))

    assert len(commits) == 10
    assert all(
        all(key in commit for key in ["author", "date", "timezone", "hash", "message", "mods"]) for commit in commits
    )
    assert [commit["hash"] for commit in commits] == [
        "3aa8f37e171384d8d38aecf9f8d6337fd6061105",
        "d466699b50dc9e6dd501d35f12c12a6c7e10e239",
        "6dbbab4364ef3567c51b8c69283d3fdb6d73ca26",
        "a7fb3b64184f0af5b08285cce14b9139baa94049",
        "f18c59a8038762be08aebdad49fd54e92fdffc1e",
        "10ee9dccf73c5ed9f4449990f0ef67404eef4166",
        "6dbd7000e8cd63dd1afe2371512d0cb1246b106e",
        "723111e261d2af9cf932ab5db36ae9d99db40e57",
        "d5b22ca5683ca56aeefc9d3994485a5197e830ea",
        "4d1a1bcb80fd39a919d080d629960880af5b9fb9",
    ]
    assert caplog.records[-2].levelname == "INFO"
    assert caplog.records[-2].message == "[saridormi#commits_dataset] 10 commits were processed"


def test_name_collision(tmp_path):
    processor = RepoProcessor(
        temp_clone_dir=str(tmp_path),
        output_dir=str(tmp_path),
        chunksize=1,
        data_format="jsonl",
        max_lines=10000,
    )

    processor.process_repo(
        repo_name="huggingface#datasets",
        repo_url="https://github.com/huggingface/datasets",
        single="8a72676689a4a3fb466cc5077884446c7302e605",
    )

    processor.process_repo(
        repo_name="tensorflow#datasets",
        repo_url="https://github.com/tensorflow/datasets",
        single="cebb04d4e66f4393190e53846c47c4d0a00e3123",
    )

    hf_commits = []
    with gzip.open(f"{tmp_path}/huggingface#datasets/commits.jsonl.gz", "rb") as f:
        for line in f:
            hf_commits.append(json.loads(line))
    assert len(hf_commits) == 1
    assert hf_commits[0]["hash"] == "8a72676689a4a3fb466cc5077884446c7302e605"

    tf_commits = []
    with gzip.open(f"{tmp_path}/tensorflow#datasets/commits.jsonl.gz", "rb") as f:
        for line in f:
            tf_commits.append(json.loads(line))
    assert len(tf_commits) == 1
    assert tf_commits[0]["hash"] == "cebb04d4e66f4393190e53846c47c4d0a00e3123"


def test_logging(tmp_path, caplog):
    processor = RepoProcessor(
        temp_clone_dir=str(tmp_path),
        output_dir=str(tmp_path),
        chunksize=1,
        max_lines=10,
        data_format="jsonl",
    )

    # this repo is old: last commit was made on May 18, 2016
    processor.process_repo(
        repo_name="netflix#asgard",
        repo_url="https://github.com/netflix/asgard",
        only_no_merge=True,
        skip_whitespaces=True,
        since=datetime.strptime("01-01-2017", "%d-%m-%Y"),
        to=datetime.strptime("07-02-2023", "%d-%m-%Y"),  # just in case it suddenly comes alive
    )
    commits = []
    with gzip.open(f"{tmp_path}/netflix#asgard/commits.jsonl.gz", "rb") as f:
        for line in f:
            commits.append(json.loads(line))
    assert len(commits) == 0
    assert caplog.records[0].levelname == "WARNING"
    assert caplog.records[0].message == "[netflix#asgard] No commits were processed"

    # this repo is old: has only one commit made since 2017
    # but this repo seems to be following structure from https://github.com/ishepard/pydriller/issues/66
    processor.process_repo(
        repo_name="DCPUTeam#DCPUToolchain",
        repo_url="https://github.com/DCPUTeam/DCPUToolchain",
        only_no_merge=True,
        skip_whitespaces=True,
        since=datetime.strptime("01-01-2017", "%d-%m-%Y"),
        to=datetime.strptime("07-02-2023", "%d-%m-%Y"),  # just in case it suddenly comes alive
    )
    commits = []
    with gzip.open(f"{tmp_path}/DCPUTeam#DCPUToolchain/commits.jsonl.gz", "rb") as f:
        for line in f:
            commits.append(json.loads(line))
    assert len(commits) == 0
    assert caplog.records[1].levelname == "ERROR"
    assert (
        caplog.records[1].message
        == "[DCPUTeam#DCPUToolchain] Caught exception when processing 1882aadbcbd4226064e4d8be9e7178ba771491d8"
    )

    assert caplog.records[2].levelname == "WARNING"
    assert caplog.records[2].message == "[DCPUTeam#DCPUToolchain] No commits were processed"

    # this repo seems to be following structure from https://github.com/ishepard/pydriller/issues/66
    processor._max_lines = 10000
    processor.process_repo(
        repo_name="squirrel#squirrel.mac",
        repo_url="https://github.com/squirrel/squirrel.mac",
        only_no_merge=True,
        skip_whitespaces=True,
        since=datetime.strptime("01-01-2017", "%d-%m-%Y"),
        to=datetime.strptime("07-02-2023", "%d-%m-%Y"),  # just in case it suddenly comes alive
    )
    commits = []
    with gzip.open(f"{tmp_path}/squirrel#squirrel.mac/commits.jsonl.gz", "rb") as f:
        for line in f:
            commits.append(json.loads(line))
    assert len(commits) == 0
    assert caplog.records[-1].levelname == "WARNING"
    assert caplog.records[-1].message == "[squirrel#squirrel.mac] No commits were processed"
