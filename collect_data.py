import os
import json
import csv
import gzip
import argparse
import pydriller
import logging
from typing import List
from joblib import Parallel, delayed
from configparser import NoOptionError
from commit_processor import CommitProcessor

logging.basicConfig(
    filename="collect_data.log",
    encoding="utf-8",
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)


def process_repo(temp_clone_dir, output_dir, full_repo_name, repo_url, hashes):
    """
    Download author, date, diff and message of all .java-related commits
    and save to .jsonl.gz
    """
    org_name, repo_name = full_repo_name.split("/", 1)
    os.makedirs(temp_clone_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, org_name, repo_name), exist_ok=True)

    if "commits.csv.gz" in os.listdir(os.path.join(output_dir, org_name, repo_name)):
        return

    # do not clone repos that are already cloned
    if repo_url.split("/")[-1].replace(".git", "") in os.listdir(temp_clone_dir):
        logging.warning(f"[{repo_name}] Already cloned")
        repo = pydriller.RepositoryMining(
            f'{temp_clone_dir}/{repo_url.split("/")[-1].replace(".git", "")}', only_no_merge=True, only_commits=hashes
        )
    else:
        logging.warning(f"[{repo_name}] Cloning...")
        try:
            repo = pydriller.RepositoryMining(
                repo_url, clone_repo_to=temp_clone_dir, only_no_merge=True, only_commits=hashes
            )
        except:  # sometimes git errors can happen during cloning
            logging.error(f"Couldn't clone {repo_url}")
            return

    logging.warning(f"[{repo_name}] Start processing; {len(hashes)} commits to go")

    csv_filename = os.path.join(output_dir, org_name, repo_name, "commits.csv")
    fieldnames = ["author", "date", "hash", "message", "diff"]
    with open(csv_filename, "w") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

    commits_data = []

    for commit in repo.traverse_commits():
        try:
            cur_data = CommitProcessor.process_commit(commit)
        except AttributeError:
            logging.error(f"AttributeError with {org_name}/{repo_name}")
            continue
        except NoOptionError:
            logging.error(f"NoOptionError with {org_name}/{repo_name}")
            continue

        if filter_diff(cur_data["diff"]) and filter_msg(cur_data["message"]):
            commits_data.append(cur_data)

        if len(commits_data) >= 1000:
            logging.warning(f"[{repo_name}] Processed more than 1000 commits, writing to file")
            with open(csv_filename, "a") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writerows(commits_data)
                commits_data = []

    if len(commits_data) > 0:
        logging.warning(f"[{repo_name}] Final writing to file")
        with open(csv_filename, "a") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerows(commits_data)

    with open(csv_filename, "rb") as f_in, gzip.open(csv_filename + ".gz", "wb") as f_out:
        f_out.writelines(f_in)
    os.remove(csv_filename)

    logging.warning(f"[{repo_name}] Finish processing")


def filter_diff(diff: List[str], min_len=1) -> bool:
    if len(diff) == 0:
        return False
    if sum(len(x.split()) for x in diff) < min_len:
        return False
    return True


def filter_msg(msg: str, min_len=1) -> bool:
    if len(msg.split()) < min_len:
        return False
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="This script collects commit data from provided list of GitHub repos.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--temp_clone_dir", type=str, default="temp", help="path to directory to clone repos to")
    parser.add_argument("--output_dir", type=str, help="path to directory to save collected commit data")

    parser.add_argument("--input_root_dir", type=str, help="path to root folder with repos info")

    args = parser.parse_args()

    inputs = []
    for repo in os.listdir(args.input_root_dir):
        with open(os.path.join(args.input_root_dir, repo), "r") as infile:
            inputs.append(json.load(infile))

    with Parallel(8) as pool:
        pool(
            delayed(process_repo)(
                temp_clone_dir=args.temp_clone_dir,
                output_dir=args.output_dir,
                full_repo_name=cur_input["repo"],
                repo_url=cur_input["url"],
                hashes=cur_input["hashes"],
            )
            for cur_input in inputs
        )
