import os
import csv
import gzip
import argparse
import pydriller
import logging
from typing import List
from joblib import Parallel, delayed
from commit_processor import CommitProcessor

logging.basicConfig(
    filename="collect_data.log",
    encoding="utf-8",
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)


def process_repo(repos_dir, commit_data_dir, repo_name, repo_url, file_types):
    """
    Download author, date, diff and message of all .java-related commits
    and save to .jsonl.gz
    """
    os.makedirs(repos_dir, exist_ok=True)
    os.makedirs(os.path.join(commit_data_dir, repo_name.split("/")[0]), exist_ok=True)

    if repo_name.split("/")[1] + ".csv.gz" in os.listdir(os.path.join(commit_data_dir, repo_name.split("/")[0])):
        logging.warning(f" [{repo_name}] Skipping, already processed")
        return

    # do not clone repos that are already cloned
    if repo_url.split("/")[-1].replace(".git", "") in os.listdir(repos_dir):
        logging.warning(f" [{repo_name}] Already cloned")
        repo = pydriller.RepositoryMining(
            f'{repos_dir}/{repo_url.split("/")[-1].replace(".git", "")}',
            only_no_merge=True,
            only_modifications_with_file_types=file_types,
        )
    else:
        try:
            repo = pydriller.RepositoryMining(
                repo_url, clone_repo_to=repos_dir, only_no_merge=True, only_modifications_with_file_types=file_types
            )
        except:  # sometimes git errors can happen during cloning
            logging.error(f"Couldn't clone {repo_url}")
            return

    logging.warning(f" [{repo_name}] Start processing")

    csv_filename = os.path.join(commit_data_dir, repo_name.split("/")[0], repo_name.split("/")[1] + ".csv")
    fieldnames = ["author", "date", "hash", "message", "diff"]
    with open(csv_filename, "w") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

    commits_data = []

    for commit in repo.traverse_commits():
        cur_data = CommitProcessor.process_commit(commit)

        if filter_diff(cur_data["diff"]) and filter_msg(cur_data["message"]):
            commits_data.append(cur_data)

        if len(commits_data) >= 1000:
            logging.warning(f" [{repo_name}] Processed more than 1000 commits, writing to file")
            with open(csv_filename, "a") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writerows(commits_data)
                commits_data = []

    if len(commits_data) > 0:
        logging.warning(f" [{repo_name}] Final writing to file")
        with open(csv_filename, "a") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerows(commits_data)

    with open(csv_filename, "rb") as f_in, gzip.open(csv_filename + ".gz", "wb") as f_out:
        f_out.writelines(f_in)
    os.remove(csv_filename)

    logging.warning(f" [{repo_name}] Finish processing")


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
    parser.add_argument("--repos_dir", type=str, default="temp", help="path to directory to clone repos to")
    parser.add_argument(
        "--commit_data_dir",
        type=str,
        default="extracted_data_csv",
        help="path to directory to save collected commit data",
    )
    parser.add_argument("--repos_urls_file", type=str, default="repos_urls.txt", help="path to file with repos urls")
    parser.add_argument("--repos_names_file", type=str, default="repos_names.txt", help="path to file with repos names")
    parser.add_argument(
        "--file_types",
        type=List[str],
        default=[".py"],
        help="only analyses commits in which at least one modification was done in provided file types",
    )
    args = parser.parse_args()

    with open(args.repos_urls_file, "r") as file:
        repo_urls_list = [line.strip() for line in file.readlines()]

    with open(args.repos_names_file, "r") as file:
        repo_names_list = [line.strip() for line in file.readlines()]

    with Parallel(4) as pool:
        pool(
            delayed(process_repo)(
                repos_dir=args.repos_dir,
                commit_data_dir=args.commit_data_dir,
                repo_name=repo_name,
                repo_url=repo_url,
                file_types=args.file_types,
            )
            for repo_name, repo_url in zip(repo_names_list, repo_urls_list)
        )
