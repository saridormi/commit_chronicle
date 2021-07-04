import os
import argparse
import pydriller
from typing import List
from dpu_utils.utils import save_jsonl_gz
from joblib import Parallel, delayed, cpu_count
from commit_processor import CommitProcessor


def process_repo(repos_dir, commit_data_dir, repo_name, repo_url, file_types):
    """
    Download author, date, diff and message of all .java-related commits
    and save to .jsonl.gz
    """
    os.makedirs(commit_data_dir, exist_ok=True)

    # as repos are processed in parallel, sometimes a combination of many workers and repos with big history
    # led to crush due to memory error. so these lines are for skipping already processed repos
    # in case you have to run this script several times (:
    if repo_name + '.jsonl.gz' in os.listdir(commit_data_dir):
        return

    # do not clone repos that are already cloned
    if repo_name.split('_')[1] in os.listdir(repos_dir):
        repo = pydriller.RepositoryMining(f'{repos_dir}/{repo_name.split("_")[1]}', only_no_merge=True,
                                          only_modifications_with_file_types=file_types)
    else:
        try:
            repo = pydriller.RepositoryMining(repo_url, clone_repo_to=repos_dir, only_no_merge=True,
                                              only_modifications_with_file_types=file_types)
        except:  # sometimes git errors can happen during cloning
            return
        
    commits_data = []
    
    print(f"Processing {repo_name}")
    
    for i, commit in enumerate(repo.traverse_commits()):
        cur_data = CommitProcessor.process_commit(commit)
        
        if filter_diff(cur_data['diff']) and filter_msg(cur_data['message']):
            commits_data.append(cur_data)

        if (i + 1) % 1000 == 0:
            print(f"Processed {i + 1} commits in {repo_name}")
    
    save_jsonl_gz(commits_data, os.path.join(commit_data_dir, f'{repo_name}.jsonl.gz'))    
    print(f"Finished processing {repo_name}")
                          

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='This script collects commit data from provided list of GitHub repos.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--repos_dir', type=str, default='temp',
                        help='path to directory to clone repos to')
    parser.add_argument('--commit_data_dir', type=str, default='extracted_data',
                        help='path to directory to save collected commit data')
    parser.add_argument('--repos_urls_file', type=str, default='repos_urls.txt',
                        help='path to file with repos urls')
    parser.add_argument('--repos_names_file', type=str, default='repos_names.txt',
                        help='path to file with repos names')
    parser.add_argument('--file_types', type=List[str], default=['.java'],
                        help='only analyses commits in which at least one modification was done in provided file types')
    args = parser.parse_args()

    with open(args.repos_urls_file, 'r') as file:
        repo_urls_list = [line.strip() for line in file.readlines()]

    with open(args.repos_names_file, 'r') as file:
        repo_names_list = [line.strip() for line in file.readlines()]

    with Parallel(cpu_count()) as pool:
        pool(delayed(process_repo)(repos_dir=args.repos_dir,
                                   commit_data_dir=args.commit_data_dir,
                                   repo_name=repo_name,
                                   repo_url=repo_url,
                                   file_types=args.file_types)
             for repo_name, repo_url in zip(repo_names_list, repo_urls_list))
