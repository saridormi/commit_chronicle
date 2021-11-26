import os
import logging
from typing import Optional, Callable, Dict, Any

import gzip
from configparser import NoOptionError
from tqdm import tqdm

from pydriller import Modification, Commit, RepositoryMining

from ..base_utils import FileProcessor


class CommitProcessor:
    @staticmethod
    def get_info_from_modification(m: Modification) -> Dict[str, str]:
        return {
            "change_type": str(m.change_type).split(".")[1],
            "old_path": m.old_path,
            "new_path": m.new_path,
            "diff": m.diff,
        }

    @staticmethod
    def process_commit(commit: Commit) -> Dict[str, Any]:
        """
        Return following information about commit:
        - author name & email
        - timestamp
        - hash
        - message
        - for each modified file:
            - which change was made (e.g. adding a new file, deleting a file, modifying an existing file)
            - old_path (relevant if file was deleted/renamed/copied)
            - new_path (relevant if file was added/renamed/copied)
            - diff
        """
        res = {
            "author": (commit.author.name, commit.author.email),
            "date": commit.author_date.strftime("%d.%m.%Y %H:%M:%S"),
            "hash": commit.hash,
            "message": commit.msg,
            "mods": [],
        }

        for m in commit.modifications:
            res["mods"].append(CommitProcessor.get_info_from_modification(m))
        return res


class RepoProcessor(FileProcessor):
    def __init__(self, temp_clone_dir: str, output_dir: str, chunksize: int, logger_f: Optional[Callable] = None):
        super(RepoProcessor, self).__init__(chunksize)
        self._temp_clone_dir = temp_clone_dir
        self._output_dir = output_dir
        self._logger_f = logger_f

    def log(self, msg: str, level: int):
        if self._logger_f:
            logger = self._logger_f()
        else:
            logger = logging.getLogger(name=None)
        logger.log(msg=msg, level=level)

    def process_repo(self, repo_name, repo_url, **repo_kwargs):
        """
        Gather commits from given repo via PyDriller and save to csv file.

        Args:
            - repo_name: full repository name, including author/organization
            - repo_url: url to clone the repository from
        """
        out_fname = os.path.join(self._output_dir, repo_name, "commits.jsonl")

        # do not process already processed repos
        if "commits.jsonl.gz" in os.listdir(os.path.join(self._output_dir, repo_name)):
            return

        # read already cloned repos from disk
        if repo_url.split("/")[-1].replace(".git", "") in os.listdir(self._temp_clone_dir):
            self.log(f"[{repo_name}] Already cloned", 10)
            repo = RepositoryMining(
                f'{self._temp_clone_dir}/{repo_url.split("/")[-1].replace(".git", "")}', **repo_kwargs
            )
        else:
            repo = RepositoryMining(repo_url, clone_repo_to=self._temp_clone_dir, **repo_kwargs)

        self.log(f"[{repo_name}] Start processing", 20)

        self._prepare_outfile(out_fname)

        commits_data = []
        try:
            for commit in repo.traverse_commits():
                try:
                    cur_data = CommitProcessor.process_commit(commit)
                except (AttributeError, NoOptionError) as e:
                    self.log(f"[{repo_name}] {e} with {commit.hash}", 40)
                    continue

                commits_data.append(cur_data)

                if len(commits_data) >= self._chunksize:
                    self.log(f"[{repo_name}] Processed more than {self._chunksize} commits, writing to file", 10)
                    self._append_to_outfile(commits_data, out_fname)
                    commits_data = []
        except Exception as e:  # sometimes random errors can happen during cloning (e.g. if repo was deleted)
            self.log(f"[{repo_name}] Couldn't clone; {e}", 40)
            return

        if len(commits_data) > 0:
            self.log(f"[{repo_name}] Final writing to file", 10)
            self._append_to_outfile(commits_data, out_fname)

        self.log(f"[{repo_name}] Zipping file", 10)
        with open(out_fname, "rb") as f_in, gzip.open(out_fname + ".gz", "wb") as f_out:
            f_out.writelines(f_in)
        os.remove(out_fname)

        self.log(f"[{repo_name}] Finish processing", 20)

    def unite_files(self, out_fname: str):
        """
        For better parallelism and faster data collection, commits from each repo are saved to its own file.
        That might be inconvenient to process, so this method allows to unite all these files into one,
         adding an unique id and repo name as features.
        """
        self._prepare_outfile(out_fname)

        cur_idx = 0
        for repo_name in tqdm(os.listdir(self._output_dir), desc=f"Processing commits from each repo", leave=False):
            # read data in chunks
            reader = self._read_input(os.path.join(self._output_dir, repo_name, "commits.jsonl.gz"), compression="gzip")
            cur_len = 0
            try:
                for i, chunk in enumerate(reader):
                    # aggregate № examples so that each example from every repo has an unique id
                    chunk["id"] = chunk.index
                    chunk["id"] += cur_idx
                    chunk["repo"] = repo_name.replace("#", "/")

                    self._append_to_outfile(chunk.to_dict(orient="records"), out_fname)

                    cur_len += chunk.shape[0]

                cur_idx += cur_len
            except ValueError as e:
                self.log(f"[{repo_name}] Couldn't read; {e}", 40)
