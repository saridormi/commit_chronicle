import os
import logging
from typing import List, Dict, Any, Optional, Callable

import gzip
import jsonlines
from configparser import NoOptionError

import pydriller
from .commit_processor import CommitProcessor


class RepoProcessor:
    def __init__(
        self, temp_clone_dir: str, output_dir: str, chunksize: int, do_gzip: bool, logger_f: Optional[Callable] = None
    ):
        self._temp_clone_dir = temp_clone_dir
        self._output_dir = output_dir
        self._chunksize = chunksize
        self._do_gzip = do_gzip
        self._logger_f = logger_f

    def log(self, msg: str, level: int):
        if self._logger_f:
            logger = self._logger_f()
        else:
            logger = logging.getLogger(name=None)
        logger.log(msg=msg, level=level)

    def _prepare_outfile(self, out_fname: str):
        """
        Do what might be required before saving to chosen output format.

        Currently data is saved as jsonl, and this method simply clears target file.
        """
        open(os.path.join(out_fname), mode="w").close()

    def _append_to_outfile(self, data: List[Dict[str, Any]], out_fname: str):
        """
        Append current data chunk to chosen output format.

        Currently data is saved as jsonl.
        """
        with jsonlines.open(out_fname, mode="a") as writer:
            writer.write_all(data)

    def process_repo(self, repo_name, repo_url, **repo_kwargs):
        """
        Gather commits from given repo via PyDriller and save to csv file.

        Args:
            - repo_name: full repository name, including author/organization
            - repo_url: url to clone the repository from
        """
        out_fname = os.path.join(self._output_dir, repo_name, "commits.jsonl")

        # do not process already processed repos
        if self._do_gzip and "commits.jsonl.gz" in os.listdir(os.path.join(self._output_dir, repo_name)):
            return
        if not self._do_gzip and "commits.jsonl" in os.listdir(os.path.join(self._output_dir, repo_name)):
            return

        # read already cloned repos from disk
        if repo_url.split("/")[-1].replace(".git", "") in os.listdir(self._temp_clone_dir):
            self.log(f"[{repo_name}] Already cloned", 10)
            repo = pydriller.RepositoryMining(
                f'{self._temp_clone_dir}/{repo_url.split("/")[-1].replace(".git", "")}', **repo_kwargs
            )
        else:
            self.log(f"[{repo_name}] Cloning...", 10)
            try:
                repo = pydriller.RepositoryMining(repo_url, clone_repo_to=self._temp_clone_dir, **repo_kwargs)
            except Exception as e:  # sometimes git errors can happen during cloning (e.g. if repo was deleted)
                self.log(f"[{repo_name}] Couldn't clone; {e}", 40)
                return

        self.log(f"[{repo_name}] Start processing", 20)

        self._prepare_outfile(out_fname)

        commits_data = []

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

        if len(commits_data) > 0:
            self.log(f"[{repo_name}] Final writing to file", 10)
            self._append_to_outfile(commits_data, out_fname)

        if self._do_gzip:
            self.log(f"[{repo_name}] Zipping file", 10)
            with open(out_fname, "rb") as f_in, gzip.open(out_fname + ".gz", "wb") as f_out:
                f_out.writelines(f_in)
            os.remove(out_fname)

        self.log(f"[{repo_name}] Finish processing", 20)
