import gzip
import os
from configparser import NoOptionError
from typing import Any, Optional

from pydriller import RepositoryMining
from tqdm import tqdm

from ..utils import JsonlManager, get_logger
from .commit_processor import CommitProcessor


class RepoProcessor:
    """Mines commit information from given repository.

    Args:
        temp_clone_dir: Directory to clone git repositories to.
        output_dir: Directory to save mined data to.
        data_format: In which format mined data is saved.
        chunksize: Number of examples to proccess at once (data is read in chunks). Optional, default value is 1000.
        logger_name: Name of logger for this class. Optional, default value is None.
    """

    def __init__(
        self,
        temp_clone_dir: str,
        output_dir: str,
        chunksize: int,
        data_format: str,
        logger_name: Optional[str] = None,
    ):
        if data_format == "jsonl":
            self._data_manager = JsonlManager()
        else:
            raise ValueError("Given data format is not supported.")
        self.data_format = data_format
        self._chunksize = chunksize
        self._logger_name = logger_name

        self._temp_clone_dir = temp_clone_dir
        self._output_dir = output_dir

    @property
    def logger(self):
        return get_logger(self._logger_name)

    def process_repo(self, repo_name: str, repo_url: str, **repo_kwargs) -> None:
        """Mines commits from given repository.

        Args:
            repo_name: Full repository name, including author/organization.
            repo_url: A valid url to remote repository.
            **repo_kwargs: Arbitrary keyword arguments, will be passed to `pydriller.RepositoryMining`.
        """
        output_dir = os.path.join(self._output_dir, repo_name)
        os.makedirs(output_dir, exist_ok=True)
        out_fname = os.path.join(output_dir, "commits")

        # sometimes I have to launch collection several times due to memory errors, so:
        # 1. do not process already processed repos
        if f"commits.{self.data_format}.gz" in os.listdir(output_dir):
            return

        # 2. read already cloned repos from disk
        if repo_url.split("/")[-1].replace(".git", "") in os.listdir(self._temp_clone_dir):
            self.logger.debug(f"[{repo_name}] Already cloned")
            repo = RepositoryMining(
                f'{self._temp_clone_dir}/{repo_url.split("/")[-1].replace(".git", "")}', **repo_kwargs
            )
        else:
            repo = RepositoryMining(repo_url, clone_repo_to=self._temp_clone_dir, **repo_kwargs)

        self.logger.info(f"[{repo_name}] Start processing")
        self._data_manager.prepare_outfile(out_fname)

        commits_data = []
        try:
            for commit in repo.traverse_commits():
                try:
                    cur_data = CommitProcessor.process_commit(commit)
                except (AttributeError, NoOptionError) as e:
                    self.logger.error(f"[{repo_name}] raised {e} when processing {commit.hash}")
                    continue

                commits_data.append(cur_data)

                if len(commits_data) >= self._chunksize:
                    self.logger.debug(f"[{repo_name}] Processed more than {self._chunksize} commits, writing to file")
                    self._data_manager.append_to_outfile(commits_data, out_fname)
                    commits_data = []
        except Exception as e:  # sometimes random errors can happen during cloning (e.g. if repo was deleted)
            self.logger.error(f"[{repo_name}] Couldn't clone: raised exception {e}  (url: {repo_url})")
            return

        if len(commits_data) > 0:
            self.logger.debug(f"[{repo_name}] Final writing to file")
            self._data_manager.append_to_outfile(commits_data, out_fname)

        self.logger.debug(f"[{repo_name}] Zipping file")
        with open(f"{out_fname}.{self.data_format}", "rb") as f_in, gzip.open(
            f"{out_fname}.{self.data_format}.gz", "wb"
        ) as f_out:
            f_out.writelines(f_in)
        os.remove(f"{out_fname}.{self.data_format}")

        self.logger.info(f"[{repo_name}] Finish processing")

    def unite_files(self, out_fname: str, org_repo_sep: str) -> None:
        """Unites separate repositories files, add unique ids, repositories names and licences types as features.

        For faster data collection, initially commits from each repo are saved to its own file.

        Args:
            out_fname: Path to resulting single file.
            org_repo_sep: Delimiter used instead of '/' in full repository name.
        """
        self._data_manager.prepare_outfile(out_fname)

        cur_idx = 0
        for repo_name in tqdm(os.listdir(self._output_dir), desc=f"Processing commits from each repo", leave=False):
            # read data in chunks
            reader = self._data_manager.read_input(
                os.path.join(self._output_dir, repo_name, f"commits.{self.data_format}.gz"),
                compression="gzip",
                add_data_format=False,
                chunksize=self._chunksize,
            )
            cur_len = 0
            try:
                for i, chunk in enumerate(reader):
                    # aggregate № examples so that each example from every repo has an unique id
                    chunk["id"] = chunk.index
                    chunk["id"] += cur_idx
                    chunk["repo"] = repo_name.replace(org_repo_sep, "/")

                    self._data_manager.append_to_outfile(chunk, out_fname)

                    cur_len += chunk.shape[0]

                cur_idx += cur_len
            except ValueError as e:
                self.logger.error(f"[{repo_name}] Couldn't read; {e}")
