import os
from configparser import NoOptionError
from typing import Optional

from git import GitCommandError
from pydriller import RepositoryMining

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
        max_lines: Optional[int] = None,
    ):
        if data_format == "jsonl":
            self._data_manager = JsonlManager()
        else:
            raise ValueError("Given data format is not supported.")
        self.data_format = data_format
        self._chunksize = chunksize
        self._logger_name = logger_name

        self._max_lines = max_lines
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
        os.makedirs(os.path.join(self._temp_clone_dir, repo_name), exist_ok=True)
        out_fname = os.path.join(output_dir, "commits")

        # sometimes I have to launch collection several times due to memory errors, so:
        # 1. do not process already processed repos
        if f"commits.{self.data_format}.gz" in os.listdir(output_dir):
            return

        # 2. read already cloned repos from disk
        cloned_repo_name = repo_url.split("/")[-1].replace(".git", "")
        if cloned_repo_name in os.listdir(os.path.join(self._temp_clone_dir, repo_name)):
            self.logger.info(f"[{repo_name}] Already cloned")
            repo = RepositoryMining(f"{self._temp_clone_dir}/{repo_name}/{cloned_repo_name}", **repo_kwargs)
        else:
            repo = RepositoryMining(
                repo_url, clone_repo_to=os.path.join(self._temp_clone_dir, repo_name), **repo_kwargs
            )

        self.logger.info(f"[{repo_name}] Start processing")
        self._data_manager.prepare_outfile(out_fname)

        commits_data = []
        total_num_commits = 0
        try:
            for commit in repo.traverse_commits():
                try:
                    if self._max_lines and commit.lines > self._max_lines:
                        self.logger.warning(
                            f"[{repo_name}] Skiping {commit.hash}, because it changed more than {self._max_lines} lines"
                        )
                        continue
                    cur_data = CommitProcessor.process_commit(commit)
                except (AttributeError, NoOptionError):
                    self.logger.exception(f"[{repo_name}] Caught exception when processing {commit.hash}")
                    continue

                commits_data.append(cur_data)

                if len(commits_data) >= self._chunksize:
                    self.logger.debug(f"[{repo_name}] Processed more than {self._chunksize} commits, writing to file")
                    self._data_manager.append_to_outfile(commits_data, out_fname)
                    total_num_commits += len(commits_data)
                    commits_data = []

        except GitCommandError:  # sometimes random errors can happen (e.g. if repo was deleted)
            self.logger.exception(f"[{repo_name}] Caught exception when first traversing commits")
            return

        if len(commits_data) > 0:
            self.logger.debug(f"[{repo_name}] Final writing to file")
            self._data_manager.append_to_outfile(commits_data, out_fname)
            total_num_commits += len(commits_data)

        if total_num_commits == 0:
            self.logger.warning(f"[{repo_name}] No commits were processed")
        else:
            self.logger.info(f"[{repo_name}] {total_num_commits} commits were processed")

        self.logger.debug(f"[{repo_name}] Zipping file")
        self._data_manager.zip_file(out_fname)
        self.logger.info(f"[{repo_name}] Finish processing")
