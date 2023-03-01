import logging
import os
from typing import Optional

import hydra
from omegaconf import DictConfig
from tqdm import tqdm

from src.utils import JsonlManager, get_logger


class FileMerger:
    def __init__(
        self,
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

    @property
    def logger(self):
        return get_logger(self._logger_name)

    def __call__(self, input_dir: str, out_fname: str) -> None:
        """Merges commits from all repositories into a single file.

        Args:
            input_dir: Path to root input directory.
            out_fname: Path to resulting single file.
        """
        self.logger.info(f"Start merging files from {input_dir}")
        self._data_manager.prepare_outfile(out_fname)

        for repo_name in tqdm(os.listdir(input_dir), desc=f"Processing repositories from {input_dir}"):
            # read data in chunks
            reader = self._data_manager.read_input(
                os.path.join(input_dir, repo_name, f"commits.{self.data_format}.gz"),
                compression="gzip",
                add_data_format=False,
                chunksize=self._chunksize,
            )
            try:
                for chunk in tqdm(reader, desc=f"Processing commits from {repo_name}", leave=False):
                    chunk["repo"] = repo_name.replace("#", "/")
                    self._data_manager.append_to_outfile(chunk, out_fname)
            except ValueError:
                self.logger.exception(f"[{repo_name}] Caught exception when processing")
        self.logger.info(f"Finish merging files from {input_dir}")


@hydra.main(config_path="../configs", config_name="merge_files")
def main(cfg: DictConfig) -> None:
    for key in cfg.paths:
        cfg.paths[key] = hydra.utils.to_absolute_path(cfg.paths[key])
        os.makedirs(cfg.paths[key], exist_ok=True)

    logging.info("======= Using config =======")
    logging.info(cfg)

    parts = cfg.parts

    file_merger = FileMerger(chunksize=cfg.file_merger.chunksize, data_format=cfg.data_format)
    for part in parts:
        file_merger(
            input_dir=os.path.join(cfg.paths.input_dir, cfg.file_merger.input_stage, part),
            out_fname=os.path.join(cfg.paths.input_dir, part),
        )


if __name__ == "__main__":
    main()
