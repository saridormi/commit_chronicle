import json
import logging
import os

import hydra
from hydra.utils import to_absolute_path
from joblib import Parallel, delayed
from omegaconf import DictConfig

from .collection import RepoProcessor


@hydra.main(config_path="../configs", config_name="collect_data")
def main(cfg: DictConfig) -> None:
    for key in cfg.paths:
        cfg.paths[key] = to_absolute_path(cfg.paths[key])

    parts = ["train"] + sorted(
        [
            part
            for part in os.listdir(cfg.paths.input_dir)
            if os.path.isdir(os.path.join(cfg.paths.input_dir, part)) and "train" not in part
        ]
    )

    logging.info("======= Using config =======")
    logging.info(cfg)

    os.makedirs(cfg.paths.temp_clone_dir, exist_ok=True)

    for part in parts:
        inputs = []
        logging.info(f"Processing {part}")
        for repo in os.listdir(os.path.join(cfg.paths.input_dir, part)):
            with open(os.path.join(cfg.paths.input_dir, part, repo), "r") as infile:
                cur_input = json.load(infile)
                cur_input["repo"] = cur_input["repo"].replace("/", cfg.org_repo_sep)
                os.makedirs(os.path.join(cfg.paths.output_dir, "raw", part, cur_input["repo"]), exist_ok=True)
                inputs.append(cur_input)

        rp = RepoProcessor(
            temp_clone_dir=cfg.paths.temp_clone_dir,
            output_dir=os.path.join(cfg.paths.output_dir, "raw", part),
            logger_name="repo_processor",
            data_format=cfg.data_format,
            **cfg.repo_processor,
        )

        with Parallel(cfg.n_workers) as pool:
            pool(
                delayed(rp.process_repo)(
                    repo_name=cur_input["repo"],
                    repo_url=cur_input["url"],
                    only_commits=cur_input["hashes"],
                )
                for cur_input in inputs
            )

        rp.unite_files(out_fname=os.path.join(cfg.paths.output_dir, part), org_repo_sep=cfg.org_repo_sep)


if __name__ == "__main__":
    main()
