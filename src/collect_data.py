import logging
import os
from datetime import datetime
from typing import Any, Dict

import hydra
import pandas as pd
from hydra.utils import to_absolute_path
from joblib import Parallel, delayed
from omegaconf import DictConfig, OmegaConf

from .collection import RepoProcessor


@hydra.main(config_path="../configs", config_name="collect_data")
def main(cfg: DictConfig) -> None:
    for key in cfg.paths:
        cfg.paths[key] = to_absolute_path(cfg.paths[key])
        os.makedirs(cfg.paths[key], exist_ok=True)

    logging.info("======= Using config =======")
    logging.info(cfg)

    # convert date-related arguments to datetime (format is %d-%m-%Y)
    pydriller_kwargs: Dict[str, Any] = OmegaConf.to_container(cfg.pydriller_kwargs)  # type: ignore[assignment]
    if "since" in pydriller_kwargs:
        assert isinstance(pydriller_kwargs["since"], str)
        pydriller_kwargs["since"] = datetime.strptime(pydriller_kwargs["since"], "%d-%m-%Y")
    if "to" in pydriller_kwargs:
        assert isinstance(pydriller_kwargs["to"], str)
        pydriller_kwargs["to"] = datetime.strptime(pydriller_kwargs["to"], "%d-%m-%Y")

    # process!
    for part in cfg.parts:
        logging.info(f"Processing {part}")

        repos_metadata = pd.read_json(f"{cfg.paths.input_dir}/{part}.jsonl", orient="records", lines=True)
        names = [name.replace("/", "#") for name in repos_metadata["name"].tolist()]
        urls = [url.replace("git://", "https://") for url in repos_metadata["github_url"].tolist()]

        rp = RepoProcessor(
            temp_clone_dir=os.path.join(cfg.paths.temp_clone_dir, part),
            output_dir=os.path.join(cfg.paths.output_dir, "raw", part),
            logger_name="repo_processor",
            data_format=cfg.data_format,
            **cfg.repo_processor,
        )

        with Parallel(cfg.n_workers) as pool:
            pool(
                delayed(rp.process_repo)(repo_name=name, repo_url=url, **pydriller_kwargs)
                for name, url in zip(names, urls)
            )


if __name__ == "__main__":
    main()
