import os
import json
import logging
import hydra

from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from joblib import Parallel, delayed

from src.data_collection.utils import RepoProcessor


def get_logger():
    """
    Workaround for logging with joblib (based on https://github.com/joblib/joblib/issues/1017)
    """
    logger = logging.getLogger("mylogger")
    if len(logger.handlers) == 0:
        logger.setLevel(logging.INFO)
        sh = logging.StreamHandler()
        sh.setFormatter(logging.Formatter("%(asctime)s %(levelname)-8s %(message)s"))
        fh = logging.FileHandler("collect_data.log", mode="a")
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)-8s %(message)s"))
        logger.addHandler(sh)
        logger.addHandler(fh)
    return logger


@hydra.main(config_path="configs", config_name="collect_data_conf")
def main(cfg: DictConfig) -> None:
    for key in cfg.paths:
        cfg.paths[key] = to_absolute_path(cfg.paths[key])

    logging.info("======= Using config =======")
    logging.info(cfg)

    os.makedirs(cfg.paths.temp_clone_dir, exist_ok=True)

    for part in ["train", "val", "test", "val_original", "test_original"]:
        inputs = []
        logging.info(f"Processing {part}")
        for repo in os.listdir(os.path.join(cfg.paths.input_dir, part)):
            with open(os.path.join(cfg.paths.input_dir, part, repo), "r") as infile:
                cur_input = json.load(infile)
                cur_input["repo"] = cur_input["repo"].replace("/", cfg.org_repo_sep)
                os.makedirs(os.path.join(cfg.paths.output_dir, part, cur_input["repo"]), exist_ok=True)
                inputs.append(cur_input)

        rp = RepoProcessor(
            temp_clone_dir=cfg.paths.temp_clone_dir,
            output_dir=os.path.join(cfg.paths.output_dir, part),
            logger_f=get_logger,
            **cfg.repo_processor,
        )

        with Parallel(cfg.num_workers) as pool:
            pool(
                delayed(rp.process_repo)(
                    repo_name=cur_input["repo"],
                    repo_url=cur_input["url"],
                    only_commits=cur_input["hashes"],
                )
                for cur_input in inputs
            )

        rp.unite_files(out_fname=os.path.join(cfg.paths.output_dir, f"{part}.jsonl"))


if __name__ == "__main__":
    main()
