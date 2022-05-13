import logging
import os

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

from .tokenization import TrainingProcessor


@hydra.main(config_path="../configs", config_name="tokenize_data")
def main(cfg: DictConfig) -> None:
    for key in cfg.paths:
        cfg.paths[key] = to_absolute_path(cfg.paths[key])

    for key in ["diff_tokenizer_name_or_path", "msg_tokenizer_name_or_path"]:
        if ".json" in cfg.training_processor[key]:
            cfg.training_processor[key] = to_absolute_path(cfg.training_processor[key])

    os.makedirs(cfg.paths.output_dir, exist_ok=True)

    parts = ["train"] + sorted(
        [
            part.split(".")[0]
            for part in os.listdir(cfg.paths.input_dir)
            if not os.path.isdir(os.path.join(cfg.paths.input_dir, part))
            and "train" not in part
            and "final" not in part
        ]
    )

    logging.info("======= Using config =======")
    logging.info(cfg)

    processor = TrainingProcessor(**cfg.training_processor, data_format=cfg.data_format)
    for part in parts:
        processor(
            in_fname=os.path.join(cfg.paths.input_dir, "tokenization", f"{part}_final"),
            output_dir=cfg.paths.output_dir,
            part=part,
        )


if __name__ == "__main__":
    main()
