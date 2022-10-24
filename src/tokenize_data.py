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
        os.makedirs(cfg.paths[key], exist_ok=True)
    os.makedirs(os.path.join(cfg.paths.output_dir, "messages"), exist_ok=True)
    os.makedirs(os.path.join(cfg.paths.output_dir, "diffs"), exist_ok=True)

    for key in ["diff_tokenizer_name_or_path", "msg_tokenizer_name_or_path"]:
        if key in cfg.paths:
            cfg.training_processor[key] = cfg.paths[key]

    os.makedirs(cfg.paths.output_dir, exist_ok=True)

    parts = ["train", "val", "test"]
    if "parts" in cfg:
        parts = cfg.parts

    logging.info("======= Using config =======")
    logging.info(cfg)

    processor = TrainingProcessor(**cfg.training_processor, data_format=cfg.data_format)
    if "tokenize_data" not in cfg or cfg.tokenize_data:
        for part in parts:
            if cfg.only_messages:
                processor.process_single_col(
                    in_fname=os.path.join(cfg.paths.input_dir, f"{part}_final"),
                    output_dir=cfg.paths.output_dir,
                    part=part,
                    line_sep=cfg.line_sep,
                    diffs_or_messages="messages",
                    temp_dir=cfg.paths.temp_dir if "temp_dir" in cfg.paths else None,
                    preprocess_data=cfg.preprocess_data if "preprocess_data" in cfg else True,
                )
            elif cfg.only_diffs:
                processor.process_single_col(
                    in_fname=os.path.join(cfg.paths.input_dir, f"{part}_final"),
                    output_dir=cfg.paths.output_dir,
                    part=part,
                    line_sep=cfg.line_sep,
                    diffs_or_messages="diffs",
                    temp_dir=cfg.paths.temp_dir if "temp_dir" in cfg.paths else None,
                    preprocess_data=cfg.preprocess_data if "preprocess_data" in cfg else True,
                )
            else:
                processor(
                    in_fname=os.path.join(cfg.paths.input_dir, f"{part}_final"),
                    output_dir=cfg.paths.output_dir,
                    part=part,
                    line_sep=cfg.line_sep,
                    temp_dir=cfg.paths.temp_dir if "temp_dir" in cfg.paths else None,
                    preprocess_data=cfg.preprocess_data if "preprocess_data" in cfg else True,
                )
    if "truncate_diffs" in cfg and cfg.truncate_diffs:
        os.makedirs(os.path.join(cfg.paths.output_dir, "diffs", f"{cfg.context_len}"), exist_ok=True)
        for part in parts:
            processor.truncate_diffs(
                in_fname=os.path.join(cfg.paths.output_dir, "diffs", f"{part}.json"),
                output_dir=os.path.join(cfg.paths.output_dir, "diffs", f"{cfg.context_len}"),
                part=part,
                context_len=cfg.context_len,
            )


if __name__ == "__main__":
    main()
