import logging
import os

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

from .processing import (
    DiffProcessor,
    Lexer,
    MessageProcessor,
    OutliersProcessor,
    PreDeduplicationProcessor,
)


@hydra.main(config_path="../configs", config_name="process_data")
def main(cfg: DictConfig) -> None:
    for key in cfg.paths:
        cfg.paths[key] = to_absolute_path(cfg.paths[key])
        os.makedirs(cfg.paths[key], exist_ok=True)

    logging.info("======= Using config =======")
    logging.info(cfg)

    parts = ["train"] + sorted(
        [
            part.split(".")[0]
            for part in os.listdir(cfg.paths.input_dir)
            if not os.path.isdir(os.path.join(cfg.paths.input_dir, part))
            and "train" not in part
            and "final" not in part
        ]
    )

    # ---------------------------------
    # -         drop outliers         -
    # ---------------------------------

    os.makedirs(os.path.join(cfg.paths.input_dir, "filtered_outliers"), exist_ok=True)
    processor = OutliersProcessor(
        **cfg.outliers_processor, data_format=cfg.data_format, logger_name="outliers_processor"
    )
    for part in parts:
        logging.info(f"Dropping outliers from {part}")

        percentile_dir = None
        if part != "train":
            percentile_dir = os.path.join(cfg.paths.tokens_percentile_dir, "train")
        os.makedirs(os.path.join(cfg.paths.tokens_percentile_dir, part), exist_ok=True)

        processor(
            in_fname=os.path.join(cfg.paths.input_dir, part),
            out_fname=os.path.join(cfg.paths.input_dir, "filtered_outliers", part),
            prepare_n_tokens_dir=os.path.join(cfg.paths.tokens_percentile_dir, part),
            prepare_percentile_dir=percentile_dir,
        )

    # -----------------------------------
    # -         filter messages         -
    # -----------------------------------

    os.makedirs(os.path.join(cfg.paths.input_dir, "filtered_msgs"), exist_ok=True)
    for part in parts:
        processor = MessageProcessor(
            **cfg.message_processor, data_format=cfg.data_format, logger_name="message_processor"
        )
        processor(
            in_fname=os.path.join(
                cfg.paths.input_dir,
                "filtered_outliers",
                part,
            ),
            out_fname=os.path.join(cfg.paths.input_dir, "filtered_msgs", part),
            line_sep=cfg.line_sep,
        )

    # ---------------------------------------
    # - filter diffs – drop unchanged lines -
    # ---------------------------------------

    os.makedirs(os.path.join(cfg.paths.input_dir, "filtered_diffs"), exist_ok=True)
    for part in parts:
        processor = DiffProcessor(**cfg.diff_processor, data_format=cfg.data_format, logger_name="diff_processor")
        processor(
            in_fname=os.path.join(cfg.paths.input_dir, "filtered_msgs", part),
            out_fname=os.path.join(cfg.paths.input_dir, "filtered_diffs", part),
        )

    # ------------------------
    # - filter diffs – lexer -
    # ------------------------

    os.makedirs(os.path.join(cfg.paths.input_dir, "lexed"), exist_ok=True)
    os.makedirs(os.path.join(cfg.paths.input_dir, "tokenization"), exist_ok=True)
    lexer = Lexer(**cfg.lexer, data_format=cfg.data_format, logger_name="lexer", line_sep=cfg.line_sep)
    for part in parts:
        os.makedirs(os.path.join(cfg.paths.literals_percentile_dir, part), exist_ok=True)

        percentile_dir = None
        if part != "train":
            percentile_dir = os.path.join(cfg.paths.literals_percentile_dir, "train")

        logging.info(f"Lexing {part}")
        lexer(
            in_fname=os.path.join(cfg.paths.input_dir, "filtered_diffs", part),
            out_fname=os.path.join(cfg.paths.input_dir, "lexed", part),
            delimiter_out_fname=os.path.join(cfg.paths.input_dir, "tokenization", part),
            prepare_literals_len_dir=os.path.join(cfg.paths.literals_percentile_dir, part),
            prepare_percentile_dir=percentile_dir,
        )

    # -------------------------------------------
    # - preprocess data into SourcererCC format -
    # -------------------------------------------

    os.makedirs(os.path.join(cfg.paths.deduplication_dir, "raw"), exist_ok=True)
    for part_id, part in enumerate(parts):

        processor = PreDeduplicationProcessor(
            **cfg.pre_deduplication_processor,
            project_id=part_id + 1,
            data_format=cfg.data_format,
            logger_name="prededupl_processor",
        )

        logging.info(f"Processing messages from {part} into SourcererCC format")
        processor(
            in_fname=os.path.join(cfg.paths.input_dir, "lexed", part),
            out_fname=os.path.join(cfg.paths.deduplication_dir, "raw", f"{part}_message.txt"),
            data_col="message",
            add_data_format=False,
        )

        logging.info(f"Processing diffs from {part} into SourcererCC format")
        processor(
            in_fname=os.path.join(cfg.paths.input_dir, "lexed", part),
            out_fname=os.path.join(cfg.paths.deduplication_dir, "raw", f"{part}_diffs.txt"),
            data_col="mods",
            add_data_format=False,
        )


if __name__ == "__main__":
    main()
