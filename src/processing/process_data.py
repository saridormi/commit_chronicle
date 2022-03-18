import os
import logging
import hydra

from hydra.utils import to_absolute_path
from omegaconf import DictConfig

from src.processing.utils import (
    AuthorProcessor,
    OutliersProcessor,
    PreDeduplicationProcessor,
    PostDeduplicationProcessor,
    MessageProcessor,
    DiffProcessor,
)


@hydra.main(config_path=".", config_name="config")
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
            if not os.path.isdir(os.path.join(cfg.paths.input_dir, part)) and "train" not in part
        ]
    )

    # ---------------------------------
    # -       convert authors         -
    # ---------------------------------
    os.makedirs(os.path.join(cfg.paths.input_dir, "converted_authors"), exist_ok=True)
    processor = AuthorProcessor(**cfg.author_processor, data_format=cfg.data_format, logger_name="author_processor")
    for part in parts:
        logging.info(f"Converting authors in {part}")

        processor(
            in_fname=os.path.join(cfg.paths.input_dir, part),
            out_fname=os.path.join(cfg.paths.input_dir, "converted_authors", part),
            prepare_in_fnames=[os.path.join(cfg.paths.input_dir, part) for part in parts],
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
            percentile_dir = os.path.join(cfg.paths.percentile_dir, "train")
        os.makedirs(os.path.join(cfg.paths.percentile_dir, part), exist_ok=True)

        processor(
            in_fname=os.path.join(cfg.paths.input_dir, "converted_authors", part),
            out_fname=os.path.join(cfg.paths.input_dir, "filtered_outliers", part),
            prepare_n_tokens_dir=os.path.join(cfg.paths.percentile_dir, part),
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
            in_fname=os.path.join(cfg.paths.input_dir, "filtered_outliers", part),
            out_fname=os.path.join(cfg.paths.deduplication_dir, "raw", f"{part}_message.txt"),
            data_col="message",
        )

        logging.info(f"Processing diffs from {part} into SourcererCC format")
        processor(
            in_fname=os.path.join(cfg.paths.input_dir, "filtered_outliers", part),
            out_fname=os.path.join(cfg.paths.deduplication_dir, "raw", f"{part}_diffs.txt"),
            data_col="mods",
        )

    # stop here if there's no clones results
    if not cfg.clones_ready:
        return

    # -----------------------------------
    # -         drop duplicates         -
    # -----------------------------------

    processor = PostDeduplicationProcessor(
        **cfg.post_deduplication_processor, data_format=cfg.data_format, logger_name="postdedupl_processor"
    )
    processor(
        in_fname=os.path.join(cfg.paths.input_dir, "filtered_outliers", "train"),
        out_fname=os.path.join(cfg.paths.input_dir, "filtered_outliers", "train_no_duplicates"),
        prepare_in_path=os.path.join(cfg.paths.input_dir, "filtered_outliers"),
        prepare_diff_clones_fname="results_messages_100_multi.pairs",
        prepare_msg_clones_fname="results_diffs_100_multi.pairs",
        prepare_deduplication_dir=cfg.paths.deduplication_dir,
        prepare_parts=parts,
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
                f"{part}{'_no_duplicates' if part == 'train' else ''}",
            ),
            out_fname=os.path.join(cfg.paths.input_dir, "filtered_msgs", part),
        )

    # -----------------------------------
    # -           filter diffs          -
    # -----------------------------------

    os.makedirs(os.path.join(cfg.paths.input_dir, "filtered_diffs"), exist_ok=True)
    for part in parts:
        processor = DiffProcessor(**cfg.diff_processor, data_format=cfg.data_format, logger_name="diff_processor")
        processor(
            in_fname=os.path.join(cfg.paths.input_dir, "filtered_msgs", part),
            out_fname=os.path.join(cfg.paths.input_dir, "filtered_diffs", part),
        )


if __name__ == "__main__":
    main()
