import os
import logging
import hydra

from hydra.utils import to_absolute_path
from omegaconf import DictConfig

from src.processing.utils import (
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

    # ---------------------------------
    # -         drop outliers         -
    # ---------------------------------

    os.makedirs(os.path.join(cfg.paths.input_dir, "filtered_outliers"), exist_ok=True)
    processor = OutliersProcessor(**cfg.outliers_processor.args)
    for part in ["train", "val", "test", "val_original", "test_original"]:
        if cfg.outliers_processor.run_anyway or f"{part}.jsonl" not in os.listdir(
            os.path.join(cfg.paths.input_dir, "filtered_outliers")
        ):
            logging.info(f"Dropping outliers from {part}")

            percentile_dir = None
            if part != "train":
                percentile_dir = os.path.join(cfg.paths.percentile_dir, "train")

            os.makedirs(os.path.join(cfg.paths.percentile_dir, part), exist_ok=True)
            processor(
                in_fname=os.path.join(cfg.paths.input_dir, f"{part}.jsonl"),
                out_fname=os.path.join(cfg.paths.input_dir, "filtered_outliers", f"{part}.jsonl"),
                prepare_n_tokens_dir=os.path.join(cfg.paths.percentile_dir, part),
                prepare_percentile_dir=percentile_dir,
            )

    # -------------------------------------------
    # - preprocess data into SourcererCC format -
    # -------------------------------------------

    for part_id, part in enumerate(["train", "val", "test", "val_original", "test_original"]):
        processor = PreDeduplicationProcessor(**cfg.pre_deduplication_processor.args, project_id=part_id + 1)
        if cfg.pre_deduplication_processor.run_anyway or f"{part}_message.txt" not in os.listdir(
            os.path.join(cfg.paths.deduplication_dir, "raw")
        ):
            logging.info(f"Processing messages from {part} into SourcererCC format")
            processor(
                in_fname=os.path.join(cfg.paths.input_dir, "filtered_outliers", f"{part}.jsonl"),
                out_fname=os.path.join(cfg.paths.deduplication_dir, "raw", f"{part}_message.txt"),
                data_col="message",
            )

        if cfg.pre_deduplication_processor.run_anyway or f"{part}_diffs.txt" not in os.listdir(
            os.path.join(cfg.paths.deduplication_dir, "raw")
        ):
            logging.info(f"Processing diffs from {part} into SourcererCC format")
            processor(
                in_fname=os.path.join(cfg.paths.input_dir, "filtered_outliers", f"{part}.jsonl"),
                out_fname=os.path.join(cfg.paths.deduplication_dir, "raw", f"{part}_diffs.txt"),
                data_col="mods",
            )

    # stop here if there's no clones results
    if not cfg.clones_ready:
        return

    # -----------------------------------
    # -         drop duplicates         -
    # -----------------------------------

    if cfg.post_deduplication_processor.run_anyway or "train_no_duplicates.jsonl" not in os.listdir(
        os.path.join(cfg.paths.input_dir, "filtered_outliers")
    ):
        processor = PostDeduplicationProcessor(**cfg.post_deduplication_processor.args)
        processor(
            in_fname=os.path.join(cfg.paths.input_dir, "filtered_outliers", "train.jsonl"),
            out_fname=os.path.join(cfg.paths.input_dir, "filtered_outliers", "train_no_duplicates.jsonl"),
            prepare_in_path=os.path.join(cfg.paths.input_dir, "filtered_outliers"),
            prepare_diff_clones_fname="results_messages_100_multi.pairs",
            prepare_msg_clones_fname="results_diffs_100_multi.pairs",
            prepare_deduplication_dir=cfg.paths.deduplication_dir,
        )

    # -----------------------------------
    # -         filter messages         -
    # -----------------------------------

    os.makedirs(os.path.join(cfg.paths.input_dir, "filtered_msgs"), exist_ok=True)
    for part in ["train", "val", "test", "val_original", "test_original"]:
        if cfg.message_processor.run_anyway or f"{part}.jsonl" not in os.listdir(
            os.path.join(cfg.paths.input_dir, "filtered_msgs")
        ):
            processor = MessageProcessor(**cfg.message_processor.args)
            processor(
                in_fname=os.path.join(
                    cfg.paths.input_dir,
                    "filtered_outliers",
                    f"{part}{'_no_duplicates' if part == 'train' else ''}.jsonl",
                ),
                out_fname=os.path.join(cfg.paths.input_dir, "filtered_msgs", f"{part}.jsonl"),
            )

    # -----------------------------------
    # -           filter diffs          -
    # -----------------------------------

    os.makedirs(os.path.join(cfg.paths.input_dir, "filtered_diffs"), exist_ok=True)
    for part in ["train", "val", "test", "val_original", "test_original"]:
        if cfg.diff_processor.run_anyway or f"{part}.jsonl" not in os.listdir(
            os.path.join(cfg.paths.input_dir, "filtered_diffs")
        ):
            processor = DiffProcessor(**cfg.diff_processor.args)
            processor(
                in_fname=os.path.join(cfg.paths.input_dir, "filtered_msgs", f"{part}.jsonl"),
                out_fname=os.path.join(cfg.paths.input_dir, "filtered_diffs", f"{part}.jsonl"),
            )


if __name__ == "__main__":
    main()
