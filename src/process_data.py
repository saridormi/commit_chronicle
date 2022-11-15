import logging
import os

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

from .processing import (
    DiffProcessor,
    Lexer,
    MessageProcessor,
    MetadataProcessor,
    OutliersProcessor,
    PostDeduplicationProcessor,
    PreDeduplicationProcessor,
)


@hydra.main(config_path="../configs", config_name="process_data")
def main(cfg: DictConfig) -> None:
    for key in cfg.paths:
        cfg.paths[key] = to_absolute_path(cfg.paths[key])
        os.makedirs(cfg.paths[key], exist_ok=True)

    logging.info("======= Using config =======")
    logging.info(cfg)

    parts = cfg.parts
    if parts[0] != "train":
        raise ValueError(
            "Some processing stages require the train part to be passed first (e.g. percentiles are computed on train and then used for other parts). Please make sure that first part name that you pass is equal to `train`"
        )

    # ---------------------------------
    # -         drop outliers         -
    # ---------------------------------
    if cfg.outliers_processor:
        os.makedirs(os.path.join(cfg.paths.input_dir, "filtered_outliers"), exist_ok=True)
        processor = OutliersProcessor(
            **cfg.outliers_processor.args,
            data_format=cfg.data_format,
            logger_name="outliers_processor",
            lower_percentile=cfg.outliers_processor.lower_percentile,
            upper_percentile=cfg.outliers_processor.upper_percentile,
        )
        for part in parts:
            logging.info(f"Dropping outliers from {part}")

            percentile_dir = None
            if part != "train":
                percentile_dir = os.path.join(cfg.paths.stats_percentile_dir, "train")
            os.makedirs(os.path.join(cfg.paths.stats_percentile_dir, part), exist_ok=True)

            processor(
                in_fname=os.path.join(cfg.paths.input_dir, part),
                out_fname=os.path.join(cfg.paths.input_dir, "filtered_outliers", part),
                prepare_stats_dir=os.path.join(cfg.paths.stats_percentile_dir, part),
                prepare_percentile_dir=percentile_dir,
            )

    # -----------------------------------
    # -         filter messages         -
    # -----------------------------------
    if cfg.message_processor:
        os.makedirs(os.path.join(cfg.paths.input_dir, "filtered_msgs"), exist_ok=True)
        for part in parts:
            message_processor = MessageProcessor(
                **cfg.message_processor.args, data_format=cfg.data_format, logger_name="message_processor"
            )
            message_processor(
                in_fname=os.path.join(
                    cfg.paths.input_dir,
                    "filtered_outliers",
                    part,
                ),
                out_fname=os.path.join(cfg.paths.input_dir, "filtered_msgs", part),
                line_sep=cfg.line_sep,
                replace_patterns=cfg.message_processor.replace_patterns,
            )

    # ---------------------------------------
    # - filter diffs – drop unchanged lines -
    # ---------------------------------------
    if cfg.diff_processor:
        os.makedirs(os.path.join(cfg.paths.input_dir, "filtered_diffs"), exist_ok=True)
        for part in parts:
            diff_processor = DiffProcessor(
                **cfg.diff_processor.args, data_format=cfg.data_format, logger_name="diff_processor"
            )
            diff_processor(
                in_fname=os.path.join(cfg.paths.input_dir, "filtered_msgs", part),
                out_fname=os.path.join(cfg.paths.input_dir, "filtered_diffs", part),
            )

    # ------------------------
    # - filter diffs – lexer -
    # ------------------------
    if cfg.lexer:
        os.makedirs(os.path.join(cfg.paths.input_dir, "lexed"), exist_ok=True)
        os.makedirs(os.path.join(cfg.paths.input_dir, "tokenization"), exist_ok=True)
        lexer = Lexer(
            **cfg.lexer.args,
            data_format=cfg.data_format,
            logger_name="lexer",
            upper_percentile=cfg.lexer.upper_percentile,
            line_sep=cfg.line_sep,
        )
        for part in parts:
            os.makedirs(os.path.join(cfg.paths.lexemes_percentile_dir, part), exist_ok=True)

            percentile_dir = None
            if part != "train":
                percentile_dir = os.path.join(cfg.paths.literals_percentile_dir, "train")

            logging.info(f"Lexing {part}")
            lexer(
                in_fname=os.path.join(cfg.paths.input_dir, "filtered_diffs", part),
                out_fname=os.path.join(cfg.paths.input_dir, "lexed", part),
                delimiter_out_fname=os.path.join(cfg.paths.input_dir, "tokenization", part),
                prepare_lexemes_percentile_dir=os.path.join(cfg.paths.lexemes_percentile_dir, part),
                prepare_percentile_dir=percentile_dir,
            )

    # -------------------------------------------
    # - preprocess data into SourcererCC format -
    # -------------------------------------------
    if cfg.pre_deduplication_processor:
        pre_d_processor = PreDeduplicationProcessor(
            **cfg.pre_deduplication_processor.args,
            data_format=cfg.data_format,
            logger_name="prededupl_processor",
            special_tokens=["[LONG]", cfg.line_sep] + list(MessageProcessor.get_special_tokens().values()),
        )
        os.makedirs(os.path.join(cfg.paths.deduplication_dir, "raw"), exist_ok=True)
        for part_id, part in enumerate(parts):
            logging.info(f"Processing messages from {part} into SourcererCC format")
            pre_d_processor(
                in_fname=os.path.join(cfg.paths.input_dir, "lexed", part),
                out_fname=os.path.join(cfg.paths.deduplication_dir, "raw", f"{part}_message.txt"),
                data_col="message",
                project_id=part_id + 1,
                add_data_format=False,
            )

            logging.info(f"Processing diffs from {part} into SourcererCC format")
            pre_d_processor(
                in_fname=os.path.join(cfg.paths.input_dir, "lexed", part),
                out_fname=os.path.join(cfg.paths.deduplication_dir, "raw", f"{part}_diffs.txt"),
                data_col="mods",
                project_id=part_id + 1,
                add_data_format=False,
            )
        pre_d_processor.save_map(os.path.join(cfg.paths.metadata_dir, "ids_map.json"))

    # ----------------------------
    # -       drop clones        -
    # ----------------------------
    if cfg.post_deduplication_processor:
        for part_id, part in enumerate(parts):
            post_d_processor = PostDeduplicationProcessor(
                **cfg.post_deduplication_processor.args, data_format=cfg.data_format, logger_name="postdedupl_processor"
            )

            post_d_processor(
                in_fname=os.path.join(cfg.paths.input_dir, "lexed", part),
                out_fname=os.path.join(cfg.paths.input_dir, "lexed", f"{part}_no_duplicates"),
                prepare_inner_part_id=part_id + 1,
                prepare_outer_part_ids=[el + 1 for el, _ in enumerate(parts) if el != part_id],
                prepare_diff_clones_fname=os.path.join(cfg.paths.deduplication_dir, "results_messages_80.pairs"),
                prepare_msg_clones_fname=os.path.join(cfg.paths.deduplication_dir, "results_diffs_80.pairs"),
                prepare_only_full_inner_clones=cfg.post_deduplication_processor.only_full_inner_clones,
                prepare_identical_clones=cfg.post_deduplication_processor.identical_clones,
                prepare_process_inner_clones=(
                    part == "train" if cfg.post_deduplication_processor.only_train_inner_clones else True
                ),
                prepare_process_outer_clones=(
                    part == "train" if cfg.post_deduplication_processor.only_train_outer_clones else True
                ),
            )
            post_d_processor(
                in_fname=os.path.join(cfg.paths.input_dir, "tokenization", part),
                out_fname=os.path.join(cfg.paths.input_dir, "tokenization", f"{part}_no_duplicates"),
                prepare_is_ready=True,
                prepare_inner_part_id=part_id + 1,
                prepare_outer_part_ids=[el + 1 for el, _ in enumerate(parts) if el != part_id],
                prepare_diff_clones_fname=os.path.join(cfg.paths.deduplication_dir, "results_messages_80.pairs"),
                prepare_msg_clones_fname=os.path.join(cfg.paths.deduplication_dir, "results_diffs_80.pairs"),
                prepare_only_full_inner_clones=cfg.post_deduplication_processor.only_full_inner_clones,
                prepare_identical_clones=cfg.post_deduplication_processor.identical_clones,
                prepare_process_inner_clones=(
                    part == "train" if cfg.post_deduplication_processor.only_train_inner_clones else True
                ),
                prepare_process_outer_clones=(
                    part == "train" if cfg.post_deduplication_processor.only_train_outer_clones else True
                ),
            )

    # ---------------------------
    # -  finalize metadata      -
    # ---------------------------
    if cfg.metadata_processor:
        metadata_processor = MetadataProcessor(
            **cfg.metadata_processor.args, data_format=cfg.data_format, logger_name="final_processor"
        )
        for i, part in enumerate(parts):
            logging.info(f"Converting authors in {part}")

            metadata_processor(
                in_fname=os.path.join(cfg.paths.input_dir, "lexed", f"{part}_no_duplicates"),
                out_fname=os.path.join(cfg.paths.input_dir, "lexed", f"{part}_final"),
                prepare_in_fnames=[os.path.join(cfg.paths.input_dir, "lexed", part) for part in parts],
                prepare_authors_map_fname=os.path.join(cfg.paths.metadata_dir, "authors_mapv1.json"),
                prepare_known_bots_fname=os.path.join(cfg.paths.metadata_dir, "bots.jsonl"),
                prepare_licenses_fname=os.path.join(cfg.paths.metadata_dir, "repo_license_map.json"),
                prepare_is_ready=i > 0,
            )

            metadata_processor(
                in_fname=os.path.join(cfg.paths.input_dir, "tokenization", f"{part}_no_duplicates"),
                out_fname=os.path.join(cfg.paths.input_dir, "tokenization", f"{part}_final"),
                prepare_in_fnames=[os.path.join(cfg.paths.input_dir, "lexed", part) for part in parts],
                prepare_authors_map_fname=os.path.join(cfg.paths.metadata_dir, "authors_mapv1.json"),
                prepare_known_bots_fname=os.path.join(cfg.paths.metadata_dir, "bots.jsonl"),
                prepare_licenses_fname=os.path.join(cfg.paths.metadata_dir, "repo_license_map.json"),
                prepare_is_ready=True,
            )


if __name__ == "__main__":
    main()
