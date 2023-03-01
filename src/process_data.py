import logging
import os
from typing import Dict, Tuple

import hydra
import jsonlines
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

from .processing import (
    DiffProcessor,
    ExactHashProcessor,
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
            os.makedirs(os.path.join(cfg.paths.stats_percentile_dir, part), exist_ok=True)
            processor.prepare(
                input_dir=os.path.join(cfg.paths.input_dir, "raw", part),
                stats_dir=os.path.join(cfg.paths.stats_percentile_dir, part),
                percentile_dir=os.path.join(cfg.paths.stats_percentile_dir, "train"),
                use_cache=False,
                part=part,
            )
            processor(
                input_dir=os.path.join(cfg.paths.input_dir, "raw", part),
                output_dir=os.path.join(cfg.paths.input_dir, "filtered_outliers", part),
                part=part,
            )

    # -----------------------------------
    # -         filter messages         -
    # -----------------------------------
    if cfg.message_processor:
        os.makedirs(os.path.join(cfg.paths.input_dir, "filtered_msgs"), exist_ok=True)
        message_processor = MessageProcessor(
            **cfg.message_processor.args, data_format=cfg.data_format, logger_name="message_processor"
        )
        for part in parts:
            message_processor(
                input_dir=os.path.join(cfg.paths.input_dir, "filtered_outliers", part),
                output_dir=os.path.join(cfg.paths.input_dir, "filtered_msgs", part),
                line_sep=cfg.line_sep,
                part=part,
                replace_patterns=cfg.message_processor.replace_patterns,
            )

    # -----------------------------------
    # -           filter diffs          -
    # -----------------------------------
    if cfg.diff_processor:
        os.makedirs(os.path.join(cfg.paths.input_dir, "filtered_diffs"), exist_ok=True)
        diff_processor = DiffProcessor(
            **cfg.diff_processor.args, data_format=cfg.data_format, logger_name="diff_processor"
        )
        for part in parts:
            diff_processor(
                input_dir=os.path.join(cfg.paths.input_dir, "filtered_msgs", part),
                output_dir=os.path.join(cfg.paths.input_dir, "filtered_diffs", part),
                line_sep=cfg.line_sep,
                part=part,
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
                input_dir=os.path.join(cfg.paths.input_dir, "filtered_diffs", part),
                diff_fname=os.path.join(cfg.paths.deduplication_dir, "raw", f"{part}_diffs.txt"),
                message_fname=os.path.join(cfg.paths.deduplication_dir, "raw", f"{part}_messages.txt"),
                project_id=part_id + 1,
                part=part,
            )

        pre_d_processor.save_map(os.path.join(cfg.paths.metadata_dir, "commits_map.jsonl"))

    # -------------------------------------------
    # - preprocess data into SourcererCC format -
    # -------------------------------------------
    if cfg.exact_hash_processor:
        exact_hash_processor = ExactHashProcessor(
            **cfg.exact_hash_processor.args,
            data_format=cfg.data_format,
            logger_name="exact_hash_processor",
        )

        logging.info(f"Searching for Exact Hash clones in messages")
        exact_hash_processor(
            data_type="messages",
            deduplication_root=cfg.paths.deduplication_dir,
            parts=cfg.parts,
            use_cache=cfg.exact_hash_processor.use_cache,
            use_tokens_hash=cfg.exact_hash_processor.use_tokens_hash,
        )

        logging.info(f"Searching for Exact Hash clones in diffs")
        exact_hash_processor(
            data_type="diffs",
            deduplication_root=cfg.paths.deduplication_dir,
            parts=cfg.parts,
            use_cache=cfg.exact_hash_processor.use_cache,
            use_tokens_hash=cfg.exact_hash_processor.use_tokens_hash,
        )

    # ----------------------------
    # -       drop clones        -
    # ----------------------------
    if cfg.post_deduplication_processor:
        # load id mapping
        with jsonlines.open(os.path.join(cfg.paths.metadata_dir, "commits_map.jsonl"), "r") as reader:
            ids_to_commits_map: Dict[int, Dict[str, str]] = {
                line["id"]: {"repo": line["repo"], "hash": line["hash"]} for line in reader
            }

        for part_id, part in enumerate(parts):
            post_d_processor = PostDeduplicationProcessor(
                **cfg.post_deduplication_processor.args,
                ids_to_commits_map=ids_to_commits_map,
                data_format=cfg.data_format,
                logger_name="postdedupl_processor",
            )
            post_d_processor.prepare(
                inner_part_id=part_id + 1,
                outer_part_ids=[el + 1 for el, _ in enumerate(parts) if el != part_id],
                diff_clones_fname=os.path.join(
                    cfg.paths.deduplication_dir, "results_str", "exact_hash", "diffs", "results.jsonl"
                ),
                msg_clones_fname=os.path.join(
                    cfg.paths.deduplication_dir, "results_str", "exact_hash", "messages", "results.jsonl"
                ),
                only_full_inner_clones=cfg.post_deduplication_processor.only_full_inner_clones,
                identical_clones=cfg.post_deduplication_processor.identical_clones,
                use_exact_hash=True,
                process_inner_clones=(
                    part == "train" if cfg.post_deduplication_processor.only_train_inner_clones else True
                ),
                process_outer_clones=(
                    part == "train" if cfg.post_deduplication_processor.only_train_outer_clones else True
                ),
            )
            post_d_processor(
                input_dir=os.path.join(cfg.paths.input_dir, "filtered_diffs", part),
                output_dir=os.path.join(cfg.paths.input_dir, "no_duplicates", part),
                part=part,
            )

    # ---------------------------
    # -  finalize metadata      -
    # ---------------------------
    if cfg.metadata_processor:
        # load id mapping
        with jsonlines.open(os.path.join(cfg.paths.metadata_dir, "commits_map.jsonl"), "r") as reader:
            ids_to_commits_map: Dict[int, Dict[str, str]] = {  # type: ignore[no-redef]
                line["id"]: {"repo": line["repo"], "hash": line["hash"]} for line in reader
            }

        metadata_processor = MetadataProcessor(
            **cfg.metadata_processor.args,
            data_format=cfg.data_format,
            logger_name="final_processor",
            ids_to_commits_map=ids_to_commits_map,
        )
        for i, part in enumerate(parts):
            logging.info(f"Converting authors in {part}")
            if i == 0:
                metadata_processor.prepare(
                    input_dir=os.path.join(cfg.paths.input_dir, "no_duplicates"),
                    parts=parts,
                    authors_map_fname=os.path.join(cfg.paths.metadata_dir, "authors_map.jsonl"),
                    known_bots_fname=os.path.join(cfg.paths.metadata_dir, "bots.jsonl"),
                    repos_metadata_fname=os.path.join(cfg.paths.metadata_dir, "filtered_ghs_results_25_jan_2023.jsonl"),
                    deduplication_raw_dir=os.path.join(cfg.paths.deduplication_dir, "raw"),
                )
            metadata_processor(
                input_dir=os.path.join(cfg.paths.input_dir, "no_duplicates", part),
                output_dir=os.path.join(cfg.paths.input_dir, "final", part),
                part=part,
                part_id=i + 1,
            )


if __name__ == "__main__":
    main()
